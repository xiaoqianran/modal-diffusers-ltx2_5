import pytest
from pydantic import ValidationError

from ltx25.director import (
    DirectorRuntime,
    ExecutionPlan,
    LTX_ENGINE,
    QWEN_ENGINE,
    RuntimeSnapshot,
    build_execution_plan,
    qwen_output_size,
)
from ltx25.schemas import GenerateRequest


def runtime(*engines, graph=True, captures=1):
    return RuntimeSnapshot(
        available_engines=frozenset(engines),
        ltx_cuda_graph_enabled=graph,
        ltx_cuda_graph_max_captures=captures,
    )


def test_auto_pure_t2i_compiles_to_qwen_plan():
    request = GenerateRequest(mode="t2i", prompt="cinematic city", width=512, height=512)

    plan = build_execution_plan(request, runtime(LTX_ENGINE, QWEN_ENGINE))

    assert plan.requested_engine == "auto"
    assert plan.engine == QWEN_ENGINE
    assert plan.fallback_engine == LTX_ENGINE
    assert plan.fallback_applied is False
    assert plan.reason == "auto:pure_t2i"
    assert plan.resource_class == "qwen_verified_1024"
    assert plan.acceleration == "qwen_default"


def test_auto_t2i_falls_back_during_planning_when_qwen_is_unavailable():
    request = GenerateRequest(mode="t2i", prompt="cinematic city", width=512, height=512)

    plan = build_execution_plan(request, runtime(LTX_ENGINE))

    assert plan.engine == LTX_ENGINE
    assert plan.fallback_engine == LTX_ENGINE
    assert plan.fallback_applied is True
    assert plan.reason == "auto:pure_t2i:qwen_unavailable"
    assert plan.acceleration == "ltx_cuda_graph_eligible"


def test_auto_video_and_lora_compile_to_ltx():
    video = GenerateRequest(mode="t2av", prompt="cinematic city")
    styled = GenerateRequest(
        mode="t2i",
        prompt="cinematic city",
        loras=[{"id": "style.safetensors", "strength": 1.0}],
    )
    snapshot = runtime(LTX_ENGINE, QWEN_ENGINE)

    video_plan = build_execution_plan(video, snapshot)
    styled_plan = build_execution_plan(styled, snapshot)

    assert video_plan.engine == LTX_ENGINE
    assert video_plan.reason == "auto:ltx_capability"
    assert video_plan.acceleration == "ltx_cuda_graph_eligible"
    assert styled_plan.engine == LTX_ENGINE
    assert styled_plan.acceleration == "ltx_eager_lora_safe"


def test_explicit_engine_never_silently_falls_back():
    qwen = GenerateRequest(mode="t2i", engine="qwen", prompt="cinematic city")
    ltx = GenerateRequest(mode="t2i", engine="ltx", prompt="cinematic city")

    qwen_plan = build_execution_plan(qwen, runtime(LTX_ENGINE))
    ltx_plan = build_execution_plan(ltx, runtime(LTX_ENGINE, QWEN_ENGINE))

    assert qwen_plan.engine == QWEN_ENGINE
    assert qwen_plan.fallback_engine is None
    assert qwen_plan.fallback_applied is False
    assert qwen_plan.reason == "explicit:qwen:unavailable"
    assert ltx_plan.engine == LTX_ENGINE
    assert ltx_plan.reason == "explicit:ltx"


def test_qwen_verified_resource_envelopes_are_descriptive_not_predictive():
    medium = GenerateRequest(
        mode="t2i", prompt="city", width=768, height=512, upscale=True
    )
    large = GenerateRequest(
        mode="t2i", prompt="city", width=960, height=544, upscale=True
    )
    snapshot = runtime(LTX_ENGINE, QWEN_ENGINE)

    assert build_execution_plan(medium, snapshot).resource_class == "qwen_verified_1536x1024"
    assert build_execution_plan(large, snapshot).resource_class == "qwen_unverified_large"


def test_graph_disabled_compiles_ltx_to_eager():
    request = GenerateRequest(mode="t2av", prompt="cinematic city")
    plan = build_execution_plan(request, runtime(LTX_ENGINE, graph=False, captures=0))
    assert plan.acceleration == "ltx_eager"


def test_qwen_t2i_preserves_final_size_contract():
    request = GenerateRequest(
        mode="t2i",
        engine="qwen",
        prompt="cinematic city",
        width=512,
        height=512,
        steps=40,
    )

    assert request.upscale is True
    assert qwen_output_size(request) == (1024, 1024)


def test_qwen_rejects_non_image_modes():
    with pytest.raises(ValidationError, match="t2i and image_edit"):
        GenerateRequest(mode="t2av", engine="qwen", prompt="cinematic city")


def test_qwen_image_edit_accepts_ordered_multi_reference_and_routes_to_qwen():
    conditions = [
        {"asset_id": f"{index:032x}", "kind": "image", "index": index}
        for index in range(1, 11)
    ]
    request = GenerateRequest(
        mode="image_edit",
        prompt="preserve the subject and change the background",
        conditions=conditions,
        width=1024,
        height=1024,
        transparent_background=True,
        qwen_true_cfg_scale=4.0,
    )

    plan = build_execution_plan(request, runtime(LTX_ENGINE, QWEN_ENGINE))

    assert plan.engine == QWEN_ENGINE
    assert plan.fallback_engine is None
    assert plan.reason == "auto:qwen_image_edit"
    assert request.transparent_background is True
    assert request.qwen_true_cfg_scale == 4.0
    assert len(request.conditions) == 10
    assert qwen_output_size(request) == (2048, 2048)


def test_qwen_large_t2i_does_not_offer_incompatible_ltx_fallback():
    request = GenerateRequest(mode="t2i", prompt="poster", width=1024, height=1024)

    plan = build_execution_plan(request, runtime(LTX_ENGINE))

    assert plan.engine == QWEN_ENGINE
    assert plan.fallback_engine is None
    assert plan.reason == "auto:pure_t2i:no_available_engine"


def test_qwen_rejects_ltx_lora_routing():
    with pytest.raises(ValidationError, match="LoRA routing"):
        GenerateRequest(
            mode="t2i",
            engine="qwen",
            prompt="cinematic city",
            loras=[{"id": "style.safetensors", "strength": 1.0}],
        )


def test_failed_ltx_execution_resets_graph_cache(monkeypatch, tmp_path):
    class Runner:
        def __init__(self):
            self.reset_calls = 0

        def stats(self):
            return {
                "enabled": True,
                "captures": 1,
                "eager_shapes": 0,
                "replays": 0,
                "replacements": 0,
                "overflow_misses": 0,
                "max_captures": 1,
            }

        def reset(self):
            self.reset_calls += 1

    class LTX:
        def __init__(self):
            self._graph_runner = Runner()

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("synthetic OOM")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    import torch

    monkeypatch.setattr(torch, "cuda", FakeCuda())
    runtime_obj = object.__new__(DirectorRuntime)
    runtime_obj.ltx = LTX()
    runtime_obj.qwen = None

    request = GenerateRequest(mode="t2av", prompt="test")
    plan = ExecutionPlan(
        requested_engine="ltx",
        engine="ltx",
        fallback_engine=None,
        fallback_applied=False,
        reason="explicit:ltx",
        resource_class="ltx_default",
        acceleration="ltx_cuda_graph_eligible",
    )

    with pytest.raises(RuntimeError, match="synthetic OOM"):
        runtime_obj.execute(plan, request, tmp_path / "out.mp4")

    assert runtime_obj.ltx._graph_runner.reset_calls == 1
