import pytest
from pydantic import ValidationError

from ltx25.director import LTX_ENGINE, QWEN_ENGINE, qwen_output_size, resolve_engine
from ltx25.schemas import GenerateRequest


def test_auto_engine_routes_pure_t2i_to_qwen():
    request = GenerateRequest(mode="t2i", prompt="cinematic city", width=512, height=512)

    assert request.engine == "auto"
    assert resolve_engine(request) == QWEN_ENGINE


def test_auto_engine_keeps_video_and_lora_work_on_ltx():
    video = GenerateRequest(mode="t2av", prompt="cinematic city")
    styled = GenerateRequest(
        mode="t2i",
        prompt="cinematic city",
        loras=[{"id": "style.safetensors", "strength": 1.0}],
    )

    assert resolve_engine(video) == LTX_ENGINE
    assert resolve_engine(styled) == LTX_ENGINE


def test_qwen_t2i_routes_to_qwen_and_preserves_final_size_contract():
    request = GenerateRequest(
        mode="t2i",
        engine="qwen",
        prompt="cinematic city",
        width=512,
        height=512,
        steps=40,
    )

    assert resolve_engine(request) == QWEN_ENGINE
    assert request.upscale is True
    assert qwen_output_size(request) == (1024, 1024)


def test_qwen_rejects_non_t2i_modes():
    with pytest.raises(ValidationError, match="t2i mode only"):
        GenerateRequest(mode="t2av", engine="qwen", prompt="cinematic city")


def test_qwen_rejects_ltx_lora_routing():
    with pytest.raises(ValidationError, match="LoRA routing"):
        GenerateRequest(
            mode="t2i",
            engine="qwen",
            prompt="cinematic city",
            loras=[{"id": "style.safetensors", "strength": 1.0}],
        )
