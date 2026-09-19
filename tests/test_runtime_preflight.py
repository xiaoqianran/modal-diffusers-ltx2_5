from __future__ import annotations

import sys
from types import SimpleNamespace
from types import ModuleType

import torch

from ltx25.models import ModelLifecycle, _create_natten_processor
from ltx25.runtime import LTXGenerator, PIXEL_DETAILING_LORA_STRENGTH, _PatchRegistry


def test_pixel_detailing_lora_matches_official_dfr_strength():
    assert PIXEL_DETAILING_LORA_STRENGTH == 0.5


def test_patch_registry_restores_temporary_pipeline_state_after_partial_cleanup():
    pipe = SimpleNamespace(prepare_latents="base", audio_scheduler="scheduler")
    patches = _PatchRegistry()

    _, restore_prepare = patches.replace(pipe, "prepare_latents", "job-prepare")
    patches.replace(pipe, "audio_scheduler", "job-scheduler")
    assert pipe.prepare_latents == "job-prepare"
    assert pipe.audio_scheduler == "job-scheduler"

    # Stage 1 may restore one patch early; the outer generate() finally must
    # still safely restore every remaining transient without double-restoring.
    restore_prepare()
    patches.restore_all()

    assert pipe.prepare_latents == "base"
    assert pipe.audio_scheduler == "scheduler"


def test_a2v_mel_transform_is_pure_torch_and_finite():
    generator = LTXGenerator.__new__(LTXGenerator)
    waveform = torch.zeros((1, 2, 16_000), dtype=torch.float32)

    transform = generator._get_mel_transform_16k(torch.device("cpu"))
    mel = transform(waveform)

    assert mel.shape[:3] == (1, 2, 64)
    assert mel.shape[-1] > 0
    assert torch.isfinite(mel).all()


def test_high_resolution_diffusion_decoder_uses_tiling(monkeypatch):
    calls = []

    class Decoder:
        def enable_tiling(self, **kwargs):
            calls.append(kwargs)

    lifecycle = ModelLifecycle.__new__(ModelLifecycle)
    lifecycle.config = SimpleNamespace(ltx25_decode_single_tile="auto")
    lifecycle._diffusion_decode_pipe = SimpleNamespace(diffusion_decoder=Decoder())

    # Reproduce the old risky I2V decode shape with about 45 GiB free VRAM.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (45 * 1024**3, 96 * 1024**3))
    lifecycle._configure_decode_tiling(num_frames=121, height=832, width=1216)

    assert len(calls) == 1
    assert calls[0]["tile_sample_min_width"] < 1216
    assert calls[0]["tile_sample_stride_width"] < 1216
    assert calls[0]["tile_sample_min_num_frames"] == 121


def test_natten_kernel_trust_is_scoped_to_exact_repo(monkeypatch):
    calls = []

    class Processor:
        pass

    fake_diffusers = ModuleType("diffusers")
    fake_models = ModuleType("diffusers.models")
    fake_autoencoders = ModuleType("diffusers.models.autoencoders")
    fake_decoder = ModuleType("diffusers.models.autoencoders.ltx2_diffusion_decoder")
    fake_decoder.LTX2VideoVaeNeighborhoodNattenProcessor = Processor
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", fake_models)
    monkeypatch.setitem(sys.modules, "diffusers.models.autoencoders", fake_autoencoders)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.models.autoencoders.ltx2_diffusion_decoder",
        fake_decoder,
    )

    def get_kernel(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return SimpleNamespace(na3d=object())

    fake_kernels = ModuleType("kernels")
    fake_kernels.get_kernel = get_kernel
    monkeypatch.setitem(sys.modules, "kernels", fake_kernels)

    processor = _create_natten_processor()

    assert processor._na3d is not None
    assert calls == [
        (
            "shi-labs/natten",
            {"version": 1, "trust_remote_code": ["shi-labs/natten"]},
        )
    ]
