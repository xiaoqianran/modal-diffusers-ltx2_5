from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_quantize_ltx25.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("download_quantize_ltx25", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_nvfp4_profile_is_cpu_staging_only(monkeypatch, tmp_path):
    module = _load_script()
    calls: list[str] = []

    monkeypatch.setattr(module, "resolve_token", lambda: "test-token")
    monkeypatch.setattr(module, "free_gib", lambda _: 999.0)

    for name in (
        "download_base",
        "download_quality_components",
        "download_temporal_component",
        "download_pixel_upscaler",
        "download_modal_text_encoder",
        "download_modal_transformer_config",
        "download_modal_diffusion_decoder",
        "download_modal_nvfp4",
        "quantize_text_encoder",
        "quantize_transformer",
    ):
        monkeypatch.setattr(module, name, lambda *args, _name=name, **kwargs: calls.append(_name))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output-dir",
            str(tmp_path / "pipeline"),
            "--component",
            "modal_nvfp4",
            "--min-free-gib",
            "0",
        ],
    )

    assert module.main() == 0
    assert set(calls) == {
        "download_base",
        "download_quality_components",
        "download_temporal_component",
        "download_pixel_upscaler",
        "download_modal_text_encoder",
        "download_modal_transformer_config",
        "download_modal_diffusion_decoder",
        "download_modal_nvfp4",
    }
    pipeline_calls = [
        name for name in calls
        if name not in {"download_pixel_upscaler", "download_modal_nvfp4"}
    ]
    assert pipeline_calls == [
        "download_base",
        "download_quality_components",
        "download_temporal_component",
        "download_modal_text_encoder",
        "download_modal_transformer_config",
        "download_modal_diffusion_decoder",
    ]
    assert "quantize_text_encoder" not in calls
    assert "quantize_transformer" not in calls
