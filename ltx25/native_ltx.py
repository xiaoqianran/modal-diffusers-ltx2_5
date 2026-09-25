"""Command boundary for exact upstream LTX-2.5 native pipelines.

The production Director stays on Diffusers + transformers 5.17 because Qwen-Image
2.1 needs that stack. Upstream ltx-pipelines 1.3.0 currently pins
transformers<5.15, so semantic-only modes that need the official implementation
run in an isolated Modal worker/venv. This module is deliberately pure: it only
validates staged assets and compiles a subprocess argv.
"""

from __future__ import annotations

import os
from pathlib import Path

from .schemas import GenerateRequest, NATIVE_LTX_MODES

NATIVE_PYTHON = os.environ.get("LTX25_NATIVE_PYTHON", "/opt/ltx2/.venv/bin/python")
NATIVE_MODEL_ROOT = Path(os.environ.get("LTX25_NATIVE_MODEL_ROOT", "/models/ltx25/native"))
NATIVE_OFFLOAD = os.environ.get("LTX25_NATIVE_OFFLOAD", "cpu").strip().lower()

DEV_TRANSFORMER = NATIVE_MODEL_ROOT / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
DISTILLED_TRANSFORMER = NATIVE_MODEL_ROOT / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = NATIVE_MODEL_ROOT / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = NATIVE_MODEL_ROOT / "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = NATIVE_MODEL_ROOT / "vae/ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSAMPLER = (
    NATIVE_MODEL_ROOT / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
TEMPORAL_UPSAMPLER = (
    NATIVE_MODEL_ROOT / "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
)
DISTILLED_LORA = NATIVE_MODEL_ROOT / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
DETAILING_LORA = (
    NATIVE_MODEL_ROOT / "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
)
DURATION_HEAD = NATIVE_MODEL_ROOT / "model_patches/ltx-2.5-duration-head-bf16.safetensors"


def required_native_paths(mode: str) -> tuple[Path, ...]:
    if mode == "t2a":
        return DEV_TRANSFORMER, TEXT_ENCODER, AUDIO_VAE, DURATION_HEAD
    if mode == "keyframe_interpolation":
        return DEV_TRANSFORMER, TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE, SPATIAL_UPSAMPLER, DISTILLED_LORA
    if mode == "dfr":
        return (
            DISTILLED_TRANSFORMER,
            TEXT_ENCODER,
            VIDEO_VAE,
            AUDIO_VAE,
            SPATIAL_UPSAMPLER,
            TEMPORAL_UPSAMPLER,
            DETAILING_LORA,
            DURATION_HEAD,
        )
    raise ValueError(f"Unsupported native LTX mode: {mode}")


def validate_native_assets(mode: str) -> None:
    missing = [str(path) for path in required_native_paths(mode) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Native LTX-2.5 assets are not prepared; missing: " + ", ".join(missing)
        )


def _find_staged_asset(input_dir: Path, asset_id: str) -> Path:
    matches = [path for path in input_dir.glob(f"{asset_id}.*") if path.is_file() or path.is_symlink()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one staged asset for {asset_id}, found {len(matches)}")
    return matches[0]


def _append_user_loras(command: list[str], request: GenerateRequest, lora_dir: Path) -> None:
    for lora in request.loras:
        path = lora_dir / lora.id
        if not path.is_file():
            raise FileNotFoundError(f"LoRA not found: {lora.id}")
        command.extend(["--lora", str(path), str(lora.strength)])


def build_native_command(
    request: GenerateRequest,
    input_dir: Path,
    target: Path,
    *,
    lora_dir: Path = Path("/data/loras"),
    python_executable: str = NATIVE_PYTHON,
) -> list[str]:
    """Compile one validated GenerateRequest into the exact upstream CLI."""

    if request.mode not in NATIVE_LTX_MODES:
        raise ValueError(f"Mode {request.mode!r} is not an LTX native mode")

    module = {
        "t2a": "t2a_one_stage",
        "keyframe_interpolation": "keyframe_interpolation",
        "dfr": "dfr_pipeline",
    }[request.mode]
    transformer = DISTILLED_TRANSFORMER if request.mode == "dfr" else DEV_TRANSFORMER
    command = [
        python_executable,
        "-m",
        f"ltx_pipelines.{module}",
        "--transformer-path",
        str(transformer),
        "--text-encoder-path",
        str(TEXT_ENCODER),
        "--audio-vae-path",
        str(AUDIO_VAE),
        "--prompt",
        request.prompt,
        "--output-path",
        str(target),
        "--seed",
        str(request.seed),
        "--frame-rate",
        str(request.fps),
    ]
    # DFR is distilled end-to-end and owns fixed stage sigma schedules.  The
    # upstream distilled parser intentionally has no negative-prompt, custom
    # inference-step or multimodal-guidance knobs.
    if request.mode != "dfr":
        command.extend(
            [
                "--negative-prompt",
                request.negative_prompt,
                "--num-inference-steps",
                str(request.steps),
            ]
        )
    if request.mode in {"t2a", "dfr"}:
        command.extend(["--duration-head-path", str(DURATION_HEAD)])
    if NATIVE_OFFLOAD in {"cpu", "disk"}:
        command.extend(["--offload", NATIVE_OFFLOAD])

    if request.num_frames is None:
        command.extend(["--auto-duration", str(request.min_seconds), str(request.max_seconds)])
    else:
        command.extend(["--num-frames", str(request.num_frames)])

    if request.enhance_prompt:
        command.append("--enhance-prompt")
    if request.mode == "keyframe_interpolation" and request.hdr_color_space is not None:
        command.extend(["--hdr", request.hdr_color_space])

    if request.mode != "dfr" and request.audio_guidance_scale is not None:
        command.extend(["--audio-cfg-guidance-scale", str(request.audio_guidance_scale)])
    if request.mode != "dfr" and request.audio_stg_scale is not None:
        command.extend(["--audio-stg-guidance-scale", str(request.audio_stg_scale)])
    if request.mode != "dfr" and request.audio_rescale_scale is not None:
        command.extend(["--audio-rescale-scale", str(request.audio_rescale_scale)])
    if request.mode != "dfr" and request.audio_skip_step is not None:
        command.extend(["--audio-skip-step", str(request.audio_skip_step)])
    if request.mode != "dfr" and request.audio_stg_blocks is not None:
        command.append("--audio-stg-blocks")
        command.extend(str(value) for value in request.audio_stg_blocks)

    _append_user_loras(command, request, lora_dir)

    if request.mode == "t2a":
        return command

    command.extend(
        [
            "--video-vae-path",
            str(VIDEO_VAE),
            "--spatial-upsampler-path",
            str(SPATIAL_UPSAMPLER),
            "--height",
            str(request.height),
            "--width",
            str(request.width),
        ]
    )
    if request.mode == "keyframe_interpolation":
        command.extend(["--distilled-lora", str(DISTILLED_LORA), "1.0"])
    else:
        command.extend(
            [
                "--detailing-lora",
                str(DETAILING_LORA),
                "0.5",
                "--spatial-upscalings",
                str(request.dfr_spatial_upscalings),
                "--temporal-upscalings",
                str(request.dfr_temporal_upscalings),
            ]
        )
        if request.dfr_temporal_upscalings > 0:
            command.extend(["--temporal-upsampler-path", str(TEMPORAL_UPSAMPLER)])
    for condition in sorted(request.conditions, key=lambda item: int(item.frame_index or 0)):
        source = _find_staged_asset(input_dir, condition.asset_id)
        command.extend(
            [
                "--image",
                str(source),
                str(condition.frame_index),
                str(condition.strength),
            ]
        )
    return command
