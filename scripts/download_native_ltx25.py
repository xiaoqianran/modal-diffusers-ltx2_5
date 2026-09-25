#!/usr/bin/env python3
"""Download the split LTX-2.5 assets required by exact upstream native pipelines."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "Lightricks/LTX-2.5"
REVISION = os.environ.get("LTX25_NATIVE_REVISION", "main")
ALLOW_PATTERNS = [
    "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
]
DETAILING_REPO_ID = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
DETAILING_FILENAME = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        token=os.environ.get("HF_TOKEN"),
        local_dir=args.output_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    lora_dir = args.output_dir / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)
    detailing = lora_dir / DETAILING_FILENAME
    existing_detailing = (
        args.output_dir.parent
        / "pipeline"
        / "pixel_spatial_upscaler"
        / DETAILING_FILENAME
    )
    if not detailing.is_file() and existing_detailing.is_file():
        shutil.copy2(existing_detailing, detailing)
        print(f"Reused existing detailing IC-LoRA: {existing_detailing}")
    if not detailing.is_file():
        hf_hub_download(
            repo_id=DETAILING_REPO_ID,
            filename=DETAILING_FILENAME,
            token=os.environ.get("HF_TOKEN"),
            local_dir=lora_dir,
        )
    missing = [name for name in ALLOW_PATTERNS if not (args.output_dir / name).is_file()]
    if not detailing.is_file():
        missing.append(str(detailing.relative_to(args.output_dir)))
    if missing:
        raise RuntimeError("Native LTX-2.5 download incomplete: " + ", ".join(missing))
    print(f"Native LTX-2.5 assets ready in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
