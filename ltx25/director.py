"""Dual-resident model routing for the Modal director worker.

This module deliberately knows nothing about Modal, FastAPI, job storage, or media
backends. It owns only model residency and the decision of which resident engine
executes a generation request.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Callable

from .schemas import GenerateRequest

QWEN_IMAGE21_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE21_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
QWEN_ENGINE = "qwen"
LTX_ENGINE = "ltx"

ProgressCallback = Callable[[float], None]


def resolve_engine(request: GenerateRequest) -> str:
    """Resolve a request to one resident engine without changing legacy behavior."""
    if request.engine == "qwen":
        return QWEN_ENGINE
    return LTX_ENGINE


def qwen_output_size(request: GenerateRequest) -> tuple[int, int]:
    """Preserve the Studio's existing t2i size contract.

    LTX t2i always uses its two-stage 2x path, so the UI treats width/height as
    base dimensions. Qwen does not need that internal path; it renders directly
    at the same final dimensions the user already expects.
    """
    scale = 2 if request.upscale else 1
    return request.width * scale, request.height * scale


class QwenImage21Generator:
    """Resident Qwen-Image 2.1 BF16 text-to-image engine."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.pipe = None
        self.load_seconds = 0.0

    def load(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Qwen-Image 2.1 model directory not found: {self.model_dir}")

        started = time.monotonic()
        self.pipe = QwenImage21Pipeline.from_pretrained(
            QWEN_IMAGE21_MODEL_ID,
            revision=QWEN_IMAGE21_REVISION,
            cache_dir=str(self.model_dir),
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        self.load_seconds = time.monotonic() - started

    def generate(
        self,
        request: GenerateRequest,
        target: Path,
        progress: ProgressCallback | None = None,
    ) -> dict[str, float | str]:
        import torch

        if self.pipe is None:
            raise RuntimeError("Qwen-Image 2.1 is not loaded")
        if request.mode != "t2i":
            raise ValueError("Qwen-Image 2.1 currently supports t2i jobs only")

        width, height = qwen_output_size(request)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if progress:
            progress(0.02)

        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            if progress:
                progress(0.05 + 0.9 * ((step_index + 1) / max(1, request.steps)))
            return callback_kwargs

        started = time.monotonic()
        image = self.pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=request.steps,
            true_cfg_scale=1.0,
            use_kv_cache=True,
            generator=torch.Generator(device="cuda").manual_seed(request.seed),
            callback_on_step_end=on_step_end,
        ).images[0]
        image.save(target)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

        del image
        gc.collect()
        torch.cuda.empty_cache()
        if progress:
            progress(1.0)
        return {
            "engine": QWEN_ENGINE,
            "generation_seconds": elapsed,
            "peak_vram_gb": peak_vram_gb,
        }


class DirectorRuntime:
    """Own both resident engines and route explicit requests between them."""

    def __init__(self, config):
        from .runtime import LTXGenerator

        self.config = config
        self.ltx = LTXGenerator(config)
        self.qwen = (
            QwenImage21Generator(config.qwen_image21_dir)
            if config.director_qwen_enabled
            else None
        )
        self.load_seconds: dict[str, float] = {}

    def load(self) -> None:
        started = time.monotonic()
        self.ltx.load()
        self.load_seconds[LTX_ENGINE] = time.monotonic() - started

        if self.qwen is not None:
            self.qwen.load()
            self.load_seconds[QWEN_ENGINE] = self.qwen.load_seconds

    @property
    def engines(self) -> list[str]:
        engines = [LTX_ENGINE]
        if self.qwen is not None:
            engines.append(QWEN_ENGINE)
        return engines

    def generate(
        self,
        request: GenerateRequest,
        target: Path,
        progress: ProgressCallback | None = None,
        *,
        input_dir: Path | None = None,
    ) -> dict:
        engine = resolve_engine(request)
        if engine == QWEN_ENGINE:
            if self.qwen is None:
                raise RuntimeError("Qwen-Image 2.1 engine is disabled")
            return self.qwen.generate(request, target, progress)

        metrics = self.ltx.generate(request, target, progress, input_dir=input_dir) or {}
        metrics["engine"] = LTX_ENGINE
        return metrics
