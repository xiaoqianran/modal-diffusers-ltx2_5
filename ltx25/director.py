"""Dual-resident model routing for the Modal director worker.

This module deliberately knows nothing about Modal, FastAPI, job storage, or media
backends. It owns only model residency and the decision of which resident engine
executes a generation request.
"""
from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .schemas import GenerateRequest

QWEN_IMAGE21_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE21_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
QWEN_ENGINE = "qwen"
LTX_ENGINE = "ltx"

ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Small immutable set of runtime facts available to the pure planner."""

    available_engines: frozenset[str]
    ltx_cuda_graph_enabled: bool = False
    ltx_cuda_graph_max_captures: int = 0


@dataclass(frozen=True)
class ExecutionPlan:
    """Decision compiled before generation side effects.

    ``fallback_engine`` is an availability fallback for ``engine=auto`` only; a
    generation failure never silently reroutes an explicit or already-started job.
    """

    requested_engine: str
    engine: str
    fallback_engine: str | None
    fallback_applied: bool
    reason: str
    resource_class: str
    acceleration: str

    def as_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def _qwen_resource_class(request: GenerateRequest) -> str:
    width, height = qwen_output_size(request)
    short, long = sorted((width, height))
    if long <= 1024:
        return "qwen_verified_1024"
    if short <= 1024 and long <= 1536:
        return "qwen_verified_1536x1024"
    return "qwen_unverified_large"


def _acceleration_for(request: GenerateRequest, engine: str, runtime: RuntimeSnapshot) -> str:
    if engine == QWEN_ENGINE:
        return "qwen_default"
    if request.loras or request.upscale_method == "pixel":
        return "ltx_eager_lora_safe"
    if runtime.ltx_cuda_graph_enabled and runtime.ltx_cuda_graph_max_captures > 0:
        return "ltx_cuda_graph_eligible"
    return "ltx_eager"


def build_execution_plan(request: GenerateRequest, runtime: RuntimeSnapshot) -> ExecutionPlan:
    """Pure Request -> Plan compiler. It performs no GPU, model, storage, or Modal I/O."""
    available = runtime.available_engines
    requested = request.engine
    fallback_engine: str | None = None
    fallback_applied = False

    if requested in {LTX_ENGINE, QWEN_ENGINE}:
        engine = requested
        reason = f"explicit:{engine}"
        if engine not in available:
            reason += ":unavailable"
    elif request.mode == "image_edit":
        engine = QWEN_ENGINE
        reason = "auto:qwen_image_edit"
        if engine not in available:
            reason += ":unavailable"
    elif request.mode == "t2i" and not request.conditions and not request.loras:
        fallback_engine = (
            LTX_ENGINE
            if LTX_ENGINE in available and _ltx_t2i_fallback_compatible(request)
            else None
        )
        if QWEN_ENGINE in available:
            engine = QWEN_ENGINE
            reason = "auto:pure_t2i"
        elif fallback_engine is not None:
            engine = fallback_engine
            fallback_applied = True
            reason = "auto:pure_t2i:qwen_unavailable"
        else:
            engine = QWEN_ENGINE
            reason = "auto:pure_t2i:no_available_engine"
    else:
        engine = LTX_ENGINE
        reason = "auto:ltx_capability"
        if engine not in available:
            reason += ":unavailable"

    resource_class = (
        _qwen_resource_class(request) if engine == QWEN_ENGINE else "ltx_default"
    )
    return ExecutionPlan(
        requested_engine=requested,
        engine=engine,
        fallback_engine=fallback_engine,
        fallback_applied=fallback_applied,
        reason=reason,
        resource_class=resource_class,
        acceleration=_acceleration_for(request, engine, runtime),
    )


def qwen_output_size(request: GenerateRequest) -> tuple[int, int]:
    """Preserve the Studio's existing t2i size contract.

    LTX t2i always uses its two-stage 2x path, so the UI treats width/height as
    base dimensions. Qwen does not need that internal path; it renders directly
    at the same final dimensions the user already expects.
    """
    scale = 2 if request.upscale else 1
    return request.width * scale, request.height * scale


def _ltx_t2i_fallback_compatible(request: GenerateRequest) -> bool:
    """Whether an auto-routed Qwen T2I can safely fall back to the LTX still path."""
    return (
        request.width % 32 == 0
        and request.height % 32 == 0
        and request.width * request.height <= 960 * 544
    )


class QwenImage21Generator:
    """Resident Qwen-Image 2.1 BF16 T2I + multi-reference image-edit engine."""

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
        print("[qwen] from_pretrained: start", flush=True)
        pipe = QwenImage21Pipeline.from_pretrained(
            QWEN_IMAGE21_MODEL_ID,
            revision=QWEN_IMAGE21_REVISION,
            cache_dir=str(self.model_dir),
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        loaded_cpu = time.monotonic()
        print(
            f"[qwen] from_pretrained: done in {loaded_cpu - started:.2f}s; "
            f"moving pipeline to cuda",
            flush=True,
        )
        pipe = pipe.to("cuda")
        moved_cuda = time.monotonic()
        print(
            f"[qwen] to(cuda): done in {moved_cuda - loaded_cpu:.2f}s; "
            f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB "
            f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
            flush=True,
        )
        self.pipe = pipe
        torch.cuda.synchronize()
        print("[qwen] cuda synchronize: done", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        self.load_seconds = time.monotonic() - started
        print(f"[qwen] resident load complete in {self.load_seconds:.2f}s", flush=True)

    def generate(
        self,
        request: GenerateRequest,
        target: Path,
        progress: ProgressCallback | None = None,
        *,
        input_dir: Path | None = None,
    ) -> dict[str, float | str]:
        import torch
        from PIL import Image

        if self.pipe is None:
            raise RuntimeError("Qwen-Image 2.1 is not loaded")
        if request.mode not in {"t2i", "image_edit"}:
            raise ValueError("Qwen-Image 2.1 supports t2i and image_edit jobs")

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

        prompt = request.prompt
        if request.transparent_background and "transparent" not in prompt.lower():
            prompt = (
                "Generate the result with a transparent background and a true RGBA alpha channel. "
                + prompt
            )

        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": request.negative_prompt or None,
            "width": width,
            "height": height,
            "num_inference_steps": request.steps,
            "true_cfg_scale": request.qwen_true_cfg_scale,
            "use_kv_cache": request.qwen_use_kv_cache,
            "generator": torch.Generator(device="cuda").manual_seed(request.seed),
            "callback_on_step_end": on_step_end,
        }
        if request.qwen_sigmas is not None:
            pipe_kwargs["sigmas"] = request.qwen_sigmas

        reference_images = []
        if request.mode == "image_edit":
            root = (input_dir or Path(".")).resolve()
            image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
            for condition in request.conditions:
                matches = list(root.glob(f"{condition.asset_id}.*"))
                if len(matches) != 1:
                    raise ValueError(f"Qwen input asset not found: {condition.asset_id}")
                source = matches[0]
                if source.suffix.lower() not in image_suffixes:
                    raise ValueError(f"Qwen image-edit input is not an image: {condition.asset_id}")
                with Image.open(source) as opened:
                    reference_images.append(opened.copy())
            pipe_kwargs["image"] = reference_images

        started = time.monotonic()
        image = None
        try:
            image = self.pipe(**pipe_kwargs).images[0]
            image.save(target)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3
            if progress:
                progress(1.0)
            return {
                "engine": QWEN_ENGINE,
                "generation_seconds": elapsed,
                "peak_vram_gb": peak_vram_gb,
            }
        finally:
            # Qwen shares a 96 GB worker with the resident LTX pipeline. An OOM
            # or decoder failure must return allocator cache to the shared pool;
            # otherwise one failed large-image request can poison later jobs.
            if image is not None:
                del image
            for reference in reference_images:
                reference.close()
            gc.collect()
            torch.cuda.empty_cache()


class DirectorRuntime:
    """Own resident engines and execute immutable plans produced by the pure planner."""

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

    def snapshot(self) -> RuntimeSnapshot:
        graph = self._ltx_graph_stats()
        return RuntimeSnapshot(
            available_engines=frozenset(self.engines),
            ltx_cuda_graph_enabled=bool(graph["enabled"]),
            ltx_cuda_graph_max_captures=int(graph["max_captures"]),
        )

    def plan(self, request: GenerateRequest) -> ExecutionPlan:
        return build_execution_plan(request, self.snapshot())

    def _ltx_graph_stats(self) -> dict[str, int | bool]:
        runner = getattr(self.ltx, "_graph_runner", None)
        if runner is None:
            return {
                "enabled": False,
                "captures": 0,
                "eager_shapes": 0,
                "replays": 0,
                "replacements": 0,
                "overflow_misses": 0,
                "max_captures": 0,
            }
        return runner.stats()

    def _reset_ltx_graphs(self, *, context: str) -> None:
        """Drop LTX CUDA Graph captures and return their private pool to CUDA.

        The director keeps LTX and Qwen resident on the same 96 GB worker. A
        captured LTX graph can retain several GiB in a private CUDA mempool, so
        Qwen requests must be able to reclaim that headroom before generation.
        """
        runner = getattr(self.ltx, "_graph_runner", None)
        if runner is None:
            return
        try:
            runner.reset()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[ltx25] cudagraph: {context} reset failed: {exc!r}", flush=True)

    def _reset_ltx_graphs_after_failure(self) -> None:
        """Invalidate graph state after a failed LTX request."""
        self._reset_ltx_graphs(context="failure")

    def _release_ltx_graphs_for_qwen(self) -> None:
        """Reclaim LTX graph memory before Qwen uses the shared GPU."""
        graph = self._ltx_graph_stats()
        if int(graph["captures"]) <= 0:
            return
        print(
            f"[director] releasing {graph['captures']} LTX CUDA graph capture(s) "
            "before Qwen generation",
            flush=True,
        )
        self._reset_ltx_graphs(context="qwen headroom")

    def execute(
        self,
        plan: ExecutionPlan,
        request: GenerateRequest,
        target: Path,
        progress: ProgressCallback | None = None,
        *,
        input_dir: Path | None = None,
    ) -> dict:
        if plan.engine == QWEN_ENGINE:
            if self.qwen is None:
                raise RuntimeError("Execution plan selected unavailable Qwen-Image 2.1 engine")
            self._release_ltx_graphs_for_qwen()
            metrics = self.qwen.generate(
                request, target, progress, input_dir=input_dir
            )
            metrics["plan"] = plan.as_dict()
            return metrics

        if plan.engine != LTX_ENGINE:
            raise RuntimeError(f"Execution plan selected unknown engine: {plan.engine}")

        before = self._ltx_graph_stats()
        try:
            metrics = self.ltx.generate(request, target, progress, input_dir=input_dir) or {}
        except Exception:
            self._reset_ltx_graphs_after_failure()
            raise
        after = self._ltx_graph_stats()
        metrics["engine"] = LTX_ENGINE
        metrics["plan"] = plan.as_dict()
        metrics["graph"] = {
            "enabled": after["enabled"],
            "captures": after["captures"],
            "capture_delta": int(after["captures"]) - int(before["captures"]),
            "replay_delta": int(after["replays"]) - int(before["replays"]),
            "eager_shape_delta": int(after["eager_shapes"]) - int(before["eager_shapes"]),
            "replacement_delta": int(after["replacements"]) - int(before["replacements"]),
            "overflow_miss_delta": int(after["overflow_misses"]) - int(before["overflow_misses"]),
            "max_captures": after["max_captures"],
        }
        return metrics

    def generate(
        self,
        request: GenerateRequest,
        target: Path,
        progress: ProgressCallback | None = None,
        *,
        input_dir: Path | None = None,
    ) -> dict:
        """Compatibility entry point; new worker code should plan once then execute."""
        return self.execute(
            self.plan(request),
            request,
            target,
            progress,
            input_dir=input_dir,
        )
