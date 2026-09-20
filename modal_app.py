"""Modal deployment for the RTX PRO 6000 LTX-2.5 NVFP4 worker.

Model artifacts are staged by a CPU-only function into a persistent Volume.
Interactive HTTP/job routing runs on the user's machine and invokes
this deployed GPU Cls directly through the Modal SDK.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
import modal

APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
MODEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_MODEL_VOLUME", "ltx25-models")
STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")
KERNEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_KERNEL_VOLUME", "ltx25-kernels")
JOB_DICT_NAME = os.environ.get("LTX25_MODAL_JOB_DICT", "ltx25-jobs")
HF_SECRET_NAME = os.environ.get("LTX25_MODAL_HF_SECRET", "huggingface")
MEDIA_BACKEND = os.environ.get("LTX25_MEDIA_BACKEND", "volume").strip().lower()
MEDIA_PRIMARY_ID = os.environ.get("LTX25_MEDIA_PRIMARY_ID", "").strip()
MEDIA_PRIMARY_BACKEND = os.environ.get("LTX25_MEDIA_PRIMARY_BACKEND", "s3").strip().lower() if MEDIA_PRIMARY_ID else MEDIA_BACKEND
MEDIA_PRIMARY_BUCKET = os.environ.get("LTX25_MEDIA_PRIMARY_S3_BUCKET") if MEDIA_PRIMARY_ID else os.environ.get("LTX25_S3_BUCKET")
MEDIA_PRIMARY_ENDPOINT_URL = os.environ.get("LTX25_MEDIA_PRIMARY_S3_ENDPOINT_URL") if MEDIA_PRIMARY_ID else os.environ.get("LTX25_S3_ENDPOINT_URL")
MEDIA_PRIMARY_REGION = os.environ.get("LTX25_MEDIA_PRIMARY_S3_REGION", "auto") if MEDIA_PRIMARY_ID else os.environ.get("LTX25_S3_REGION", "auto")
MEDIA_FALLBACK_ID = os.environ.get("LTX25_MEDIA_FALLBACK_ID", "").strip()
MEDIA_FALLBACK_BACKEND = os.environ.get("LTX25_MEDIA_FALLBACK_BACKEND", "s3").strip().lower()
MEDIA_SECRET_NAME = os.environ.get("LTX25_MODAL_MEDIA_SECRET", "media-storage")

GPU_IDLE_SECONDS = int(os.environ.get("LTX25_MODAL_GPU_IDLE_SECONDS", "600"))

# IMPORTANT: keep container paths as POSIX strings. This module is evaluated by
# the local Modal CLI on Windows, where pathlib.Path would turn these into
# backslash paths before they are serialized into the remote function config.
MODEL_ROOT = "/models/ltx25"
PIPELINE_DIR = f"{MODEL_ROOT}/pipeline"
NVFP4_CKPT = f"{MODEL_ROOT}/checkpoints/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
state_volume = modal.Volume.from_name(STATE_VOLUME_NAME, create_if_missing=True)
kernel_volume = modal.Volume.from_name(KERNEL_VOLUME_NAME, create_if_missing=True)
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

media_mount = None
media_secret = None
MEDIA_ROOT = "/data"
if MEDIA_PRIMARY_BACKEND == "s3":
    if not MEDIA_PRIMARY_BUCKET:
        raise RuntimeError("Primary S3 bucket is required for object-storage media")
    media_secret = modal.Secret.from_name(
        MEDIA_SECRET_NAME,
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    )
    mount_kwargs = {
        "bucket_name": MEDIA_PRIMARY_BUCKET,
        "secret": media_secret,
        "read_only": True,
    }
    if MEDIA_PRIMARY_ENDPOINT_URL:
        mount_kwargs["bucket_endpoint_url"] = MEDIA_PRIMARY_ENDPOINT_URL
        mount_kwargs["force_path_style"] = True
    media_mount = modal.CloudBucketMount(**mount_kwargs)
    MEDIA_ROOT = "/media-primary"
elif MEDIA_PRIMARY_BACKEND != "volume":
    raise RuntimeError("Primary media backend must be volume or s3")

WORKER_OUTPUT_DIR = "/data/outputs" if MEDIA_PRIMARY_BACKEND == "volume" else "/tmp/ltx25-outputs"


def _cancel_key(job_id: str) -> str:
    return f"cancel:{job_id}"


def _honor_interrupt(job_id: str, fallback: dict) -> dict | None:
    """Make an accepted cancel request terminal across worker state races.

    Modal cancellation and the worker execute on different machines. A durable
    cancel tombstone lets the worker repair any `running`/`completed` write that
    raced *after* the cancel request, while preserving a completion that truly
    happened before the request arrived.
    """
    marker = job_store.get(_cancel_key(job_id))
    current = job_store.get(f"job:{job_id}") or fallback
    if current.get("status") == "interrupted":
        return current
    if not isinstance(marker, dict):
        return None

    requested_at = marker.get("requested_at", "")
    state_time = current.get("updated_at", "")
    if current.get("status") in {"completed", "failed"} and state_time and requested_at:
        if state_time < requested_at:
            return None

    current.update(
        {
            "status": "interrupted",
            "error": "Interrupted by user",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    job_store.put(f"job:{job_id}", current)
    return current

# Model preparation deliberately does not install torch/CUDA. The modal_nvfp4
# download profile is download/copy/validation only and therefore belongs on CPU.
prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub>=0.30",
        "safetensors>=0.4",
        "numpy>=1.26,<3",
    )
    .add_local_file(
        "scripts/download_quantize_ltx25.py",
        "/app/scripts/download_quantize_ltx25.py",
        copy=True,
    )
)

# Modal runtime is deliberately split into two layers:
#
#   runtime_base_image  = stable OS/CUDA-adjacent Python dependencies
#   runtime_image       = lightweight project source mount (copy=False)
#
# The base image is invalidated only when this dependency declaration or
# requirements.txt changes. Editing ltx25/*.py no longer rebuilds Torch/CUDA.
runtime_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && "
        "apt-get install -y --no-install-recommends build-essential ffmpeg && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .pip_install(
        "torch==2.13.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install_from_requirements("requirements.txt")
)

runtime_image = runtime_base_image.add_local_dir(
    "ltx25",
    "/app/ltx25",
    copy=False,
    ignore=["**/__pycache__/**", "**/*.pyc"],
)


@app.function(
    image=prep_image,
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    env={"HF_XET_HIGH_PERFORMANCE": "1"},
    volumes={"/models": model_volume},
    secrets=[hf_secret],
)
def prepare_models() -> dict[str, str]:
    """Download every artifact needed by the production NVFP4 preset on CPU."""
    subprocess.run(
        [
            sys.executable,
            "/app/scripts/download_quantize_ltx25.py",
            "--output-dir",
            PIPELINE_DIR,
            "--component",
            "modal_nvfp4",
            "--min-free-gib",
            "0",
        ],
        check=True,
    )
    model_volume.commit()
    return {
        "pipeline": PIPELINE_DIR,
        "nvfp4_checkpoint": NVFP4_CKPT,
        "status": "ready",
    }


GPU_ENV = {
    "PYTHONPATH": "/app",
    # Keep deployment-time and remote module evaluation on the same media
    # branch. Modal re-evaluates this module inside the worker container; if
    # these values are missing there, the code falls back to `volume` and the
    # serialized CloudBucketMount dependency graph no longer matches.
    "LTX25_MEDIA_BACKEND": MEDIA_BACKEND,
    "LTX25_MEDIA_PRIMARY_ID": MEDIA_PRIMARY_ID,
    "LTX25_MEDIA_PRIMARY_BACKEND": MEDIA_PRIMARY_BACKEND,
    "LTX25_MEDIA_PRIMARY_S3_BUCKET": MEDIA_PRIMARY_BUCKET or "",
    "LTX25_MEDIA_PRIMARY_S3_ENDPOINT_URL": MEDIA_PRIMARY_ENDPOINT_URL or "",
    "LTX25_MEDIA_PRIMARY_S3_REGION": MEDIA_PRIMARY_REGION,
    "LTX25_MEDIA_FALLBACK_ID": MEDIA_FALLBACK_ID,
    "LTX25_MEDIA_FALLBACK_BACKEND": MEDIA_FALLBACK_BACKEND,
    "LTX25_MEDIA_FALLBACK_S3_BUCKET": os.environ.get("LTX25_MEDIA_FALLBACK_S3_BUCKET", ""),
    "LTX25_MEDIA_FALLBACK_S3_ENDPOINT_URL": os.environ.get("LTX25_MEDIA_FALLBACK_S3_ENDPOINT_URL", ""),
    "LTX25_MEDIA_FALLBACK_S3_REGION": os.environ.get("LTX25_MEDIA_FALLBACK_S3_REGION", "auto"),
    "LTX25_MEDIA_BREAKER_FAILURES": os.environ.get("LTX25_MEDIA_BREAKER_FAILURES", "3"),
    "LTX25_MEDIA_BREAKER_COOLDOWN_SECONDS": os.environ.get("LTX25_MEDIA_BREAKER_COOLDOWN_SECONDS", "60"),
    "LTX25_INPUT_PREFETCH_MB": os.environ.get("LTX25_INPUT_PREFETCH_MB", "32"),
    "LTX25_INPUT_PREFETCH_WORKERS": os.environ.get("LTX25_INPUT_PREFETCH_WORKERS", "4"),
    "LTX25_MODAL_MEDIA_SECRET": MEDIA_SECRET_NAME,
    "QUANTIZED_MODEL_DIR": PIPELINE_DIR,
    "LTX25_TEXT_ENCODER_DIR": f"{PIPELINE_DIR}/text_encoder",
    "LTX25_TRANSFORMER_CONFIG_DIR": f"{PIPELINE_DIR}/transformer",
    "LTX25_NVFP4_CKPT": NVFP4_CKPT,
    "LTX25_REQUIRE_LOCAL_ASSETS": "1",
    "LTX25_TRANSFORMER_PRECISION": "nvfp4",
    "LTX25_PARALLEL_COLD_LOAD": "1",
    "OFFLOAD_MODE": "none",
    "LTX25_CUDA_GRAPH": "1",
    "LTX25_COMPILE_BLOCKS": "off",
    # MP4 muxing performs seeks (notably for faststart), so object-store mode
    # encodes locally first and uploads the finalized file through the S3 API.
    # CloudBucketMount is kept read-only for source-media access.
    "OUTPUT_DIR": WORKER_OUTPUT_DIR,
    "INPUT_DIR": "/tmp/ltx25-inputs",
    "LORA_DIR": "/data/loras",
    "HISTORY_DB": "/data/history.sqlite3",
    # Model files must be local. Kernel Hub remains online because the NATTEN
    # binary is selected for the actual torch/CUDA/SM combination at runtime.
    # Keep hardware-specific kernel binaries on a dedicated Volume. The state
    # Volume is reloaded before each job so newly-uploaded inputs/LoRAs become
    # visible; loaded NATTEN shared libraries keep files open and would make a
    # reload of that same Volume fail with "open files preventing the operation".
    "HF_HOME": "/kernel-cache/hf",
}

WORKER_VOLUMES = {
    "/models": model_volume.with_mount_options(read_only=True),
    "/data": state_volume,
    "/kernel-cache": kernel_volume,
}
if media_mount is not None:
    WORKER_VOLUMES["/media-primary"] = media_mount

WORKER_SECRETS = [hf_secret] + ([media_secret] if media_secret is not None else [])


@app.cls(
    image=runtime_image,
    gpu="RTX-PRO-6000",
    memory=65536,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=GPU_IDLE_SECONDS,
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=1.5,
        initial_delay=2.0,
        max_delay=10.0,
    ),
    secrets=WORKER_SECRETS,
    env=GPU_ENV,
    volumes=WORKER_VOLUMES,
)
@modal.concurrent(max_inputs=1)
class LTX25Worker:
    @modal.enter()
    def load(self):
        """Build the resident model once per GPU container."""
        import time

        from ltx25.config import settings
        from ltx25.media_storage import create_media_storage
        from ltx25.runtime import LTXGenerator

        started = time.monotonic()
        self.media_storage = create_media_storage(state_volume)
        self.generator = LTXGenerator(settings)
        self.generator.load()
        self.load_seconds = time.monotonic() - started

    @modal.method()
    def ready(self) -> dict:
        """Cheap warm-up probe used by the local router to start the GPU asynchronously."""
        import torch

        return {
            "status": "ready",
            "load_seconds": self.load_seconds,
            "gpu": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        }

    @modal.method()
    def generate(self, job_id: str, request_payload: dict) -> dict:
        """Run one durable generation input.

        Normal model/validation failures are converted into a failed job result,
        while process-level failures such as GPU preemption escape this method and
        are retried by Modal with the same job id/output path.
        """
        import os as _os
        import shutil
        import time
        from pathlib import Path

        from ltx25.config import settings
        from ltx25.media_storage import MediaRef
        from ltx25.schemas import GenerateRequest, STILL_IMAGE_MODES

        record_key = f"job:{job_id}"
        record = job_store.get(record_key) or {}
        worker_started = time.monotonic()
        try:
            created_at = datetime.fromisoformat(str(record.get("created_at")))
            queue_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - created_at).total_seconds(),
            )
        except Exception:
            queue_seconds = 0.0
        if record.get("status") in {"completed", "interrupted"}:
            return record
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        reload_started = time.monotonic()
        state_volume.reload()
        state_reload_seconds = time.monotonic() - reload_started
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        record.update({
            "status": "running",
            "progress": 0.0,
            "error": None,
            "timings": {
                "queue_seconds": queue_seconds,
                "state_reload_seconds": state_reload_seconds,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        job_store.put(record_key, record)
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted

        request = GenerateRequest.model_validate(request_payload)
        workspace = Path(f"/tmp/ltx25-jobs/{job_id}")
        input_dir = workspace / "inputs"
        shutil.rmtree(workspace, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        staging_started = time.monotonic()
        mounted_inputs = 0
        prefetched_inputs = 0
        try:
            from concurrent.futures import ThreadPoolExecutor

            asset_ids = {item.asset_id for item in request.conditions}
            if request.audio_asset_id:
                asset_ids.add(request.audio_asset_id)
            downloads: list[tuple[MediaRef, Path]] = []
            prefetch_limit = max(
                1,
                int(_os.environ.get("LTX25_INPUT_PREFETCH_MB", "32")),
            ) * 1024 * 1024
            for asset_id in asset_ids:
                asset_record = job_store.get(f"asset:{asset_id}")
                if not isinstance(asset_record, dict):
                    raise FileNotFoundError(f"Input asset metadata not found: {asset_id}")
                ref = MediaRef.from_value(asset_record, default_store_id=self.media_storage.primary_id)
                target_input = input_dir / Path(ref.key).name
                size = int(asset_record.get("size") or 0)
                if (
                    media_mount is not None
                    and ref.store_id == self.media_storage.primary_id
                    and size > prefetch_limit
                ):
                    mounted = Path("/media-primary") / ref.key
                    _os.symlink(mounted, target_input)
                    mounted_inputs += 1
                else:
                    downloads.append((ref, target_input))
            if downloads:
                workers = min(
                    len(downloads),
                    max(1, int(_os.environ.get("LTX25_INPUT_PREFETCH_WORKERS", "4"))),
                )
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ltx25-input") as pool:
                    futures = [
                        pool.submit(self.media_storage.download_to, ref, target)
                        for ref, target in downloads
                    ]
                    for future in futures:
                        future.result()
                prefetched_inputs = len(downloads)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise
        input_staging_seconds = time.monotonic() - staging_started
        if request.mode in STILL_IMAGE_MODES:
            prefix = {"t2i": "t2i", "refine_image": "refine", "ref2i": "ref2i"}[request.mode]
            target = Path(settings.output_dir) / f"{prefix}_{job_id}.png"
        else:
            target = Path(settings.output_dir) / f"{job_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            generation_started = time.monotonic()
            generation_seconds = None
            output_store_seconds = None
            output_store_started = None
            last_bucket = -1

            def progress(value: float) -> None:
                nonlocal last_bucket
                bucket = int(max(0.0, min(1.0, value)) * 20)
                if bucket <= last_bucket:
                    return
                last_bucket = bucket
                current = job_store.get(record_key) or record
                if _honor_interrupt(job_id, current) is not None:
                    return
                current["status"] = "running"
                current["progress"] = max(0.0, min(1.0, value))
                current["updated_at"] = datetime.now(timezone.utc).isoformat()
                job_store.put(record_key, current)
                _honor_interrupt(job_id, current)

            try:
                metrics = self.generator.generate(request, target, progress, input_dir=input_dir) or {}
                generation_seconds = time.monotonic() - generation_started
                interrupted = _honor_interrupt(job_id, record)
                if interrupted is not None:
                    return interrupted
                output_size = target.stat().st_size
                output_store_started = time.monotonic()
                if MEDIA_PRIMARY_BACKEND == "volume" and not MEDIA_FALLBACK_ID:
                    state_volume.commit()
                    output_ref = MediaRef(self.media_storage.primary_id, f"outputs/{target.name}")
                else:
                    output_ref = self.media_storage.upload_local(target, f"outputs/{target.name}")
                    target.unlink(missing_ok=True)
                output_store_seconds = time.monotonic() - output_store_started
                job_store.put(
                    f"output:{target.name}",
                    {**output_ref.as_dict(), "size": output_size},
                )
                interrupted = _honor_interrupt(job_id, record)
                if interrupted is not None:
                    return interrupted
            except Exception as exc:  # ordinary generation failure: do not waste retries
                if output_store_seconds is None and output_store_started is not None:
                    output_store_seconds = time.monotonic() - output_store_started
                interrupted = _honor_interrupt(job_id, record)
                if interrupted is not None:
                    return interrupted
                current = job_store.get(record_key) or record
                current.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "generation_seconds": (
                            generation_seconds
                            if generation_seconds is not None
                            else time.monotonic() - generation_started
                        ),
                        "timings": {
                            "queue_seconds": queue_seconds,
                            "state_reload_seconds": state_reload_seconds,
                            "input_staging_seconds": input_staging_seconds,
                            "output_store_seconds": output_store_seconds or 0.0,
                            "worker_seconds": time.monotonic() - worker_started,
                        },
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                job_store.put(record_key, current)
                interrupted = _honor_interrupt(job_id, current)
                if interrupted is not None:
                    return interrupted
                return current

            current = job_store.get(record_key) or record
            current.update(
                {
                    "status": "completed",
                    "progress": 1.0,
                    "error": None,
                    "generation_seconds": generation_seconds,
                    "peak_vram_gb": metrics.get("peak_vram_gb"),
                    "timings": {
                        "queue_seconds": queue_seconds,
                        "state_reload_seconds": state_reload_seconds,
                        "input_staging_seconds": input_staging_seconds,
                        "output_store_seconds": output_store_seconds or 0.0,
                        "worker_seconds": time.monotonic() - worker_started,
                    },
                    "input_staging": {
                        "prefetched": prefetched_inputs,
                        "mounted": mounted_inputs,
                    },
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "image_key": f"outputs/{target.name}" if request.mode in STILL_IMAGE_MODES else None,
                    "video_key": None if request.mode in STILL_IMAGE_MODES else f"outputs/{target.name}",
                    "image_url": None,
                    "video_url": None,
                }
            )
            job_store.put(record_key, current)
            interrupted = _honor_interrupt(job_id, current)
            if interrupted is not None:
                return interrupted
            return current
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


@app.local_entrypoint()
def prepare() -> None:
    """Run once (or rerun safely) before deploy: modal run modal_app.py::prepare."""
    print(prepare_models.remote())
