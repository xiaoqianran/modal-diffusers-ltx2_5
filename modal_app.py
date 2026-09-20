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
MEDIA_BUCKET = os.environ.get("LTX25_R2_BUCKET") or os.environ.get("LTX25_S3_BUCKET")
MEDIA_ENDPOINT_URL = os.environ.get("LTX25_R2_ENDPOINT_URL") or os.environ.get("LTX25_S3_ENDPOINT_URL")
MEDIA_REGION = os.environ.get("LTX25_S3_REGION", "auto")
MEDIA_SECRET_NAME = os.environ.get("LTX25_MODAL_MEDIA_SECRET", "s3-media")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


GPU_SNAPSHOT = _env_flag("LTX25_MODAL_GPU_SNAPSHOT", False)
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
if MEDIA_BACKEND in {"r2", "s3"}:
    if not MEDIA_BUCKET:
        raise RuntimeError("LTX25_R2_BUCKET/LTX25_S3_BUCKET is required for object-storage media")
    media_secret = modal.Secret.from_name(
        MEDIA_SECRET_NAME,
        required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    )
    mount_kwargs = {
        "bucket_name": MEDIA_BUCKET,
        "secret": media_secret,
        # The worker only reads source media through Mountpoint. Generated
        # outputs are uploaded with the S3 API after local mux/finalization.
        "read_only": True,
    }
    if MEDIA_ENDPOINT_URL:
        mount_kwargs["bucket_endpoint_url"] = MEDIA_ENDPOINT_URL
        # Self-hosted S3 endpoints such as MinIO may not provide wildcard
        # bucket DNS. Keep the bucket in the request path instead of resolving
        # <bucket>.<endpoint>.
        mount_kwargs["force_path_style"] = True
    media_mount = modal.CloudBucketMount(**mount_kwargs)
    MEDIA_ROOT = "/media"
elif MEDIA_BACKEND != "volume":
    raise RuntimeError("LTX25_MEDIA_BACKEND must be volume, r2, or s3")

WORKER_OUTPUT_DIR = "/data/outputs" if MEDIA_BACKEND == "volume" else "/tmp/ltx25-outputs"


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
    "LTX25_S3_BUCKET": MEDIA_BUCKET or "",
    "LTX25_S3_ENDPOINT_URL": MEDIA_ENDPOINT_URL or "",
    "LTX25_S3_REGION": MEDIA_REGION,
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
    "INPUT_DIR": f"{MEDIA_ROOT}/inputs",
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
    WORKER_VOLUMES["/media"] = media_mount

WORKER_SECRETS = [hf_secret] + ([media_secret] if media_secret is not None else [])


@app.cls(
    image=runtime_image,
    gpu="RTX-PRO-6000",
    memory=65536,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=GPU_IDLE_SECONDS,
    enable_memory_snapshot=GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if GPU_SNAPSHOT else {},
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
    @modal.enter(snap=GPU_SNAPSHOT)
    def load(self):
        """Build the resident model once per GPU container."""
        import time

        from ltx25.config import settings
        from ltx25.media_store import create_media_store
        from ltx25.runtime import LTXGenerator

        started = time.monotonic()
        self.media_store = create_media_store(None) if MEDIA_BACKEND != "volume" else None
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
            "snapshot_enabled": GPU_SNAPSHOT,
        }

    @modal.method()
    def generate(self, job_id: str, request_payload: dict) -> dict:
        """Run one durable generation input.

        Normal model/validation failures are converted into a failed job result,
        while process-level failures such as GPU preemption escape this method and
        are retried by Modal with the same job id/output path.
        """
        import time
        from pathlib import Path

        from ltx25.config import settings
        from ltx25.schemas import GenerateRequest, STILL_IMAGE_MODES

        record_key = f"job:{job_id}"
        record = job_store.get(record_key) or {}
        if record.get("status") in {"completed", "interrupted"}:
            return record
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        state_volume.reload()
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        record.update({"status": "running", "progress": 0.0, "error": None,
                       "updated_at": datetime.now(timezone.utc).isoformat()})
        job_store.put(record_key, record)
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted

        request = GenerateRequest.model_validate(request_payload)
        if request.mode in STILL_IMAGE_MODES:
            prefix = {"t2i": "t2i", "refine_image": "refine", "ref2i": "ref2i"}[request.mode]
            target = Path(settings.output_dir) / f"{prefix}_{job_id}.png"
        else:
            target = Path(settings.output_dir) / f"{job_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
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
            metrics = self.generator.generate(request, target, progress) or {}
            interrupted = _honor_interrupt(job_id, record)
            if interrupted is not None:
                return interrupted
            if MEDIA_BACKEND == "volume":
                state_volume.commit()
            else:
                self.media_store.upload_local(target, f"outputs/{target.name}")
                target.unlink(missing_ok=True)
            interrupted = _honor_interrupt(job_id, record)
            if interrupted is not None:
                return interrupted
        except Exception as exc:  # ordinary generation failure: do not waste retries
            interrupted = _honor_interrupt(job_id, record)
            if interrupted is not None:
                return interrupted
            current = job_store.get(record_key) or record
            current.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "generation_seconds": time.monotonic() - started,
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
                "generation_seconds": time.monotonic() - started,
                "peak_vram_gb": metrics.get("peak_vram_gb"),
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



@app.local_entrypoint()
def prepare() -> None:
    """Run once (or rerun safely) before deploy: modal run modal_app.py::prepare."""
    print(prepare_models.remote())
