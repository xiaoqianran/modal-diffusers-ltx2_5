"""Modal deployment for the RTX PRO 6000 LTX-2.5 NVFP4 worker.

Model artifacts are staged by a CPU-only function into a persistent Volume.
Interactive HTTP/job routing runs on the user's machine and invokes
this deployed GPU Cls directly through the Modal SDK.
"""

from __future__ import annotations

import os
import subprocess
import sys
import modal

APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
MODEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_MODEL_VOLUME", "ltx25-models")
STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")
KERNEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_KERNEL_VOLUME", "ltx25-kernels")
JOB_DICT_NAME = os.environ.get("LTX25_MODAL_JOB_DICT", "ltx25-jobs")
HF_SECRET_NAME = os.environ.get("LTX25_MODAL_HF_SECRET", "huggingface")


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
    "OUTPUT_DIR": "/data/outputs",
    "INPUT_DIR": "/data/inputs",
    "LORA_DIR": "/data/loras",
    "HISTORY_DB": "/data/outputs/history.sqlite3",
    # Model files must be local. Kernel Hub remains online because the NATTEN
    # binary is selected for the actual torch/CUDA/SM combination at runtime.
    # Keep hardware-specific kernel binaries on a dedicated Volume. The state
    # Volume is reloaded before each job so newly-uploaded inputs/LoRAs become
    # visible; loaded NATTEN shared libraries keep files open and would make a
    # reload of that same Volume fail with "open files preventing the operation".
    "HF_HOME": "/kernel-cache/hf",
}


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
    secrets=[hf_secret],
    env=GPU_ENV,
    volumes={
        "/models": model_volume.with_mount_options(read_only=True),
        "/data": state_volume,
        "/kernel-cache": kernel_volume,
    },
)
@modal.concurrent(max_inputs=1)
class LTX25Worker:
    @modal.enter(snap=GPU_SNAPSHOT)
    def load(self):
        """Build the resident model once per GPU container."""
        import time

        from ltx25.config import settings
        from ltx25.runtime import LTXGenerator

        started = time.monotonic()
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
        from datetime import datetime, timezone
        from pathlib import Path

        from ltx25.config import settings
        from ltx25.schemas import GenerateRequest, STILL_IMAGE_MODES

        record_key = f"job:{job_id}"
        record = job_store.get(record_key) or {}
        if record.get("status") in {"completed", "interrupted"}:
            return record
        state_volume.reload()
        record.update({"status": "running", "progress": 0.0, "error": None,
                       "updated_at": datetime.now(timezone.utc).isoformat()})
        job_store.put(record_key, record)

        request = GenerateRequest.model_validate(request_payload)
        if request.mode in STILL_IMAGE_MODES:
            prefix = {"t2i": "t2i", "refine_image": "refine", "ref2i": "ref2i"}[request.mode]
            target = Path(settings.output_dir) / f"{prefix}_{job_id}.png"
        else:
            target = Path(settings.output_dir) / f"{job_id}.mp4"

        started = time.monotonic()
        last_bucket = -1

        def progress(value: float) -> None:
            nonlocal last_bucket
            bucket = int(max(0.0, min(1.0, value)) * 20)
            if bucket <= last_bucket:
                return
            last_bucket = bucket
            current = job_store.get(record_key) or record
            current["status"] = "running"
            current["progress"] = max(0.0, min(1.0, value))
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            job_store.put(record_key, current)

        try:
            metrics = self.generator.generate(request, target, progress) or {}
            state_volume.commit()
        except Exception as exc:  # ordinary generation failure: do not waste retries
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
                "image_url": f"/outputs/{target.name}" if request.mode in STILL_IMAGE_MODES else None,
                "video_url": None if request.mode in STILL_IMAGE_MODES else f"/outputs/{target.name}",
            }
        )
        job_store.put(record_key, current)
        return current



@app.local_entrypoint()
def prepare() -> None:
    """Run once (or rerun safely) before deploy: modal run modal_app.py::prepare."""
    print(prepare_models.remote())
