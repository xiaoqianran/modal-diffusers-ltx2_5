"""Modal deployment for the RTX PRO 6000 LTX-2.5 NVFP4 serving path.

Model artifacts are staged by a CPU-only function into a persistent Volume.
The GPU web function mounts that Volume read-only and only assembles/loads
already-local weights before serving the existing FastAPI application.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
MODEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_MODEL_VOLUME", "ltx25-models")
STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")
HF_SECRET_NAME = os.environ.get("LTX25_MODAL_HF_SECRET", "huggingface")

MODEL_ROOT = Path("/models/ltx25")
PIPELINE_DIR = MODEL_ROOT / "pipeline"
NVFP4_CKPT = MODEL_ROOT / "checkpoints" / "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
state_volume = modal.Volume.from_name(STATE_VOLUME_NAME, create_if_missing=True)
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

# Reuse the repository's tested CUDA/runtime dependency definition. Modal image
# construction itself is CPU-only; a GPU is attached only to the web function.
runtime_image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")


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
            str(PIPELINE_DIR),
            "--component",
            "modal_nvfp4",
            "--min-free-gib",
            "0",
        ],
        check=True,
    )
    model_volume.commit()
    return {
        "pipeline": str(PIPELINE_DIR),
        "nvfp4_checkpoint": str(NVFP4_CKPT),
        "status": "ready",
    }


GPU_ENV = {
    "PYTHONPATH": "/app",
    "QUANTIZED_MODEL_DIR": str(PIPELINE_DIR),
    "LTX25_TEXT_ENCODER_DIR": str(PIPELINE_DIR / "text_encoder"),
    "LTX25_TRANSFORMER_CONFIG_DIR": str(PIPELINE_DIR / "transformer"),
    "LTX25_NVFP4_CKPT": str(NVFP4_CKPT),
    "LTX25_REQUIRE_LOCAL_ASSETS": "1",
    "LTX25_TRANSFORMER_PRECISION": "nvfp4",
    "OFFLOAD_MODE": "none",
    "LTX25_CUDA_GRAPH": "1",
    "LTX25_COMPILE_BLOCKS": "off",
    "OUTPUT_DIR": "/data/outputs",
    "INPUT_DIR": "/data/inputs",
    "LORA_DIR": "/data/loras",
    "HISTORY_DB": "/data/outputs/history.sqlite3",
    # Model files must be local. Kernel Hub remains online because the NATTEN
    # binary is selected for the actual torch/CUDA/SM combination at runtime.
    # Keep that small hardware-specific cache on the state Volume so it is paid
    # only once rather than once per GPU container cold start.
    "HF_HOME": "/data/hf-cache",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


@app.cls(
    image=runtime_image,
    gpu="RTX-PRO-6000",
    memory=65536,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=300,
    env=GPU_ENV,
    volumes={
        "/models": model_volume.with_mount_options(read_only=True),
        "/data": state_volume,
    },
)
@modal.concurrent(max_inputs=16)
class LTX25Server:
    @modal.enter()
    def load(self):
        """Assemble all pre-staged components once per RTX PRO 6000 container."""
        from app.main import manager

        manager.generator.load()

    @modal.asgi_app()
    def web(self):
        from app.main import app as fastapi_app

        return fastapi_app


@app.local_entrypoint()
def prepare() -> None:
    """Run once (or rerun safely) before deploy: modal run modal_app.py::prepare."""
    print(prepare_models.remote())
