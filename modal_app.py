"""Modal deployment for the RTX PRO 6000 dual-resident Director worker.

LTX-2.5 NVFP4 artifacts live in the model Volume while Qwen-Image 2.1 reuses
a dedicated read-only Hugging Face cache Volume. Interactive HTTP/job routing
runs on the user's machine and invokes this deployed GPU Cls through Modal RPC.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
import modal

APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
MODAL_ENVIRONMENT = os.environ.get("LTX25_MODAL_ENVIRONMENT", "main")
MODEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_MODEL_VOLUME", "ltx25-models")
STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")
KERNEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_KERNEL_VOLUME", "ltx25-kernels")
QWEN_CACHE_VOLUME_NAME = os.environ.get("QWEN_IMAGE21_CACHE_VOLUME", "qwen-image21-cache")
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
GPU_MIN_CONTAINERS = int(os.environ.get("LTX25_MODAL_GPU_MIN_CONTAINERS", "1"))
DIRECTOR_QWEN_ENABLED = os.environ.get("DIRECTOR_QWEN_ENABLED", "1").strip().lower() not in {
    "", "0", "false", "no", "off"
}

# IMPORTANT: keep container paths as POSIX strings. This module is evaluated by
# the local Modal CLI on Windows, where pathlib.Path would turn these into
# backslash paths before they are serialized into the remote function config.
MODEL_ROOT = "/models/ltx25"
PIPELINE_DIR = f"{MODEL_ROOT}/pipeline"
NVFP4_CKPT = f"{MODEL_ROOT}/checkpoints/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
QWEN_IMAGE21_DIR = "/qwen-cache/huggingface/hub"
QWEN_IMAGE21_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN_IMAGE21_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
)
state_volume = modal.Volume.from_name(
    STATE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
)
kernel_volume = modal.Volume.from_name(
    KERNEL_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
)
qwen_cache_volume = modal.Volume.from_name(
    QWEN_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
)
job_store = modal.Dict.from_name(
    JOB_DICT_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)

media_mount = None
media_secret = None
MEDIA_ROOT = "/data"
if MEDIA_PRIMARY_BACKEND == "s3":
    if not MEDIA_PRIMARY_BUCKET:
        raise RuntimeError("Primary S3 bucket is required for object-storage media")
    media_secret = modal.Secret.from_name(
        MEDIA_SECRET_NAME,
        environment_name=MODAL_ENVIRONMENT,
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
    .add_local_file(
        "scripts/download_native_ltx25.py",
        "/app/scripts/download_native_ltx25.py",
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
        "apt-get install -y --no-install-recommends build-essential ffmpeg git && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .pip_install(
        "torch==2.13.0+cu130",
        "torchvision==0.28.0+cu130",
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

# Exact upstream ltx-pipelines must live in an isolated environment.  ltx-core
# 1.3.0 currently requires transformers<5.15 while the resident Qwen-Image 2.1
# path uses transformers 5.17.
LTX_NATIVE_COMMIT = "a95ab856bf29407b6b066ede0abe1846050db56c"
native_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && "
        "apt-get install -y --no-install-recommends build-essential ffmpeg git && "
        "rm -rf /var/lib/apt/lists/*",
        "python -m pip install --no-cache-dir uv pydantic pydantic-settings boto3",
        "git clone --filter=blob:none https://github.com/Lightricks/LTX-2.git /opt/ltx2",
        f"cd /opt/ltx2 && git checkout {LTX_NATIVE_COMMIT}",
        "cd /opt/ltx2 && uv sync --package ltx-pipelines --no-dev",
        # DiffVAE's production neighborhood-attention backend is an optional
        # ltx-core extra. ltx-pipelines depends on bare ltx-core, so syncing the
        # pipeline package alone intentionally does not install NATTEN. Use the
        # exact upstream command from the pinned LTX commit; its pyproject pins
        # torch 2.13 / cu132 and natten 0.21.7 from https://whl.natten.org.
        "cd /opt/ltx2 && uv sync --package ltx-core --extra natten --no-dev",
        "cd /opt/ltx2 && .venv/bin/python -c "
        "\"import torch, natten; "
        "assert torch.__version__.startswith('2.13.0'), torch.__version__; "
        "assert torch.version.cuda == '13.2', torch.version.cuda; "
        "assert natten.__version__.startswith('0.21.7'), natten.__version__; "
        "print('native runtime:', torch.__version__, 'cuda', torch.version.cuda, "
        "'natten', natten.__version__)\"",
    )
)
native_runtime_image = native_base_image.add_local_dir(
    "ltx25",
    "/app/ltx25",
    copy=False,
    ignore=["**/__pycache__/**", "**/*.pyc"],
)


@app.function(
    image=prep_image,
    cpu=4,
    memory=32768,
    timeout=6 * 60 * 60,
    env={"HF_XET_HIGH_PERFORMANCE": "1"},
    volumes={"/models": model_volume},
    secrets=[hf_secret],
)
def prepare_models() -> dict[str, str]:
    """Prepare resident and native LTX assets concurrently in one Volume writer."""
    native_root = f"{MODEL_ROOT}/native"
    resident_cmd = [
        sys.executable,
        "/app/scripts/download_quantize_ltx25.py",
        "--output-dir",
        PIPELINE_DIR,
        "--component",
        "modal_nvfp4",
        "--min-free-gib",
        "0",
    ]
    native_cmd = [
        sys.executable,
        "/app/scripts/download_native_ltx25.py",
        "--output-dir",
        native_root,
        "--skip-detailing",
    ]
    print("[prepare] starting resident + native LTX downloads in parallel", flush=True)
    resident_proc = subprocess.Popen(resident_cmd)
    native_proc = subprocess.Popen(native_cmd)
    resident_rc = resident_proc.wait()
    native_rc = native_proc.wait()
    if resident_rc != 0:
        raise subprocess.CalledProcessError(resident_rc, resident_cmd)
    if native_rc != 0:
        raise subprocess.CalledProcessError(native_rc, native_cmd)

    # Both downloads are complete now, so native detailing can reuse the
    # resident Pixel IC-LoRA instead of fetching the same artifact again.
    subprocess.run(
        [
            sys.executable,
            "/app/scripts/download_native_ltx25.py",
            "--output-dir",
            native_root,
            "--detail-only",
        ],
        check=True,
    )
    model_volume.commit()
    return {
        "pipeline": PIPELINE_DIR,
        "nvfp4_checkpoint": NVFP4_CKPT,
        "native": native_root,
        "qwen_image21_cache": QWEN_CACHE_VOLUME_NAME if DIRECTOR_QWEN_ENABLED else "disabled",
        "status": "ready",
    }


@app.function(
    image=prep_image,
    cpu=4,
    memory=32768,
    timeout=6 * 60 * 60,
    env={"HF_XET_HIGH_PERFORMANCE": "1"},
    volumes={"/qwen-cache": qwen_cache_volume},
    secrets=[hf_secret],
)
def prepare_qwen_image21() -> dict[str, str]:
    """Populate the read-only Qwen Hub cache used by the resident Director."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=QWEN_IMAGE21_MODEL_ID,
        revision=QWEN_IMAGE21_REVISION,
        token=os.environ.get("HF_TOKEN"),
        cache_dir=QWEN_IMAGE21_DIR,
    )
    qwen_cache_volume.commit()
    return {
        "model": QWEN_IMAGE21_MODEL_ID,
        "revision": QWEN_IMAGE21_REVISION,
        "cache": QWEN_CACHE_VOLUME_NAME,
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
    "DIRECTOR_QWEN_ENABLED": "1" if DIRECTOR_QWEN_ENABLED else "0",
    "QWEN_IMAGE21_DIR": QWEN_IMAGE21_DIR,
    "OFFLOAD_MODE": "none",
    # Dual-resident production smoke shows one retained LTX graph costs <1 GiB
    # while materially reducing repeat latency. Bound the cache to one shape so
    # Qwen keeps predictable activation headroom on the 96 GB worker.
    "LTX25_CUDA_GRAPH": os.environ.get("LTX25_CUDA_GRAPH", "1"),
    "LTX25_CUDA_GRAPH_MAX_CAPTURES": os.environ.get(
        "LTX25_CUDA_GRAPH_MAX_CAPTURES", "1" if DIRECTOR_QWEN_ENABLED else "8"
    ),
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
if DIRECTOR_QWEN_ENABLED:
    WORKER_VOLUMES["/qwen-cache"] = qwen_cache_volume.with_mount_options(read_only=True)

if media_mount is not None:
    WORKER_VOLUMES["/media-primary"] = media_mount

WORKER_SECRETS = [hf_secret] + ([media_secret] if media_secret is not None else [])


@app.cls(
    image=runtime_image,
    gpu="RTX-PRO-6000",
    memory=65536,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    # Production deploy keeps one RTX PRO 6000 resident. Model preparation sets
    # LTX25_MODAL_GPU_MIN_CONTAINERS=0 so the ephemeral `modal run ::prepare`
    # cannot race the still-populating model Volumes and allocate a GPU early.
    min_containers=GPU_MIN_CONTAINERS,
    max_containers=1,
    # scaledown_window=GPU_IDLE_SECONDS,  # disabled: this worker must not scale to zero
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
class DirectorWorker:
    @modal.enter()
    def load(self):
        """Build both resident model engines once per GPU container."""
        import time

        from ltx25.config import settings
        from ltx25.director import DirectorRuntime
        from ltx25.media_storage import create_media_storage

        started = time.monotonic()
        self.media_storage = create_media_storage(state_volume)
        self.director = DirectorRuntime(settings)
        self.director.load()
        self.load_seconds = time.monotonic() - started

    @modal.method()
    def ready(self) -> dict:
        """Cheap warm-up probe used by the local router to start the GPU asynchronously."""
        import torch

        return {
            "status": "ready",
            "load_seconds": self.load_seconds,
            "model_load_seconds": self.director.load_seconds,
            "engines": self.director.engines,
            "gpu": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
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
        plan = self.director.plan(request)
        record = job_store.get(record_key) or record
        record["plan"] = plan.as_dict()
        record["engine"] = plan.engine
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        job_store.put(record_key, record)

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
                target_input = input_dir / f"{asset_id}{Path(ref.key).suffix}"
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
            prefix = {"t2i": "t2i", "image_edit": "qwen-edit", "refine_image": "refine", "ref2i": "ref2i"}[request.mode]
            if plan.engine == "qwen":
                prefix = "qwen"
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
                metrics = self.director.execute(
                    plan, request, target, progress, input_dir=input_dir
                ) or {}
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
                    "engine": metrics.get("engine"),
                    "plan": metrics.get("plan", plan.as_dict()),
                    "graph": metrics.get("graph"),
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


NATIVE_WORKER_VOLUMES = {
    "/models": model_volume.with_mount_options(read_only=True),
    "/data": state_volume,
}
if media_mount is not None:
    NATIVE_WORKER_VOLUMES["/media-primary"] = media_mount


@app.function(
    image=native_runtime_image,
    gpu="RTX-PRO-6000",
    memory=131072,
    timeout=45 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=60,
    retries=modal.Retries(
        max_retries=1,
        backoff_coefficient=1.5,
        initial_delay=2.0,
        max_delay=10.0,
    ),
    secrets=WORKER_SECRETS,
    env={
        **GPU_ENV,
        "DIRECTOR_QWEN_ENABLED": "0",
        "LTX25_NATIVE_MODEL_ROOT": f"{MODEL_ROOT}/native",
        "LTX25_NATIVE_PYTHON": "/opt/ltx2/.venv/bin/python",
        "LTX25_NATIVE_OFFLOAD": os.environ.get("LTX25_NATIVE_OFFLOAD", "cpu"),
    },
    volumes=NATIVE_WORKER_VOLUMES,
)
def native_generate(job_id: str, request_payload: dict) -> dict:
    """Execute exact upstream T2A, keyframe interpolation or DFR in an isolated stack."""
    import shutil
    import time
    from pathlib import Path

    from ltx25.media_storage import MediaRef, create_media_storage
    from ltx25.native_ltx import build_native_command, validate_native_assets
    from ltx25.schemas import GenerateRequest, NATIVE_LTX_MODES

    record_key = f"job:{job_id}"
    record = job_store.get(record_key) or {}
    started = time.monotonic()

    def save(**updates):
        current = job_store.get(record_key) or record
        current.update(updates)
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        job_store.put(record_key, current)
        return current

    try:
        request = GenerateRequest.model_validate(request_payload)
        if request.mode not in NATIVE_LTX_MODES:
            raise ValueError(f"native_generate does not support mode {request.mode!r}")

        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted

        state_volume.reload()
        media_storage = create_media_storage(state_volume)
        plan = {
            "requested_engine": request.engine,
            "engine": "ltx-native",
            "fallback_engine": None,
            "reason": f"native:{request.mode}",
            "mode": request.mode,
        }
        record = save(
            status="running",
            progress=0.02,
            error=None,
            engine="ltx-native",
            plan=plan,
        )

        workspace = Path(f"/tmp/ltx25-native/{job_id}")
        input_dir = workspace / "inputs"
        shutil.rmtree(workspace, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        for condition in request.conditions:
            asset_record = job_store.get(f"asset:{condition.asset_id}")
            if not isinstance(asset_record, dict):
                raise FileNotFoundError(f"Input asset metadata not found: {condition.asset_id}")
            ref = MediaRef.from_value(asset_record, default_store_id=media_storage.primary_id)
            suffix = Path(ref.key).suffix
            target_input = input_dir / f"{condition.asset_id}{suffix}"
            media_storage.download_to(ref, target_input)

        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        save(progress=0.08)

        suffix = ".wav" if request.mode == "t2a" else ".mp4"
        prefix = {"t2a": "t2a", "keyframe_interpolation": "keyframe", "dfr": "dfr"}[request.mode]
        target = (
            Path("/data/outputs") / f"{prefix}_{job_id}{suffix}"
            if MEDIA_PRIMARY_BACKEND == "volume"
            else workspace / f"{prefix}_{job_id}{suffix}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        validate_native_assets(request.mode)
        command = build_native_command(
            request,
            input_dir,
            target,
            lora_dir=Path("/data/loras"),
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-4000:]
            raise RuntimeError(f"ltx-pipelines {request.mode} failed: {tail}")
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"ltx-pipelines {request.mode} produced no output")

        hdr_archive = None
        if request.hdr_color_space is not None:
            exr_dir = target.with_name(f"{target.stem}_exr")
            if not exr_dir.is_dir() or not any(exr_dir.glob("*.exr")):
                raise RuntimeError("HDR generation produced no EXR frame sequence")
            archive_base = target.with_name(f"{target.stem}_exr")
            hdr_archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=exr_dir))

        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        save(progress=0.95)

        output_size = target.stat().st_size
        if MEDIA_PRIMARY_BACKEND == "volume" and not MEDIA_FALLBACK_ID:
            state_volume.commit()
            output_ref = MediaRef(media_storage.primary_id, f"outputs/{target.name}")
        else:
            output_ref = media_storage.upload_local(target, f"outputs/{target.name}")
            target.unlink(missing_ok=True)
        job_store.put(
            f"output:{Path(output_ref.key).name}",
            {**output_ref.as_dict(), "size": output_size},
        )

        hdr_exr_key = None
        if hdr_archive is not None:
            archive_size = hdr_archive.stat().st_size
            if MEDIA_PRIMARY_BACKEND == "volume" and not MEDIA_FALLBACK_ID:
                state_volume.commit()
                archive_ref = MediaRef(media_storage.primary_id, f"outputs/{hdr_archive.name}")
            else:
                archive_ref = media_storage.upload_local(hdr_archive, f"outputs/{hdr_archive.name}")
                hdr_archive.unlink(missing_ok=True)
            job_store.put(
                f"output:{Path(archive_ref.key).name}",
                {**archive_ref.as_dict(), "size": archive_size},
            )
            hdr_exr_key = f"outputs/{Path(archive_ref.key).name}"

        output_key = f"outputs/{Path(output_ref.key).name}"
        record = save(
            status="completed",
            progress=1.0,
            error=None,
            generation_seconds=time.monotonic() - started,
            peak_vram_gb=None,
            engine="ltx-native",
            plan=plan,
            graph=None,
            image_key=None,
            video_key=output_key if request.mode != "t2a" else None,
            audio_key=output_key if request.mode == "t2a" else None,
            hdr_exr_key=hdr_exr_key,
            image_url=None,
            video_url=None,
            audio_url=None,
            hdr_exr_url=None,
        )
        return record
    except Exception as exc:
        interrupted = _honor_interrupt(job_id, record)
        if interrupted is not None:
            return interrupted
        return save(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            generation_seconds=time.monotonic() - started,
            engine="ltx-native",
        )
    finally:
        shutil.rmtree(Path(f"/tmp/ltx25-native/{job_id}"), ignore_errors=True)


@app.local_entrypoint()
def prepare() -> None:
    """Run once (or rerun safely) before deploy: modal run modal_app.py::prepare."""
    # Qwen writes to its own Volume, so it can safely download in parallel with
    # the resident LTX preparation. Resident and native LTX both write to the
    # shared model Volume and therefore remain serialized to avoid commit races.
    qwen_call = prepare_qwen_image21.spawn() if DIRECTOR_QWEN_ENABLED else None

    resident = prepare_models.remote()
    qwen = qwen_call.get() if qwen_call is not None else {"status": "disabled"}

    print({
        "resident": resident,
        "qwen": qwen,
        "native": {"status": "ready", "path": resident["native"]},
    })
