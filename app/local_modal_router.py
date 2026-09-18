from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from fastapi import Body, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image

from .schemas import (
    AssetResponse,
    ConcatRequest,
    GenerateRequest,
    JobResponse,
    LoraResponse,
    PromptEnhanceRequest,
    PromptEnhanceResponse,
    SessionResponse,
    STILL_IMAGE_MODES,
)


APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")
JOB_DICT_NAME = os.environ.get("LTX25_MODAL_JOB_DICT", "ltx25-jobs")
MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
ACTIVE_STATUSES = {"queued", "running"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

CACHE_ROOT = Path(os.environ.get("LTX25_LOCAL_CACHE", ".ltx25-cache")).resolve()
OUTPUT_CACHE = CACHE_ROOT / "outputs"
UPLOAD_CACHE = CACHE_ROOT / "uploads"
GPU_IDLE_SECONDS = int(os.environ.get("LTX25_MODAL_GPU_IDLE_SECONDS", "600"))
KEEP_GPU_WARM = os.environ.get("LTX25_LOCAL_KEEP_GPU_WARM", "1").strip().lower() not in {
    "", "0", "false", "no", "off"
}

state_volume = modal.Volume.from_name(STATE_VOLUME_NAME)
job_store = modal.Dict.from_name(JOB_DICT_NAME)
Worker = modal.Cls.from_name(APP_NAME, "LTX25Worker")
worker = Worker()
generate_fn = worker.generate
ready_fn = worker.ready
warm_call = None
output_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def key(job_id: str) -> str:
    return f"job:{job_id}"


def public_job(record: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in record.items() if name != "call_id"}


def save(record: dict[str, Any]) -> dict[str, Any]:
    record["updated_at"] = utc_now()
    job_store.put(key(record["id"]), record)
    return record


def get_record(job_id: str) -> dict[str, Any] | None:
    return job_store.get(key(job_id))


def fail_record(record: dict[str, Any], exc: Exception) -> dict[str, Any]:
    current = get_record(record["id"]) or record
    if current.get("status") in ACTIVE_STATUSES:
        current["status"] = "failed"
        current["error"] = f"Modal worker failed: {type(exc).__name__}: {exc}"
        save(current)
    return current


def refresh(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("status") not in ACTIVE_STATUSES:
        return record
    call_id = job_store.get(f"call:{record['id']}") or record.get("call_id")
    if not call_id:
        return record
    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except modal.exception.FunctionTimeoutError as exc:
        return fail_record(record, exc)
    except (TimeoutError, modal.exception.TimeoutError):
        return record
    except (modal.exception.ConnectionError, modal.exception.InternalError):
        return record
    except Exception as exc:
        return fail_record(record, exc)
    if isinstance(result, dict):
        current = get_record(record["id"]) or record
        if current.get("status") in ACTIVE_STATUSES:
            current.update(result)
            save(current)
            return current
    return get_record(record["id"]) or record


def _set_idle_window(seconds: int) -> None:
    worker.update_autoscaler(
        min_containers=0,
        scaledown_window=seconds,
    )


def _start_warmup() -> None:
    global warm_call
    warm_call = ready_fn.spawn()


async def _keep_warm_loop() -> None:
    # Avoid a persistent min_containers=1 floor: if the local app crashes, a
    # min-container override could keep billing indefinitely. A lightweight
    # ready() heartbeat keeps the worker hot only while this local router lives.
    interval = max(60, min(GPU_IDLE_SECONDS // 2, 300))
    while True:
        await asyncio.sleep(interval)
        await asyncio.to_thread(_start_warmup)


def _warm_status() -> dict[str, Any]:
    if warm_call is None:
        return {"state": "disabled" if not KEEP_GPU_WARM else "idle"}
    try:
        result = warm_call.get(timeout=0)
        return {"state": "ready", **(result if isinstance(result, dict) else {})}
    except (TimeoutError, modal.exception.TimeoutError):
        return {"state": "warming"}
    except Exception as exc:
        return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}


def _upload_file(local_path: Path, remote_path: str) -> None:
    with state_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote_path)


def _download_output(filename: str) -> Path:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid output filename")
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_CACHE / filename
    if target.is_file():
        return target
    with output_lock:
        if target.is_file():
            return target
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            with temporary.open("wb") as output:
                for chunk in state_volume.read_file(f"outputs/{filename}"):
                    output.write(chunk)
            temporary.replace(target)
        except FileNotFoundError as exc:
            temporary.unlink(missing_ok=True)
            raise HTTPException(status_code=404, detail="Output not found") from exc
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return target


def _remove_output(filename: str) -> None:
    try:
        state_volume.remove_file(f"outputs/{filename}")
    except FileNotFoundError:
        pass
    (OUTPUT_CACHE / filename).unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    keep_warm_task = None
    if KEEP_GPU_WARM:
        await asyncio.to_thread(_set_idle_window, GPU_IDLE_SECONDS)
        await asyncio.to_thread(_start_warmup)
        keep_warm_task = asyncio.create_task(_keep_warm_loop())
    try:
        yield
    finally:
        if keep_warm_task is not None:
            keep_warm_task.cancel()
        if KEEP_GPU_WARM:
            try:
                await asyncio.to_thread(_set_idle_window, 120)
            except Exception:
                pass


app = FastAPI(title="LTX-2.5 Local Modal Router", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "worker": "modal-direct",
        "transport": "local-modal-sdk",
        "gpu_attached_to_web": False,
        "transformer_precision": "nvfp4",
        "keep_gpu_warm": KEEP_GPU_WARM,
        "gpu_idle_seconds": GPU_IDLE_SECONDS,
        "warmup": _warm_status(),
    }


@app.post("/api/admin/warm")
def admin_warm():
    _set_idle_window(GPU_IDLE_SECONDS)
    _start_warmup()
    return {"status": "warming"}


@app.post("/api/admin/unload")
def admin_unload():
    _set_idle_window(2)
    return {"result": "GPU worker will scale to zero after its current input becomes idle"}


@app.post("/api/sessions", response_model=SessionResponse, status_code=201)
def create_session():
    return SessionResponse(session_number=int(time.time() * 1000))


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
def create_job(request: GenerateRequest):
    session_number = request.session_number or int(time.time() * 1000)
    request.session_number = session_number
    now = utc_now()
    job_id = uuid.uuid4().hex
    record: dict[str, Any] = {
        "id": job_id,
        "session_number": session_number,
        "status": "queued",
        "progress": 0.0,
        "error": None,
        "video_url": None,
        "image_url": None,
        "request": request.model_dump(mode="json"),
        "created_at": now,
        "updated_at": now,
        "generation_seconds": None,
        "peak_vram_gb": None,
        "call_id": None,
    }
    save(record)
    try:
        call = generate_fn.spawn(job_id, request.model_dump(mode="json"))
        job_store.put(f"call:{job_id}", call.object_id)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"Could not submit GPU job: {type(exc).__name__}: {exc}"
        save(record)
        raise HTTPException(status_code=503, detail=record["error"]) from exc
    return public_job(get_record(job_id) or record)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    record = get_record(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return public_job(refresh(record))


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs(session_number: int | None = None, limit: int = 50):
    records: list[dict[str, Any]] = []
    for item_key, item in job_store.items():
        if not isinstance(item_key, str) or not item_key.startswith("job:") or not isinstance(item, dict):
            continue
        if session_number is not None and item.get("session_number") != session_number:
            continue
        records.append(item)
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [public_job(refresh(item)) for item in records[: min(max(limit, 1), 200)]]


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    record = get_record(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    if record.get("status") in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="A queued or running job cannot be deleted")
    _remove_output(f"{job_id}.mp4")
    for prefix in ("t2i", "refine", "ref2i"):
        _remove_output(f"{prefix}_{job_id}.png")
    job_store.pop(key(job_id), None)
    job_store.pop(f"call:{job_id}", None)
    return Response(status_code=204)


@app.post("/api/interrupt")
def interrupt(job_id: str | None = Body(None, embed=True)):
    target = get_record(job_id) if job_id else None
    if target is None and job_id is None:
        for item_key, item in job_store.items():
            if isinstance(item_key, str) and item_key.startswith("job:") and isinstance(item, dict):
                if item.get("status") in ACTIVE_STATUSES:
                    target = item
                    break
    if not target or target.get("status") not in ACTIVE_STATUSES:
        return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}
    call_id = job_store.get(f"call:{target['id']}") or target.get("call_id")
    if not call_id:
        raise HTTPException(status_code=409, detail="Job submission is still in progress; retry shortly")
    try:
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not cancel the Modal job; retry shortly") from exc
    target = get_record(target["id"]) or target
    if target.get("status") in ACTIVE_STATUSES:
        target["status"] = "interrupted"
        target["error"] = "Interrupted by user"
        save(target)
        return {"interrupted": True, "current_job_id": target["id"], "requested_job_id": job_id}
    return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}


@app.post("/api/assets", response_model=AssetResponse, status_code=201)
def upload_asset(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        kind = "image"
    elif suffix in VIDEO_SUFFIXES:
        kind = "video"
    elif suffix in AUDIO_SUFFIXES:
        kind = "audio"
    else:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    asset_id = uuid.uuid4().hex
    UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    local_path = UPLOAD_CACHE / f"{asset_id}{suffix}"
    size = 0
    try:
        with local_path.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Upload exceeds 1 GiB")
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if kind == "image":
            with Image.open(local_path) as image:
                image.verify()
        elif kind in {"video", "audio"} and suffix != ".gif":
            ffprobe = shutil.which("ffprobe")
            if ffprobe:
                selector = "v:0" if kind == "video" else "a:0"
                probe = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", selector, "-show_entries",
                     "stream=codec_name", "-of", "csv=p=0", str(local_path)],
                    capture_output=True, text=True, timeout=30,
                )
                if probe.returncode != 0 or not probe.stdout.strip():
                    raise HTTPException(status_code=400, detail=f"Uploaded {kind} could not be decoded")
        _upload_file(local_path, f"inputs/{asset_id}{suffix}")
    except HTTPException:
        local_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        local_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid {kind} file: {exc}") from exc
    finally:
        file.file.close()
    return AssetResponse(id=asset_id, kind=kind, filename=file.filename or local_path.name, size=size)


@app.get("/api/loras", response_model=list[LoraResponse])
def list_loras():
    items = []
    try:
        entries = list(state_volume.iterdir("loras", recursive=False))
    except FileNotFoundError:
        entries = []
    for entry in entries:
        path = Path(entry.path)
        if path.suffix.lower() != ".safetensors":
            continue
        lowered = path.name.lower()
        items.append(
            LoraResponse(
                id=path.name,
                name=path.stem,
                size=getattr(entry, "size", 0),
                kind="iclora" if "ic-lora" in lowered or "iclora" in lowered else "standard",
            )
        )
    return sorted(items, key=lambda item: item.name.lower())


@app.post("/api/prompts/enhance", response_model=PromptEnhanceResponse)
def enhance_prompt(_: PromptEnhanceRequest):
    raise HTTPException(status_code=503, detail="Prompt enhancer is not configured in the local router")


@app.post("/api/jobs/concat")
def concat_jobs(request: ConcatRequest):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="ffmpeg is required locally for concat")
    sources: list[Path] = []
    for job_id in request.job_ids:
        record = get_record(job_id)
        if record is None or record.get("status") != "completed":
            raise HTTPException(status_code=404, detail=f"Completed video not found: {job_id}")
        sources.append(_download_output(f"{job_id}.mp4"))

    concat_id = uuid.uuid4().hex
    target = OUTPUT_CACHE / f"combined-{concat_id}.mp4"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as listing:
        list_path = Path(listing.name)
        for source in sources:
            escaped = source.as_posix().replace("'", "'\\''")
            listing.write(f"file '{escaped}'\n")
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c", "copy", "-movflags", "+faststart", str(target)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Video concatenation failed: {result.stderr[-500:]}")
        _upload_file(target, f"outputs/{target.name}")
    finally:
        list_path.unlink(missing_ok=True)
    return {"video_url": f"/outputs/{target.name}", "filename": target.name}


@app.api_route("/outputs/{filename}", methods=["GET", "HEAD"])
def output_file(filename: str):
    path = _download_output(filename)
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/png"
    return FileResponse(path, media_type=media_type)
