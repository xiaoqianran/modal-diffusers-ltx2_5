"""Primary browser-facing API.

The API layer owns HTTP concerns and local media validation only. Modal SDK,
job state, worker lifecycle, and Volume access live in modal_client.py.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
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
)
from .modal_client import (
    ActiveJobError,
    JobStateError,
    ModalClient,
    ModalOperationError,
    SubmissionError,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

modal_client = ModalClient()
CACHE_ROOT = modal_client.cache_root
OUTPUT_CACHE = modal_client.output_cache
UPLOAD_CACHE = modal_client.upload_cache


async def _keep_warm_loop() -> None:
    interval = max(60, min(modal_client.gpu_idle_seconds // 2, 300))
    while True:
        await asyncio.sleep(interval)
        await asyncio.to_thread(modal_client.start_warmup)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)

    keep_warm_task = None
    if modal_client.keep_gpu_warm:
        await asyncio.to_thread(modal_client.set_idle_window, modal_client.gpu_idle_seconds)
        await asyncio.to_thread(modal_client.start_warmup)
        keep_warm_task = asyncio.create_task(_keep_warm_loop())

    try:
        yield
    finally:
        if keep_warm_task is not None:
            keep_warm_task.cancel()
        if modal_client.keep_gpu_warm:
            try:
                await asyncio.to_thread(modal_client.set_idle_window, 120)
            except Exception:
                pass


app = FastAPI(title="LTX-2.5 Local Modal Router", version="4.0.0", lifespan=lifespan)
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
        "model": modal_client.model_id,
        "worker": "modal-direct",
        "transport": "local-modal-sdk",
        "gpu_attached_to_web": False,
        "transformer_precision": "nvfp4",
        "keep_gpu_warm": modal_client.keep_gpu_warm,
        "gpu_idle_seconds": modal_client.gpu_idle_seconds,
        "warmup": modal_client.warm_status(),
    }


@app.post("/api/admin/warm")
def admin_warm():
    modal_client.set_idle_window(modal_client.gpu_idle_seconds)
    modal_client.start_warmup()
    return {"status": "warming"}


@app.post("/api/admin/unload")
def admin_unload():
    modal_client.set_idle_window(2)
    return {"result": "GPU worker will scale to zero after its current input becomes idle"}


@app.post("/api/sessions", response_model=SessionResponse, status_code=201)
def create_session():
    return SessionResponse(session_number=modal_client.create_session())


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
def create_job(request: GenerateRequest):
    try:
        return modal_client.create_job(request)
    except SubmissionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    record = modal_client.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs(session_number: int | None = None, limit: int = 50):
    return modal_client.list_jobs(session_number, limit)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    try:
        if not modal_client.delete_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
    except ActiveJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


@app.post("/api/interrupt")
def interrupt(job_id: str | None = Body(None, embed=True)):
    try:
        return modal_client.interrupt(job_id)
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModalOperationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-select_streams",
                        selector,
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "csv=p=0",
                        str(local_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if probe.returncode != 0 or not probe.stdout.strip():
                    raise HTTPException(status_code=400, detail=f"Uploaded {kind} could not be decoded")

        modal_client.upload_file(local_path, f"inputs/{asset_id}{suffix}")
    except HTTPException:
        local_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        local_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid {kind} file: {exc}") from exc
    finally:
        file.file.close()

    return AssetResponse(
        id=asset_id,
        kind=kind,
        filename=file.filename or local_path.name,
        size=size,
    )


@app.get("/api/loras", response_model=list[LoraResponse])
def list_loras():
    return modal_client.list_loras()


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
        record = modal_client.get_job(job_id)
        if record is None or record.get("status") != "completed":
            raise HTTPException(status_code=404, detail=f"Completed video not found: {job_id}")
        try:
            sources.append(modal_client.download_output(f"{job_id}.mp4"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Completed video not found: {job_id}") from exc

    concat_id = uuid.uuid4().hex
    target = OUTPUT_CACHE / f"combined-{concat_id}.mp4"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as listing:
        list_path = Path(listing.name)
        for source in sources:
            escaped = source.as_posix().replace("'", "'\''")
            listing.write(f"file '{escaped}'\n")

    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Video concatenation failed: {result.stderr[-500:]}")
        modal_client.upload_file(target, f"outputs/{target.name}")
    finally:
        list_path.unlink(missing_ok=True)

    return {"video_url": f"/outputs/{target.name}", "filename": target.name}


@app.api_route("/outputs/{filename}", methods=["GET", "HEAD"])
def output_file(filename: str):
    try:
        path = modal_client.download_output(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Output not found") from exc

    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/png"
    return FileResponse(path, media_type=media_type)
