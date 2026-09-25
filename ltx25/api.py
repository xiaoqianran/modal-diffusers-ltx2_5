"""Primary browser-facing API.

The API layer owns HTTP concerns and local media validation only. Modal SDK,
job state, worker lifecycle, and Volume access live in modal_client.py.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from PIL import Image

from .media_store import MediaStoreError, media_content_type
from .schemas import (
    AssetResponse,
    AssetUploadCompleteRequest,
    AssetUploadPrepareRequest,
    AssetUploadPrepareResponse,
    ConcatRequest,
    GenerateRequest,
    JobResponse,
    JobSummaryResponse,
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
    QueueFullError,
    SubmissionError,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".exr"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
logger = logging.getLogger(__name__)

modal_client = ModalClient()
CACHE_ROOT = modal_client.cache_root
OUTPUT_CACHE = modal_client.output_cache
UPLOAD_CACHE = modal_client.upload_cache


def _prune_local_upload_cache(max_age_seconds: int = 24 * 60 * 60) -> int:
    """Delete crash leftovers; successful uploads are removed immediately."""
    if not UPLOAD_CACHE.exists():
        return 0
    now = time.time()
    removed = 0
    for path in UPLOAD_CACHE.iterdir():
        try:
            if path.is_file() and now - path.stat().st_mtime >= max_age_seconds:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


async def _keep_warm_loop() -> None:
    interval = max(10, min(modal_client.warm_lease_seconds // 3, 30))
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(modal_client.maintain_keep_warm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("keep-warm maintenance failed; retrying on next tick", exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    _prune_local_upload_cache()
    try:
        await asyncio.to_thread(modal_client.cleanup_assets)
    except Exception:
        pass

    keep_warm_task = asyncio.create_task(_keep_warm_loop())
    if modal_client.keep_gpu_warm:
        await asyncio.to_thread(modal_client.enable_keep_warm)

    try:
        yield
    finally:
        keep_warm_task.cancel()
        with suppress(asyncio.CancelledError):
            await keep_warm_task
        try:
            await asyncio.to_thread(modal_client.set_idle_window, 2)
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
        "queue": modal_client.queue_stats(),
        "media": modal_client.media_info(),
        "transfer_metrics": modal_client.transfer_metrics(),
    }


@app.post("/api/admin/warm")
def admin_warm():
    modal_client.enable_keep_warm()
    return {"status": "warming"}


@app.post("/api/admin/unload")
def admin_unload():
    modal_client.disable_keep_warm()
    return {"result": "GPU worker will scale to zero after its current input becomes idle"}


@app.post("/api/sessions", response_model=SessionResponse, status_code=201)
def create_session():
    return SessionResponse(session_number=modal_client.create_session())


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
def create_job(request: GenerateRequest):
    try:
        return modal_client.create_job(request)
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SubmissionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/jobs/status", response_model=list[JobSummaryResponse])
def list_job_status(session_number: int | None = None, limit: int = 50):
    """Compact polling endpoint; full request payloads stay on the detail route."""
    return modal_client.list_job_summaries(session_number, limit)


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
    kind, suffix = _upload_kind(file.filename or "")

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

        _validate_uploaded_media(local_path, kind, suffix)

        remote_path = f"inputs/{asset_id}{suffix}"
        media_ref = modal_client.upload_file(local_path, remote_path)
        local_path.unlink(missing_ok=True)
        try:
            modal_client.register_asset(
                asset_id,
                remote_path,
                kind=kind,
                filename=file.filename or local_path.name,
                size=size,
                store_id=media_ref.store_id,
            )
        except Exception as exc:
            try:
                modal_client.remove_input(remote_path)
            except Exception:
                pass
            raise HTTPException(status_code=502, detail="Uploaded asset could not be registered") from exc
        try:
            modal_client.cleanup_assets()
        except Exception:
            # Retention cleanup is best-effort and must never turn a valid upload
            # into a failed user request.
            pass
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


def _upload_kind(filename: str) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image", suffix
    if suffix in VIDEO_SUFFIXES:
        return "video", suffix
    if suffix in AUDIO_SUFFIXES:
        return "audio", suffix
    raise HTTPException(status_code=415, detail="Unsupported media type")


def _validate_uploaded_media(local_path: Path, kind: str, suffix: str) -> None:
    if kind == "image":
        if suffix == ".exr":
            # Native LTX validates OpenEXR structure while loading. Pillow builds
            # commonly lack OpenEXR support, so do not reject valid EXR plates here.
            return
        with Image.open(local_path) as image:
            image.verify()
        return
    if suffix == ".gif":
        return
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return
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


@app.post("/api/assets/prepare", response_model=AssetUploadPrepareResponse, status_code=201)
def prepare_asset_upload(request: AssetUploadPrepareRequest):
    kind, suffix = _upload_kind(request.filename)
    try:
        return modal_client.prepare_asset_upload(
            filename=request.filename,
            content_type=request.content_type,
            size=request.size,
            kind=kind,
            suffix=suffix,
        )
    except (ValueError, MediaStoreError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not prepare media upload") from exc


@app.post("/api/assets/{asset_id}/fallback", response_model=AssetUploadPrepareResponse)
def prepare_asset_fallback(asset_id: str):
    try:
        return modal_client.prepare_asset_fallback(asset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Upload intent not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not prepare fallback media upload") from exc


@app.put("/api/assets/{asset_id}/content", status_code=204)
async def upload_asset_content(asset_id: str, request: Request):
    intent = await asyncio.to_thread(modal_client.get_upload_intent, asset_id)
    if not intent:
        raise HTTPException(status_code=404, detail="Upload intent not found")
    if intent.get("mode") != "proxy":
        raise HTTPException(status_code=409, detail="This upload must go directly to object storage")

    suffix = Path(str(intent["key"])).suffix.lower()
    kind = str(intent["kind"])
    expected_size = int(intent["size"])
    UPLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    local_path = UPLOAD_CACHE / f"{asset_id}{suffix}.part"
    received = 0
    try:
        with local_path.open("wb") as output:
            async for chunk in request.stream():
                received += len(chunk)
                if received > expected_size:
                    raise HTTPException(status_code=413, detail="Upload exceeds declared size")
                output.write(chunk)
        if received != expected_size:
            raise HTTPException(
                status_code=400,
                detail=f"Upload size mismatch: expected {expected_size}, got {received}",
            )
        _validate_uploaded_media(local_path, kind, suffix)
        await asyncio.to_thread(modal_client.upload_proxy_file, asset_id, local_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {kind} file: {exc}") from exc
    finally:
        local_path.unlink(missing_ok=True)
    return Response(status_code=204)


@app.post("/api/assets/{asset_id}/complete", response_model=AssetResponse)
def complete_asset_upload(asset_id: str, request: AssetUploadCompleteRequest):
    try:
        asset = modal_client.complete_asset_upload(
            asset_id,
            [item.model_dump() for item in request.parts],
            request.client_upload_seconds,
        )
        try:
            modal_client.cleanup_assets()
        except Exception:
            pass
        return AssetResponse(**asset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Upload intent or uploaded object not found") from exc
    except MediaStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not finalize media upload") from exc


@app.delete("/api/assets/{asset_id}/upload", status_code=204)
def abort_asset_upload(asset_id: str):
    modal_client.abort_asset_upload(asset_id)
    return Response(status_code=204)


@app.post("/api/assets/from-job/{job_id}", response_model=AssetResponse, status_code=201)
def asset_from_job(job_id: str):
    """Reuse a completed video entirely inside Modal storage."""
    try:
        return AssetResponse(**modal_client.copy_output_to_input(job_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Completed video output not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not reuse output inside Modal storage") from exc


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
        media_ref = modal_client.upload_file(target, f"outputs/{target.name}")
        modal_client.register_output(target.name, media_ref, size=target.stat().st_size)
    finally:
        list_path.unlink(missing_ok=True)

    return {"video_url": f"/outputs/{target.name}", "filename": target.name}


@app.api_route("/outputs/{filename}", methods=["GET", "HEAD"])
def output_file(filename: str, request: Request):
    try:
        redirect_url = modal_client.output_delivery_url(filename, method=request.method)
        if redirect_url:
            return RedirectResponse(redirect_url, status_code=307)
        path = modal_client.download_output(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Output not found") from exc

    return FileResponse(path, media_type=media_content_type(path.name))
