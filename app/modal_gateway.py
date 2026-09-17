from __future__ import annotations

import asyncio
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from fastapi import Body, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
ACTIVE_STATUSES = {"queued", "running"}


class VolumeAccessMiddleware:
    """Serialize volume access through the complete ASGI response lifetime.

    FileResponse opens files after the endpoint returns. A route-level lock
    cannot prevent a concurrent reload from racing with that open file.
    API health and job polling remain concurrent and never acquire this lock.
    """

    def __init__(self, app):
        self.app = app
        self.lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        uses_volume = (
            path.startswith("/outputs/")
            or path in {"/api/assets", "/api/loras", "/api/jobs/concat"}
            or (scope.get("method") == "DELETE" and path.startswith("/api/jobs/"))
        )
        if scope["type"] == "http" and uses_volume:
            async with self.lock:
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_job(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "call_id"}


def build_gateway(*, worker_cls, job_store, state_volume, state_root: str = "/data", model_id: str) -> FastAPI:
    root = Path(state_root)
    input_dir = root / "inputs"
    output_dir = root / "outputs"
    lora_dir = root / "loras"
    for directory in (input_dir, output_dir, lora_dir):
        directory.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="LTX-2.5 Modal Gateway", version="2.0.0")
    app.add_middleware(VolumeAccessMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    def key(job_id: str) -> str:
        return f"job:{job_id}"

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
            call = modal.FunctionCall.from_id(call_id)
            result = call.get(timeout=0)
        except modal.exception.FunctionTimeoutError as exc:
            # This subclasses Modal TimeoutError but means the GPU execution
            # expired, not that a non-blocking poll has no result yet.
            return fail_record(record, exc)
        except (TimeoutError, modal.exception.TimeoutError):
            return record
        except (modal.exception.ConnectionError, modal.exception.InternalError):
            # Transport/service outages say nothing about the GPU input state.
            # Let the next poll reconcile it instead of persisting a false failure.
            return record
        except Exception as exc:  # a terminal Modal failure after retries are exhausted
            return fail_record(record, exc)
        if isinstance(result, dict):
            current = get_record(record["id"]) or record
            if current.get("status") not in ACTIVE_STATUSES:
                return current
            current.update(result)
            save(current)
            return current
        return get_record(record["id"]) or record

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model": model_id,
            "loaded": False,
            "worker": "on-demand",
            "gpu_attached_to_web": False,
            "prompt_enhancer": False,
            "transformer_precision": "nvfp4",
        }

    @app.post("/api/sessions", response_model=SessionResponse, status_code=201)
    def create_session():
        # Unique enough for UI grouping while avoiding a separate transactional counter.
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
            call = worker_cls().generate.spawn(job_id, request.model_dump(mode="json"))
            # Store submission metadata separately: a fast worker can already
            # have updated the job before spawn returns. Never overwrite it.
            job_store.put(f"call:{job_id}", call.object_id)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"Could not submit GPU job: {type(exc).__name__}: {exc}"
            save(record)
            raise HTTPException(status_code=503, detail=record["error"]) from exc
        return _public_job(get_record(job_id) or record)

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        record = get_record(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found")
        return _public_job(refresh(record))

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
        return [_public_job(refresh(item)) for item in records[: min(max(limit, 1), 200)]]

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        record = get_record(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.get("status") in ACTIVE_STATUSES:
            raise HTTPException(status_code=409, detail="A queued or running job cannot be deleted")
        state_volume.reload()
        (output_dir / f"{job_id}.mp4").unlink(missing_ok=True)
        for prefix in ("t2i", "refine", "ref2i"):
            (output_dir / f"{prefix}_{job_id}.png").unlink(missing_ok=True)
        state_volume.commit()
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
        if call_id:
            try:
                modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
            except Exception as exc:
                raise HTTPException(status_code=502, detail="Could not cancel the Modal job; retry shortly") from exc
        target = get_record(target["id"]) or target
        if target.get("status") not in ACTIVE_STATUSES:
            return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}
        target["status"] = "interrupted"
        target["error"] = "Interrupted by user"
        save(target)
        return {
            "interrupted": True,
            "current_job_id": target["id"],
            "requested_job_id": job_id,
        }

    @app.post("/api/admin/unload")
    def admin_unload():
        # GPU workers scale down independently after their idle window; web stays CPU-only.
        return {"result": "on-demand worker; GPU releases automatically after idle timeout"}

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
        temporary = input_dir / f".{asset_id}{suffix}.part"
        destination = input_dir / f"{asset_id}{suffix}"
        max_bytes = 1024 * 1024 * 1024
        size = 0
        try:
            with temporary.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(status_code=413, detail="Upload exceeds 1 GiB")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            if kind == "image":
                with Image.open(temporary) as image:
                    image.verify()
            elif kind in {"video", "audio"} and suffix != ".gif":
                selector = "v:0" if kind == "video" else "a:0"
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", selector, "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(temporary)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if probe.returncode != 0 or not probe.stdout.strip():
                    raise HTTPException(status_code=400, detail=f"Uploaded {kind} could not be decoded")
            temporary.replace(destination)
            state_volume.commit()
        except HTTPException:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid {kind} file") from exc
        finally:
            file.file.close()
        return AssetResponse(id=asset_id, kind=kind, filename=file.filename or destination.name, size=size)

    @app.get("/api/loras", response_model=list[LoraResponse])
    def list_loras():
        state_volume.reload()
        items = []
        for path in sorted(lora_dir.glob("*.safetensors"), key=lambda item: item.name.lower()):
            if path.is_file():
                lowered = path.name.lower()
                items.append(
                    LoraResponse(
                        id=path.name,
                        name=path.stem,
                        size=path.stat().st_size,
                        kind="iclora" if "ic-lora" in lowered or "iclora" in lowered else "standard",
                    )
                )
        return items

    @app.post("/api/prompts/enhance", response_model=PromptEnhanceResponse)
    def enhance_prompt(_: PromptEnhanceRequest):
        raise HTTPException(status_code=503, detail="Prompt enhancer is not configured on the CPU gateway")

    @app.post("/api/jobs/concat")
    def concat_jobs(request: ConcatRequest):
        state_volume.reload()
        sources: list[Path] = []
        for job_id in request.job_ids:
            record = get_record(job_id)
            path = output_dir / f"{job_id}.mp4"
            if record is None or record.get("status") != "completed" or not path.is_file():
                raise HTTPException(status_code=404, detail=f"Completed video not found: {job_id}")
            sources.append(path)
        concat_id = uuid.uuid4().hex
        list_file = output_dir / f".concat-{concat_id}.txt"
        target = output_dir / f"combined-{concat_id}.mp4"
        try:
            list_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in sources), encoding="utf-8")
            result = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", "-movflags", "+faststart", str(target)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"Video concatenation failed: {result.stderr[-500:]}")
            state_volume.commit()
        finally:
            list_file.unlink(missing_ok=True)
        return {"video_url": f"/outputs/{target.name}", "filename": target.name}

    @app.api_route("/outputs/{filename}", methods=["GET", "HEAD"])
    def output_file(filename: str):
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid output filename")
        path = output_dir / filename
        # Outputs are immutable; only refresh when this container has not seen
        # the file yet. Range requests then avoid an extra Volume RPC each time.
        if not path.is_file():
            state_volume.reload()
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Output not found")
        media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "image/png"
        return FileResponse(path, media_type=media_type)

    return app
