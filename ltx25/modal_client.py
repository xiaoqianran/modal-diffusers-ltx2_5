"""Modal SDK client for the primary local API.

This module owns all Modal SDK details: worker lookup, job state, warmup,
Volume I/O, and cancellation. HTTP/FastAPI code lives in api.py.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from .schemas import GenerateRequest, LoraResponse


APP_NAME = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")
JOB_DICT_NAME = os.environ.get("LTX25_MODAL_JOB_DICT", "ltx25-jobs")
MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
ACTIVE_STATUSES = {"queued", "running"}


class ActiveJobError(RuntimeError):
    pass


class SubmissionError(RuntimeError):
    pass


class JobStateError(RuntimeError):
    pass


class ModalOperationError(RuntimeError):
    pass


class ModalClient:
    """Single client boundary between the local API and Modal."""

    def __init__(self) -> None:
        self.app_name = APP_NAME
        self.model_id = MODEL_ID
        self.gpu_idle_seconds = int(os.environ.get("LTX25_MODAL_GPU_IDLE_SECONDS", "600"))
        self.keep_gpu_warm = os.environ.get("LTX25_LOCAL_KEEP_GPU_WARM", "1").strip().lower() not in {
            "", "0", "false", "no", "off"
        }

        self.cache_root = Path(os.environ.get("LTX25_LOCAL_CACHE", ".ltx25-cache")).resolve()
        self.output_cache = self.cache_root / "outputs"
        self.upload_cache = self.cache_root / "uploads"

        self.state_volume = modal.Volume.from_name(STATE_VOLUME_NAME)
        self.job_store = modal.Dict.from_name(JOB_DICT_NAME)

        worker_cls = modal.Cls.from_name(self.app_name, "LTX25Worker")
        self.worker = worker_cls()
        self.generate_fn = self.worker.generate
        self.ready_fn = self.worker.ready

        self._warm_call = None
        self._output_lock = threading.Lock()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def _public_job(record: dict[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in record.items() if name != "call_id"}

    def _save(self, record: dict[str, Any]) -> dict[str, Any]:
        record["updated_at"] = self._utc_now()
        self.job_store.put(self._key(record["id"]), record)
        return record

    def _get_record(self, job_id: str) -> dict[str, Any] | None:
        return self.job_store.get(self._key(job_id))

    def _fail_record(self, record: dict[str, Any], exc: Exception) -> dict[str, Any]:
        current = self._get_record(record["id"]) or record
        if current.get("status") in ACTIVE_STATUSES:
            current["status"] = "failed"
            current["error"] = f"Modal worker failed: {type(exc).__name__}: {exc}"
            self._save(current)
        return current

    def _refresh(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("status") not in ACTIVE_STATUSES:
            return record

        call_id = self.job_store.get(f"call:{record['id']}") or record.get("call_id")
        if not call_id:
            return record

        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except modal.exception.FunctionTimeoutError as exc:
            return self._fail_record(record, exc)
        except (TimeoutError, modal.exception.TimeoutError):
            return record
        except (modal.exception.ConnectionError, modal.exception.InternalError):
            return record
        except Exception as exc:
            return self._fail_record(record, exc)

        if isinstance(result, dict):
            current = self._get_record(record["id"]) or record
            if current.get("status") in ACTIVE_STATUSES:
                current.update(result)
                self._save(current)
                return current

        return self._get_record(record["id"]) or record

    def set_idle_window(self, seconds: int) -> None:
        self.worker.update_autoscaler(min_containers=0, scaledown_window=seconds)

    def start_warmup(self) -> None:
        self._warm_call = self.ready_fn.spawn()

    def warm_status(self) -> dict[str, Any]:
        if self._warm_call is None:
            return {"state": "disabled" if not self.keep_gpu_warm else "idle"}
        try:
            result = self._warm_call.get(timeout=0)
            return {"state": "ready", **(result if isinstance(result, dict) else {})}
        except (TimeoutError, modal.exception.TimeoutError):
            return {"state": "warming"}
        except Exception as exc:
            return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def create_session() -> int:
        return int(time.time() * 1000)

    def create_job(self, request: GenerateRequest) -> dict[str, Any]:
        session_number = request.session_number or self.create_session()
        request.session_number = session_number
        now = self._utc_now()
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
        self._save(record)

        try:
            call = self.generate_fn.spawn(job_id, request.model_dump(mode="json"))
            self.job_store.put(f"call:{job_id}", call.object_id)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"Could not submit GPU job: {type(exc).__name__}: {exc}"
            self._save(record)
            raise SubmissionError(record["error"]) from exc

        return self._public_job(self._get_record(job_id) or record)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        record = self._get_record(job_id)
        if not record:
            return None
        return self._public_job(self._refresh(record))

    def list_jobs(self, session_number: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item_key, item in self.job_store.items():
            if not isinstance(item_key, str) or not item_key.startswith("job:") or not isinstance(item, dict):
                continue
            if session_number is not None and item.get("session_number") != session_number:
                continue
            records.append(item)

        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return [
            self._public_job(self._refresh(item))
            for item in records[: min(max(limit, 1), 200)]
        ]

    def delete_job(self, job_id: str) -> bool:
        record = self._get_record(job_id)
        if not record:
            return False
        if record.get("status") in ACTIVE_STATUSES:
            raise ActiveJobError("A queued or running job cannot be deleted")

        self.remove_output(f"{job_id}.mp4")
        for prefix in ("t2i", "refine", "ref2i"):
            self.remove_output(f"{prefix}_{job_id}.png")
        self.job_store.pop(self._key(job_id), None)
        self.job_store.pop(f"call:{job_id}", None)
        return True

    def interrupt(self, job_id: str | None) -> dict[str, Any]:
        target = self._get_record(job_id) if job_id else None
        if target is None and job_id is None:
            for item_key, item in self.job_store.items():
                if (
                    isinstance(item_key, str)
                    and item_key.startswith("job:")
                    and isinstance(item, dict)
                    and item.get("status") in ACTIVE_STATUSES
                ):
                    target = item
                    break

        if not target or target.get("status") not in ACTIVE_STATUSES:
            return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}

        call_id = self.job_store.get(f"call:{target['id']}") or target.get("call_id")
        if not call_id:
            raise JobStateError("Job submission is still in progress; retry shortly")

        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=False)
        except Exception as exc:
            raise ModalOperationError("Could not cancel the Modal job; retry shortly") from exc

        target = self._get_record(target["id"]) or target
        if target.get("status") in ACTIVE_STATUSES:
            target["status"] = "interrupted"
            target["error"] = "Interrupted by user"
            self._save(target)
            return {
                "interrupted": True,
                "current_job_id": target["id"],
                "requested_job_id": job_id,
            }

        return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        with self.state_volume.batch_upload(force=True) as batch:
            batch.put_file(str(local_path), remote_path)

    def download_output(self, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError("Invalid output filename")

        self.output_cache.mkdir(parents=True, exist_ok=True)
        target = self.output_cache / filename
        if target.is_file():
            return target

        with self._output_lock:
            if target.is_file():
                return target

            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with temporary.open("wb") as output:
                    for chunk in self.state_volume.read_file(f"outputs/{filename}"):
                        output.write(chunk)
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

        return target

    def remove_output(self, filename: str) -> None:
        try:
            self.state_volume.remove_file(f"outputs/{filename}")
        except FileNotFoundError:
            pass
        (self.output_cache / filename).unlink(missing_ok=True)

    def list_loras(self) -> list[LoraResponse]:
        try:
            entries = list(self.state_volume.iterdir("loras", recursive=False))
        except FileNotFoundError:
            entries = []

        items: list[LoraResponse] = []
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
