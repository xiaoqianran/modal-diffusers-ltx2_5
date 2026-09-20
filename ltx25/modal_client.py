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

from .media_store import MediaStoreError, media_content_type
from .media_storage import MediaRef, create_media_storage
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
        self.media_storage = create_media_storage(self.state_volume)
        self.media_store = self.media_storage  # compatibility alias for tests/callers

        worker_cls = modal.Cls.from_name(self.app_name, "LTX25Worker")
        self.worker = worker_cls()
        self.generate_fn = self.worker.generate
        self.ready_fn = self.worker.ready

        self._warm_call = None
        self._warm_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._session_lock = threading.Lock()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def _cancel_key(job_id: str) -> str:
        return f"cancel:{job_id}"

    @staticmethod
    def _session_key(session_number: int) -> str:
        return f"session:{session_number}:jobs"

    @staticmethod
    def _asset_key(asset_id: str) -> str:
        return f"asset:{asset_id}"

    @staticmethod
    def _upload_key(asset_id: str) -> str:
        return f"upload:{asset_id}"

    @staticmethod
    def _output_key(filename: str) -> str:
        return f"output:{filename}"

    @staticmethod
    def _public_job(record: dict[str, Any]) -> dict[str, Any]:
        public = {
            name: value
            for name, value in record.items()
            if name not in {"call_id", "video_key", "image_key"}
        }
        video_key = record.get("video_key")
        image_key = record.get("image_key")
        if isinstance(video_key, str) and video_key:
            public["video_url"] = f"/outputs/{Path(video_key).name}"
        if isinstance(image_key, str) and image_key:
            public["image_url"] = f"/outputs/{Path(image_key).name}"
        return public

    @staticmethod
    def _job_summary(record: dict[str, Any]) -> dict[str, Any]:
        public = ModalClient._public_job(record)
        request = public.get("request") or {}
        public["request"] = {
            key: request.get(key)
            for key in ("mode", "prompt", "width", "height", "num_frames", "fps", "steps", "upscale")
        }
        return public

    def _save(self, record: dict[str, Any]) -> dict[str, Any]:
        record["updated_at"] = self._utc_now()
        self.job_store.put(self._key(record["id"]), record)
        return record

    def _get_record(self, job_id: str) -> dict[str, Any] | None:
        return self.job_store.get(self._key(job_id))

    def _index_job(self, session_number: int, job_id: str) -> None:
        """Keep a compact per-session lookup so UI polling does not scan all jobs."""
        key = self._session_key(session_number)
        with self._session_lock:
            job_ids = self.job_store.get(key) or []
            if job_id not in job_ids:
                self.job_store.put(key, [job_id, *job_ids])

    def _remove_job_from_index(self, session_number: int, job_id: str) -> None:
        key = self._session_key(session_number)
        with self._session_lock:
            job_ids = self.job_store.get(key) or []
            next_ids = [item for item in job_ids if item != job_id]
            if next_ids:
                self.job_store.put(key, next_ids)
            else:
                self.job_store.pop(key, None)

    def _fail_record(self, record: dict[str, Any], exc: Exception) -> dict[str, Any]:
        current = self._get_record(record["id"]) or record
        if current.get("status") in ACTIVE_STATUSES:
            current["status"] = "failed"
            current["error"] = f"Modal worker failed: {type(exc).__name__}: {exc}"
            self._save(current)
        return current

    def transfer_metrics(self) -> dict[str, Any]:
        metrics = self.media_storage.metrics()
        if self.media_storage.fallback_id is None:
            single = metrics["stores"][self.media_storage.primary_id]
            return {**single, "routing": metrics["routing"], "stores": metrics["stores"]}
        return metrics

    def media_info(self) -> dict[str, Any]:
        return {
            "backend": self.media_storage.backend,
            "primary_id": self.media_storage.primary_id,
            "fallback_id": self.media_storage.fallback_id,
            "direct_upload": self.media_storage.direct_upload,
            "direct_download": self.media_storage.direct_download,
        }

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

    def start_warmup(self) -> bool:
        """Start one warmup call unless keep-warm is disabled or one is already pending."""
        if not self.keep_gpu_warm:
            return False
        with self._warm_lock:
            if self._warm_call is not None:
                try:
                    self._warm_call.get(timeout=0)
                except (TimeoutError, modal.exception.TimeoutError):
                    return False
                except Exception:
                    pass
            self._warm_call = self.ready_fn.spawn()
            return True

    def enable_keep_warm(self) -> bool:
        self.keep_gpu_warm = True
        self.set_idle_window(self.gpu_idle_seconds)
        return self.start_warmup()

    def disable_keep_warm(self) -> None:
        self.keep_gpu_warm = False
        with self._warm_lock:
            warm_call = self._warm_call
            self._warm_call = None
        if warm_call is not None:
            try:
                warm_call.cancel(terminate_containers=False)
            except Exception:
                pass
        self.set_idle_window(2)

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
        self._index_job(session_number, job_id)

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
        if session_number is not None:
            job_ids = self.job_store.get(self._session_key(session_number))
            if isinstance(job_ids, list):
                records = [record for job_id in job_ids if (record := self._get_record(job_id))]
            else:
                # Backward-compatible one-time fallback for sessions created before
                # the per-session index existed; backfill it for subsequent polls.
                for item_key, item in self.job_store.items():
                    if (
                        isinstance(item_key, str)
                        and item_key.startswith("job:")
                        and isinstance(item, dict)
                        and item.get("session_number") == session_number
                    ):
                        records.append(item)
                records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
                with self._session_lock:
                    self.job_store.put(
                        self._session_key(session_number),
                        [item["id"] for item in records],
                    )
        else:
            for item_key, item in self.job_store.items():
                if isinstance(item_key, str) and item_key.startswith("job:") and isinstance(item, dict):
                    records.append(item)

        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return [
            self._public_job(self._refresh(item))
            for item in records[: min(max(limit, 1), 200)]
        ]

    def list_job_summaries(self, session_number: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [self._job_summary(item) for item in self.list_jobs(session_number, limit)]

    def delete_job(self, job_id: str) -> bool:
        record = self._get_record(job_id)
        if not record:
            return False
        if record.get("status") in ACTIVE_STATUSES:
            raise ActiveJobError("A queued or running job cannot be deleted")

        self.remove_output(f"{job_id}.mp4")
        for prefix in ("t2i", "refine", "ref2i"):
            self.remove_output(f"{prefix}_{job_id}.png")
        self._remove_job_from_index(record["session_number"], job_id)
        self.job_store.pop(self._key(job_id), None)
        self.job_store.pop(f"call:{job_id}", None)
        self.job_store.pop(self._cancel_key(job_id), None)
        return True

    def interrupt(self, job_id: str | None) -> dict[str, Any]:
        target = self._get_record(job_id) if job_id else None
        if target is None and job_id is None:
            active: list[dict[str, Any]] = []
            for item_key, item in self.job_store.items():
                if (
                    isinstance(item_key, str)
                    and item_key.startswith("job:")
                    and isinstance(item, dict)
                    and item.get("status") in ACTIVE_STATUSES
                ):
                    active.append(item)
            running = [item for item in active if item.get("status") == "running"]
            candidates = running or [item for item in active if item.get("status") == "queued"]
            if candidates:
                target = min(candidates, key=lambda item: item.get("created_at", ""))

        if not target or target.get("status") not in ACTIVE_STATUSES:
            return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}

        call_id = self.job_store.get(f"call:{target['id']}") or target.get("call_id")
        if not call_id:
            raise JobStateError("Job submission is still in progress; retry shortly")

        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=False)
        except Exception as exc:
            raise ModalOperationError("Could not cancel the Modal job; retry shortly") from exc

        # A durable tombstone closes the race where the remote worker is still
        # inside a progress callback after FunctionCall.cancel() returns. The
        # worker checks this key before every state transition and completion.
        self.job_store.put(
            self._cancel_key(target["id"]),
            {"requested_at": self._utc_now()},
        )

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

    def upload_file(self, local_path: Path, remote_path: str) -> MediaRef:
        return self.media_storage.upload_local(local_path, remote_path)

    def prepare_asset_upload(
        self,
        *,
        filename: str,
        content_type: str | None,
        size: int,
        kind: str,
        suffix: str,
    ) -> dict[str, Any]:
        asset_id = uuid.uuid4().hex
        key = f"inputs/{asset_id}{suffix}"
        resolved_content_type = media_content_type(filename, content_type)
        plan = self.media_storage.prepare_upload(
            key=key,
            content_type=resolved_content_type,
            size=size,
        )
        intent = {
            "asset_id": asset_id,
            "store_id": plan["store_id"],
            "key": key,
            "filename": filename,
            "content_type": resolved_content_type,
            "size": size,
            "kind": kind,
            "mode": plan["mode"],
            "upload_id": plan.get("upload_id"),
            "created_at": time.time(),
        }
        self.job_store.put(self._upload_key(asset_id), intent)
        result = {
            "asset_id": asset_id,
            "filename": filename,
            "kind": kind,
            "size": size,
            "backend": self.media_storage.store(plan["store_id"]).backend,
            **{name: value for name, value in plan.items() if name != "store_id"},
        }
        if plan["mode"] == "proxy":
            result["method"] = "PUT"
            result["url"] = f"/api/assets/{asset_id}/content"
        return result


    def prepare_asset_fallback(self, asset_id: str) -> dict[str, Any]:
        intent = self.get_upload_intent(asset_id)
        if not intent:
            raise FileNotFoundError(asset_id)
        current = MediaRef(intent["store_id"], intent["key"])
        try:
            self.media_storage.abort_upload(current, upload_id=intent.get("upload_id"))
        except Exception:
            pass
        plan = self.media_storage.prepare_fallback_upload(
            key=intent["key"],
            content_type=intent["content_type"],
            size=int(intent["size"]),
        )
        intent.update(
            store_id=plan["store_id"],
            mode=plan["mode"],
            upload_id=plan.get("upload_id"),
        )
        self.job_store.put(self._upload_key(asset_id), intent)
        result = {
            "asset_id": asset_id,
            "filename": intent["filename"],
            "kind": intent["kind"],
            "size": int(intent["size"]),
            "backend": self.media_storage.store(plan["store_id"]).backend,
            **{name: value for name, value in plan.items() if name != "store_id"},
        }
        if plan["mode"] == "proxy":
            result["method"] = "PUT"
            result["url"] = f"/api/assets/{asset_id}/content"
        return result

    def get_upload_intent(self, asset_id: str) -> dict[str, Any] | None:
        return self.job_store.get(self._upload_key(asset_id))

    def upload_proxy_file(self, asset_id: str, local_path: Path) -> None:
        intent = self.get_upload_intent(asset_id)
        if not intent or intent.get("mode") != "proxy":
            raise FileNotFoundError(asset_id)
        actual = local_path.stat().st_size
        if actual != int(intent["size"]):
            raise MediaStoreError(
                f"Uploaded size mismatch: expected {intent['size']}, got {actual}"
            )
        ref = self.media_storage.upload_local(local_path, intent["key"])
        intent["store_id"] = ref.store_id
        self.job_store.put(self._upload_key(asset_id), intent)

    def complete_asset_upload(
        self,
        asset_id: str,
        parts: list[dict[str, Any]] | None = None,
        client_upload_seconds: float | None = None,
    ) -> dict[str, Any]:
        intent = self.get_upload_intent(asset_id)
        if not intent:
            completed = self.job_store.get(self._asset_key(asset_id))
            if isinstance(completed, dict) and all(
                completed.get(name) is not None for name in ("kind", "filename", "size")
            ):
                return {
                    "id": asset_id,
                    "kind": completed["kind"],
                    "filename": completed["filename"],
                    "size": int(completed["size"]),
                }
            raise FileNotFoundError(asset_id)
        ref = MediaRef(intent["store_id"], intent["key"])
        actual = self.media_storage.complete_upload(
            ref,
            expected_size=int(intent["size"]),
            upload_id=intent.get("upload_id"),
            parts=parts,
        )
        self.media_storage.record_client_upload(ref.store_id, client_upload_seconds)
        try:
            self.register_asset(
                asset_id,
                intent["key"],
                kind=intent["kind"],
                filename=intent["filename"],
                size=actual,
                store_id=ref.store_id,
            )
        except Exception:
            # Keep both the durable object and upload intent. A transient Dict
            # failure can then retry registration without retransmitting bytes.
            raise
        self.job_store.pop(self._upload_key(asset_id), None)
        return {
            "id": asset_id,
            "kind": intent["kind"],
            "filename": intent["filename"],
            "size": actual,
        }

    def abort_asset_upload(self, asset_id: str) -> bool:
        intent = self.get_upload_intent(asset_id)
        if not intent:
            return False
        ref = MediaRef(intent["store_id"], intent["key"])
        self.media_storage.abort_upload(ref, upload_id=intent.get("upload_id"))
        try:
            self.media_storage.remove(ref)
        except Exception:
            pass
        self.job_store.pop(self._upload_key(asset_id), None)
        return True

    def _asset_ref(self, record: dict[str, Any]) -> MediaRef:
        return MediaRef.from_value(record, default_store_id=self.media_storage.primary_id)

    def _output_ref(self, filename: str) -> MediaRef:
        record = self.job_store.get(self._output_key(filename))
        if isinstance(record, dict):
            return self._asset_ref(record)
        # Compatibility for outputs produced before routed metadata existed.
        return MediaRef(self.media_storage.primary_id, f"outputs/{filename}")

    def register_output(self, filename: str, ref: MediaRef, *, size: int | None = None) -> None:
        record: dict[str, Any] = {**ref.as_dict(), "created_at": time.time()}
        if size is not None:
            record["size"] = int(size)
        self.job_store.put(self._output_key(filename), record)

    def copy_output_to_input(self, job_id: str) -> dict[str, Any]:
        """Expose a completed video output as a new input without a browser round-trip."""
        record = self._get_record(job_id)
        if not record or record.get("status") != "completed":
            raise FileNotFoundError(job_id)

        source = record.get("video_key")
        if not source and record.get("video_url"):
            source = f"outputs/{Path(record['video_url']).name}"
        if not isinstance(source, str) or not source:
            raise FileNotFoundError(job_id)
        filename = Path(source).name
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".mp4":
            raise ValueError("Job output is not a reusable MP4")

        source_ref = self._output_ref(filename)
        asset_id = uuid.uuid4().hex
        destination = f"inputs/{asset_id}.mp4"
        destination_ref, size = self.media_storage.copy(source_ref, destination)

        try:
            self.register_asset(
                asset_id,
                destination,
                kind="video",
                filename=filename,
                size=size,
                store_id=destination_ref.store_id,
            )
        except Exception:
            self.media_storage.remove(destination_ref)
            raise

        return {
            "id": asset_id,
            "kind": "video",
            "filename": filename,
            "size": size,
        }

    def register_asset(
        self,
        asset_id: str,
        remote_path: str,
        *,
        kind: str | None = None,
        filename: str | None = None,
        size: int | None = None,
        store_id: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "store_id": store_id or self.media_storage.primary_id,
            "key": remote_path,
            "uploaded_at": time.time(),
        }
        if kind is not None:
            record["kind"] = kind
        if filename is not None:
            record["filename"] = filename
        if size is not None:
            record["size"] = int(size)
        self.job_store.put(self._asset_key(asset_id), record)

    def remove_input(self, remote_path: str, *, store_id: str | None = None) -> None:
        self.media_storage.remove(MediaRef(store_id or self.media_storage.primary_id, remote_path))

    @staticmethod
    def _request_asset_ids(request: dict[str, Any]) -> set[str]:
        asset_ids = {
            item.get("asset_id")
            for item in request.get("conditions", [])
            if isinstance(item, dict) and item.get("asset_id")
        }
        if request.get("audio_asset_id"):
            asset_ids.add(request["audio_asset_id"])
        return asset_ids

    def cleanup_assets(self, max_age_seconds: int = 24 * 60 * 60) -> int:
        """Remove stale uploaded inputs that no active job can still consume."""
        now = time.time()
        active_assets: set[str] = set()
        asset_records: list[tuple[str, dict[str, Any]]] = []

        for key, value in self.job_store.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            if key.startswith("job:") and value.get("status") in ACTIVE_STATUSES:
                active_assets.update(self._request_asset_ids(value.get("request") or {}))
            elif key.startswith("asset:"):
                asset_records.append((key, value))

        removed = 0
        for key, value in asset_records:
            asset_id = key.removeprefix("asset:")
            uploaded_at = float(value.get("uploaded_at") or 0)
            if asset_id in active_assets or now - uploaded_at < max_age_seconds:
                continue
            remote_path = value.get("key") or value.get("remote_path")
            if isinstance(remote_path, str) and remote_path:
                try:
                    self.media_storage.remove(self._asset_ref(value))
                except Exception:
                    pass
            self.job_store.pop(key, None)
            removed += 1

        for key, value in list(self.job_store.items()):
            if not isinstance(key, str) or not key.startswith("upload:") or not isinstance(value, dict):
                continue
            created_at = float(value.get("created_at") or 0)
            if now - created_at < max_age_seconds:
                continue
            media_key = value.get("key")
            if isinstance(media_key, str) and media_key:
                ref = MediaRef(value.get("store_id") or self.media_storage.primary_id, media_key)
                try:
                    self.media_storage.abort_upload(ref, upload_id=value.get("upload_id"))
                except Exception:
                    pass
                try:
                    self.media_storage.remove(ref)
                except Exception:
                    pass
            self.job_store.pop(key, None)
        return removed

    def download_output(self, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError("Invalid output filename")

        self.output_cache.mkdir(parents=True, exist_ok=True)
        target = self.output_cache / filename
        with self._output_lock:
            return self.media_storage.download_to(self._output_ref(filename), target)

    def output_delivery_url(self, filename: str, *, method: str = "GET") -> str | None:
        if Path(filename).name != filename:
            raise ValueError("Invalid output filename")
        return self.media_storage.delivery_url(self._output_ref(filename), method=method)

    def remove_output(self, filename: str) -> None:
        try:
            self.media_storage.remove(self._output_ref(filename))
        except FileNotFoundError:
            pass
        self.job_store.pop(self._output_key(filename), None)
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
