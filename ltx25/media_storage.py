"""Routed media storage over one or more physical MediaStore backends.

`media_store.py` owns physical backend mechanics. This module owns stable store
identity, primary/fallback policy, and durable MediaRef routing. Provider names
are configuration only; callers route by opaque store ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import threading

from .media_store import MediaStore, MediaStoreError, S3MediaStore, VolumeMediaStore


@dataclass(frozen=True, slots=True)
class MediaRef:
    store_id: str
    key: str

    def as_dict(self) -> dict[str, str]:
        return {"store_id": self.store_id, "key": self.key}

    @classmethod
    def from_value(cls, value: Any, *, default_store_id: str | None = None) -> "MediaRef":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            store_id = value.get("store_id") or default_store_id
            key = value.get("key") or value.get("remote_path")
            if isinstance(store_id, str) and store_id and isinstance(key, str) and key:
                return cls(store_id=store_id, key=key)
        if isinstance(value, str) and value and default_store_id:
            return cls(store_id=default_store_id, key=value)
        raise ValueError("Invalid media reference")


def _should_failover(exc: Exception) -> bool:
    # Local/programming/validation failures must remain visible. Failover is for
    # backend availability failures, not for hiding invalid application state.
    return not isinstance(exc, (FileNotFoundError, ValueError, MediaStoreError))


class MediaStorage:
    """Routes media operations while keeping physical stores provider-neutral."""

    def __init__(
        self,
        *,
        stores: dict[str, MediaStore],
        primary_id: str,
        fallback_id: str | None = None,
    ) -> None:
        if primary_id not in stores:
            raise ValueError(f"Unknown primary media store: {primary_id}")
        if fallback_id is not None and fallback_id not in stores:
            raise ValueError(f"Unknown fallback media store: {fallback_id}")
        if fallback_id == primary_id:
            raise ValueError("Primary and fallback media stores must differ")
        self.stores = dict(stores)
        self.primary_id = primary_id
        self.fallback_id = fallback_id
        self._lock = threading.Lock()
        self._failovers = 0

    @property
    def backend(self) -> str:
        return "failover" if self.fallback_id else self.stores[self.primary_id].backend

    @property
    def direct_upload(self) -> bool:
        return self.stores[self.primary_id].direct_upload

    @property
    def direct_download(self) -> bool:
        return self.stores[self.primary_id].direct_download

    def store(self, store_id: str) -> MediaStore:
        try:
            return self.stores[store_id]
        except KeyError as exc:
            raise ValueError(f"Unknown media store id: {store_id}") from exc

    def _note_failover(self) -> None:
        with self._lock:
            self._failovers += 1

    def prepare_upload(
        self,
        *,
        key: str,
        content_type: str,
        size: int,
        store_id: str | None = None,
    ) -> dict[str, Any]:
        selected = store_id or self.primary_id
        try:
            plan = self.store(selected).prepare_upload(key=key, content_type=content_type, size=size)
        except Exception as exc:
            if store_id is not None or not self.fallback_id or not _should_failover(exc):
                raise
            selected = self.fallback_id
            self._note_failover()
            plan = self.store(selected).prepare_upload(key=key, content_type=content_type, size=size)
        return {**plan, "store_id": selected}

    def prepare_fallback_upload(
        self,
        *,
        key: str,
        content_type: str,
        size: int,
    ) -> dict[str, Any]:
        if not self.fallback_id:
            raise RuntimeError("No fallback media store is configured")
        self._note_failover()
        return self.prepare_upload(
            key=key,
            content_type=content_type,
            size=size,
            store_id=self.fallback_id,
        )

    def complete_upload(
        self,
        ref: MediaRef,
        *,
        expected_size: int,
        upload_id: str | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> int:
        return self.store(ref.store_id).complete_upload(
            key=ref.key,
            expected_size=expected_size,
            upload_id=upload_id,
            parts=parts,
        )

    def abort_upload(self, ref: MediaRef, *, upload_id: str | None) -> None:
        self.store(ref.store_id).abort_upload(key=ref.key, upload_id=upload_id)

    def upload_local(self, local_path: Path, key: str) -> MediaRef:
        primary = MediaRef(self.primary_id, key)
        try:
            self.store(primary.store_id).upload_local(local_path, key)
            return primary
        except Exception as exc:
            if not self.fallback_id or not _should_failover(exc):
                raise
        fallback = MediaRef(self.fallback_id, key)
        self._note_failover()
        self.store(fallback.store_id).upload_local(local_path, key)
        return fallback

    def copy(self, source: MediaRef, destination_key: str) -> tuple[MediaRef, int]:
        # Prefer a server-side copy inside the source store. Cross-store fallback
        # is intentionally not hidden here; callers can materialize and re-upload
        # if a future workflow needs it.
        size = self.store(source.store_id).copy(source.key, destination_key)
        return MediaRef(source.store_id, destination_key), size

    def remove(self, ref: MediaRef) -> None:
        self.store(ref.store_id).remove(ref.key)

    def stat_size(self, ref: MediaRef) -> int:
        return self.store(ref.store_id).stat_size(ref.key)

    def download_to(self, ref: MediaRef, target: Path) -> Path:
        return self.store(ref.store_id).download_to(ref.key, target)

    def delivery_url(self, ref: MediaRef, *, method: str = "GET") -> str | None:
        return self.store(ref.store_id).delivery_url(ref.key, method=method)

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            failovers = self._failovers
        per_store = {store_id: store.metrics() for store_id, store in self.stores.items()}
        aggregate: dict[str, dict[str, int | float]] = {}
        for kind in ("upload", "download", "copy"):
            keys = {key for metrics in per_store.values() for key in metrics.get(kind, {})}
            aggregate[kind] = {
                key: sum(float(metrics.get(kind, {}).get(key, 0)) for metrics in per_store.values())
                for key in keys
            }
            for key, value in list(aggregate[kind].items()):
                if all(isinstance(metrics.get(kind, {}).get(key, 0), int) for metrics in per_store.values()):
                    aggregate[kind][key] = int(value)
        return {
            **aggregate,
            "routing": {
                "primary_id": self.primary_id,
                "fallback_id": self.fallback_id,
                "failover_count": failovers,
            },
            "stores": per_store,
        }

    def record_client_upload(self, store_id: str, seconds: float | None) -> None:
        self.store(store_id).record_client_upload(seconds)


def _s3_store(values: dict[str, str] | os._Environ[str], prefix: str) -> S3MediaStore:
    def get(name: str, default: str = "") -> str:
        return values.get(f"{prefix}{name}", default) or default

    return S3MediaStore(
        bucket=get("S3_BUCKET"),
        endpoint_url=get("S3_ENDPOINT_URL"),
        access_key_id=get("S3_ACCESS_KEY_ID"),
        secret_access_key=get("S3_SECRET_ACCESS_KEY"),
        region=get("S3_REGION", "auto"),
        presign_seconds=int(values.get("LTX25_MEDIA_PRESIGN_SECONDS", "3600")),
        multipart_threshold=int(values.get("LTX25_MEDIA_MULTIPART_THRESHOLD_MB", "96")) * 1024 * 1024,
        part_size=int(values.get("LTX25_MEDIA_PART_SIZE_MB", "16")) * 1024 * 1024,
    )


def create_media_storage(volume: Any, env: dict[str, str] | None = None) -> MediaStorage:
    values = os.environ if env is None else env

    primary_id = values.get("LTX25_MEDIA_PRIMARY_ID", "").strip()
    if primary_id:
        primary_backend = values.get("LTX25_MEDIA_PRIMARY_BACKEND", "s3").strip().lower()
        if primary_backend == "volume":
            primary_store: MediaStore = VolumeMediaStore(volume)
        elif primary_backend == "s3":
            primary_store = _s3_store(values, "LTX25_MEDIA_PRIMARY_")
        else:
            raise ValueError("LTX25_MEDIA_PRIMARY_BACKEND must be 'volume' or 's3'")

        stores: dict[str, MediaStore] = {primary_id: primary_store}
        fallback_id = values.get("LTX25_MEDIA_FALLBACK_ID", "").strip() or None
        if fallback_id:
            fallback_backend = values.get("LTX25_MEDIA_FALLBACK_BACKEND", "s3").strip().lower()
            if fallback_backend == "volume":
                fallback_store: MediaStore = VolumeMediaStore(volume)
            elif fallback_backend == "s3":
                fallback_store = _s3_store(values, "LTX25_MEDIA_FALLBACK_")
            else:
                raise ValueError("LTX25_MEDIA_FALLBACK_BACKEND must be 'volume' or 's3'")
            stores[fallback_id] = fallback_store
        return MediaStorage(stores=stores, primary_id=primary_id, fallback_id=fallback_id)

    # Backward-compatible single-backend configuration.
    backend = values.get("LTX25_MEDIA_BACKEND", "volume").strip().lower()
    if backend == "volume":
        return MediaStorage(stores={"volume": VolumeMediaStore(volume)}, primary_id="volume")
    if backend != "s3":
        raise ValueError("LTX25_MEDIA_BACKEND must be 'volume' or 's3'")
    legacy = S3MediaStore(
        bucket=values.get("LTX25_S3_BUCKET", ""),
        endpoint_url=values.get("LTX25_S3_ENDPOINT_URL", ""),
        access_key_id=values.get("AWS_ACCESS_KEY_ID", ""),
        secret_access_key=values.get("AWS_SECRET_ACCESS_KEY", ""),
        region=values.get("LTX25_S3_REGION", "auto"),
        presign_seconds=int(values.get("LTX25_MEDIA_PRESIGN_SECONDS", "3600")),
        multipart_threshold=int(values.get("LTX25_MEDIA_MULTIPART_THRESHOLD_MB", "96")) * 1024 * 1024,
        part_size=int(values.get("LTX25_MEDIA_PART_SIZE_MB", "16")) * 1024 * 1024,
    )
    return MediaStorage(stores={"s3": legacy}, primary_id="s3")
