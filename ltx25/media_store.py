"""Media data plane for inputs and generated outputs.

The serving layer deals only in stable object keys under inputs/ and outputs/.
This module decides whether those keys live in a Modal Volume or an
S3-compatible object storage such as MinIO or another self-hosted provider.

Control-plane state remains in Modal Dict; expiring presigned URLs are never
persisted in job records.
"""
from __future__ import annotations

import math
import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Any


class MediaStoreError(RuntimeError):
    pass


class MediaStore:
    backend = "base"
    direct_upload = False
    direct_download = False

    def __init__(self) -> None:
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "upload": {
                "count": 0,
                "bytes": 0,
                "seconds": 0.0,
                "client_count": 0,
                "client_seconds": 0.0,
            },
            "download": {
                "count": 0,
                "bytes": 0,
                "seconds": 0.0,
                "first_byte_seconds": 0.0,
                "cache_hits": 0,
            },
            "copy": {"count": 0, "seconds": 0.0},
        }

    def _record(
        self,
        kind: str,
        *,
        elapsed: float = 0.0,
        byte_count: int = 0,
        first_byte_seconds: float | None = None,
        cache_hit: bool = False,
    ) -> None:
        with self._metrics_lock:
            bucket = self._metrics[kind]
            if cache_hit:
                bucket["cache_hits"] += 1
                return
            bucket["count"] += 1
            bucket["seconds"] += max(0.0, elapsed)
            if "bytes" in bucket:
                bucket["bytes"] += max(0, byte_count)
            if first_byte_seconds is not None:
                bucket["first_byte_seconds"] = max(0.0, first_byte_seconds)

    def metrics(self) -> dict[str, dict[str, int | float]]:
        with self._metrics_lock:
            return {
                kind: {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in bucket.items()
                }
                for kind, bucket in self._metrics.items()
            }

    def record_client_upload(self, seconds: float | None) -> None:
        if seconds is None:
            return
        with self._metrics_lock:
            self._metrics["upload"]["client_count"] += 1
            self._metrics["upload"]["client_seconds"] += max(0.0, float(seconds))

    def prepare_upload(
        self,
        *,
        key: str,
        content_type: str,
        size: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def complete_upload(
        self,
        *,
        key: str,
        expected_size: int,
        upload_id: str | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> int:
        raise NotImplementedError

    def abort_upload(self, *, key: str, upload_id: str | None) -> None:
        return None

    def upload_local(self, local_path: Path, key: str) -> None:
        raise NotImplementedError

    def copy(self, source_key: str, destination_key: str) -> int:
        raise NotImplementedError

    def remove(self, key: str) -> None:
        raise NotImplementedError

    def download_to(self, key: str, target: Path) -> Path:
        raise NotImplementedError

    def delivery_url(self, key: str, *, method: str = "GET") -> str | None:
        return None

    def stat_size(self, key: str) -> int:
        raise NotImplementedError


class VolumeMediaStore(MediaStore):
    backend = "volume"

    def __init__(self, volume: Any) -> None:
        super().__init__()
        self.volume = volume

    def prepare_upload(self, *, key: str, content_type: str, size: int) -> dict[str, Any]:
        return {"mode": "proxy", "headers": {"Content-Type": content_type}}

    def complete_upload(
        self,
        *,
        key: str,
        expected_size: int,
        upload_id: str | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> int:
        actual = self.stat_size(key)
        if actual != expected_size:
            raise MediaStoreError(f"Uploaded size mismatch: expected {expected_size}, got {actual}")
        return actual

    def upload_local(self, local_path: Path, key: str) -> None:
        started = time.monotonic()
        byte_count = local_path.stat().st_size
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(str(local_path), key)
        self._record("upload", elapsed=time.monotonic() - started, byte_count=byte_count)

    def copy(self, source_key: str, destination_key: str) -> int:
        started = time.monotonic()
        self.volume.copy_files([source_key], destination_key)
        self._record("copy", elapsed=time.monotonic() - started)
        return self.stat_size(destination_key)

    def remove(self, key: str) -> None:
        try:
            self.volume.remove_file(key)
        except FileNotFoundError:
            pass

    def stat_size(self, key: str) -> int:
        try:
            entries = list(self.volume.iterdir(key, recursive=False))
        except FileNotFoundError as exc:
            raise FileNotFoundError(key) from exc
        for entry in entries:
            if str(getattr(entry, "path", "")).rstrip("/") == key.rstrip("/"):
                return int(getattr(entry, "size", 0) or 0)
        if len(entries) == 1:
            return int(getattr(entries[0], "size", 0) or 0)
        raise FileNotFoundError(key)

    def download_to(self, key: str, target: Path) -> Path:
        if target.is_file():
            self._record("download", cache_hit=True)
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        started = time.monotonic()
        first_byte_seconds: float | None = None
        byte_count = 0
        try:
            with temporary.open("wb") as output:
                for chunk in self.volume.read_file(key):
                    if first_byte_seconds is None:
                        first_byte_seconds = time.monotonic() - started
                    byte_count += len(chunk)
                    output.write(chunk)
            temporary.replace(target)
            self._record(
                "download",
                elapsed=time.monotonic() - started,
                byte_count=byte_count,
                first_byte_seconds=first_byte_seconds,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target


class S3MediaStore(MediaStore):
    """Generic S3-compatible media store."""

    backend = "s3"
    direct_upload = True
    direct_download = True

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "auto",
        presign_seconds: int = 3600,
        multipart_threshold: int = 16 * 1024 * 1024,
        part_size: int = 16 * 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        if not bucket or not endpoint_url or not access_key_id or not secret_access_key:
            raise ValueError("S3 media backend requires bucket, endpoint and access credentials")
        if part_size < 5 * 1024 * 1024:
            raise ValueError("S3 multipart part size must be at least 5 MiB")
        self.bucket = bucket
        self.endpoint_url = endpoint_url.rstrip("/")
        self.presign_seconds = max(1, min(int(presign_seconds), 604800))
        self.multipart_threshold = max(5 * 1024 * 1024, int(multipart_threshold))
        self.part_size = int(part_size)
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required when LTX25_MEDIA_BACKEND uses S3-compatible storage; install requirements-local.txt"
                ) from exc
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    max_pool_connections=16,
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    connect_timeout=5,
                    read_timeout=60,
                ),
            )
        self.client = client

    def _transfer_shape(self, size: int) -> tuple[int, int]:
        """Choose multipart part size/concurrency from object size.

        Small media stays cheap; medium/large video gets enough independent PUTs
        to fill a typical WAN path without creating hundreds of S3 requests.
        """
        mib = 1024 * 1024
        if size < 64 * mib:
            return max(5 * mib, min(self.part_size, 8 * mib)), 2
        if size < 256 * mib:
            return max(8 * mib, min(max(self.part_size, 16 * mib), 32 * mib)), 4
        return max(self.part_size, 32 * mib), 6

    def prepare_upload(self, *, key: str, content_type: str, size: int) -> dict[str, Any]:
        if size < self.multipart_threshold:
            url = self.client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
                ExpiresIn=self.presign_seconds,
            )
            return {
                "mode": "single",
                "method": "PUT",
                "url": url,
                "headers": {"Content-Type": content_type},
            }

        part_size, concurrency = self._transfer_shape(size)
        created = self.client.create_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            ContentType=content_type,
        )
        upload_id = created["UploadId"]
        part_count = math.ceil(size / part_size)
        if part_count > 10_000:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
            raise MediaStoreError("Upload would exceed S3's 10,000-part limit")
        parts = [
            {
                "part_number": number,
                "url": self.client.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": self.bucket,
                        "Key": key,
                        "UploadId": upload_id,
                        "PartNumber": number,
                    },
                    ExpiresIn=self.presign_seconds,
                ),
            }
            for number in range(1, part_count + 1)
        ]
        return {
            "mode": "multipart",
            "method": "PUT",
            "upload_id": upload_id,
            "part_size": part_size,
            "concurrency": min(concurrency, part_count),
            "parts": parts,
            "headers": {},
        }

    def complete_upload(
        self,
        *,
        key: str,
        expected_size: int,
        upload_id: str | None = None,
        parts: list[dict[str, Any]] | None = None,
    ) -> int:
        started = time.monotonic()
        if upload_id:
            # Finalize is deliberately idempotent. A multipart completion can
            # succeed in object storage while its HTTP response is lost. On retry the upload
            # id no longer exists, but the final object is already durable.
            try:
                existing = int(
                    self.client.head_object(Bucket=self.bucket, Key=key).get("ContentLength", 0)
                )
            except Exception:
                existing = None
            if existing != expected_size:
                if not parts:
                    raise MediaStoreError("Multipart completion requires uploaded part ETags")
                normalized = [
                    {"PartNumber": int(part["part_number"]), "ETag": str(part["etag"])}
                    for part in sorted(parts, key=lambda item: int(item["part_number"]))
                ]
                self.client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": normalized},
                )

        head = self.client.head_object(Bucket=self.bucket, Key=key)
        actual = int(head.get("ContentLength", 0))
        if actual != expected_size:
            self.remove(key)
            raise MediaStoreError(f"Uploaded size mismatch: expected {expected_size}, got {actual}")
        self._record("upload", elapsed=time.monotonic() - started, byte_count=actual)
        return actual

    def abort_upload(self, *, key: str, upload_id: str | None) -> None:
        if not upload_id:
            return
        try:
            self.client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
            )
        except Exception:
            pass

    def upload_local(self, local_path: Path, key: str) -> None:
        started = time.monotonic()
        byte_count = local_path.stat().st_size
        part_size, concurrency = self._transfer_shape(byte_count)
        try:
            from boto3.s3.transfer import TransferConfig

            transfer_config = TransferConfig(
                multipart_threshold=self.multipart_threshold,
                multipart_chunksize=part_size,
                max_concurrency=concurrency,
                use_threads=True,
            )
        except ImportError:
            transfer_config = None
        with local_path.open("rb") as source:
            kwargs = {"ExtraArgs": {"ContentType": media_content_type(key)}}
            if transfer_config is not None:
                kwargs["Config"] = transfer_config
            self.client.upload_fileobj(source, self.bucket, key, **kwargs)
        self._record("upload", elapsed=time.monotonic() - started, byte_count=byte_count)

    def copy(self, source_key: str, destination_key: str) -> int:
        started = time.monotonic()
        self.client.copy_object(
            Bucket=self.bucket,
            Key=destination_key,
            CopySource={"Bucket": self.bucket, "Key": source_key},
        )
        self._record("copy", elapsed=time.monotonic() - started)
        return self.stat_size(destination_key)

    def remove(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def stat_size(self, key: str) -> int:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = None
            code = ""
            if isinstance(response, dict):
                status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
                code = str((response.get("Error") or {}).get("Code") or "")
            if status == 404 or code in {"404", "NoSuchKey", "NotFound", "NoSuchObject"}:
                raise FileNotFoundError(key) from exc
            raise
        return int(head.get("ContentLength", 0))

    def download_to(self, key: str, target: Path) -> Path:
        if target.is_file():
            self._record("download", cache_hit=True)
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        started = time.monotonic()
        try:
            with temporary.open("wb") as output:
                self.client.download_fileobj(self.bucket, key, output)
            byte_count = temporary.stat().st_size
            temporary.replace(target)
            self._record("download", elapsed=time.monotonic() - started, byte_count=byte_count)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def delivery_url(self, key: str, *, method: str = "GET") -> str | None:
        operation = "head_object" if method.upper() == "HEAD" else "get_object"
        return self.client.generate_presigned_url(
            operation,
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.presign_seconds,
        )


def media_content_type(filename: str, supplied: str | None = None) -> str:
    if supplied and supplied != "application/octet-stream":
        return supplied
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def create_media_store(volume: Any, env: dict[str, str] | None = None) -> MediaStore:
    values = os.environ if env is None else env
    backend = values.get("LTX25_MEDIA_BACKEND", "volume").strip().lower()
    if backend == "volume":
        return VolumeMediaStore(volume)
    if backend != "s3":
        raise ValueError("LTX25_MEDIA_BACKEND must be 'volume' or 's3'")

    endpoint = values.get("LTX25_S3_ENDPOINT_URL") or ""
    bucket = values.get("LTX25_S3_BUCKET") or ""
    access_key = values.get("AWS_ACCESS_KEY_ID") or ""
    secret_key = values.get("AWS_SECRET_ACCESS_KEY") or ""
    region = values.get("LTX25_S3_REGION", "auto")
    presign = int(values.get("LTX25_MEDIA_PRESIGN_SECONDS", "3600"))
    threshold_mb = int(values.get("LTX25_MEDIA_MULTIPART_THRESHOLD_MB", "16"))
    part_mb = int(values.get("LTX25_MEDIA_PART_SIZE_MB", "16"))

    store = S3MediaStore(
        bucket=bucket,
        endpoint_url=endpoint,
        access_key_id=access_key,
        secret_access_key=secret_key,
        region=region,
        presign_seconds=presign,
        multipart_threshold=threshold_mb * 1024 * 1024,
        part_size=part_mb * 1024 * 1024,
    )
    store.backend = backend
    return store
