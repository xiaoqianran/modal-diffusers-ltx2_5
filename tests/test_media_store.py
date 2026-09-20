from __future__ import annotations

from pathlib import Path

import sys
from types import SimpleNamespace

from ltx25.media_store import S3MediaStore, create_media_store


MIB = 1024 * 1024


class FakeS3:
    def __init__(self):
        self.calls = []
        self.sizes = {}
        self.objects = {}
        self.complete_size = None

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        self.calls.append(("presign", operation, dict(Params), ExpiresIn))
        part = Params.get("PartNumber")
        suffix = f"?part={part}" if part else ""
        return f"https://s3.test/{operation}/{Params['Key']}{suffix}"

    def create_multipart_upload(self, **kwargs):
        self.calls.append(("create_multipart_upload", kwargs))
        return {"UploadId": "upload-1"}

    def complete_multipart_upload(self, **kwargs):
        self.calls.append(("complete_multipart_upload", kwargs))
        if self.complete_size is not None:
            self.sizes[kwargs["Key"]] = self.complete_size
        return {}

    def abort_multipart_upload(self, **kwargs):
        self.calls.append(("abort_multipart_upload", kwargs))

    def head_object(self, Bucket, Key):
        if Key not in self.sizes:
            raise RuntimeError("missing")
        return {"ContentLength": self.sizes[Key]}

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete_object", Bucket, Key))
        self.sizes.pop(Key, None)
        self.objects.pop(Key, None)

    def copy_object(self, Bucket, Key, CopySource):
        source = CopySource["Key"]
        self.calls.append(("copy_object", source, Key))
        self.sizes[Key] = self.sizes[source]
        if source in self.objects:
            self.objects[Key] = self.objects[source]

    def upload_fileobj(self, source, bucket, key, ExtraArgs=None, Config=None):
        data = source.read()
        self.calls.append(("upload_fileobj", bucket, key, ExtraArgs or {}, Config))
        self.objects[key] = data
        self.sizes[key] = len(data)

    def download_fileobj(self, bucket, key, output):
        output.write(self.objects[key])


def make_store(client: FakeS3, **overrides):
    return S3MediaStore(
        bucket="media",
        endpoint_url="https://s3.example.test",
        access_key_id="key",
        secret_access_key="secret",
        client=client,
        presign_seconds=900,
        multipart_threshold=overrides.get("multipart_threshold", 16 * MIB),
        part_size=overrides.get("part_size", 16 * MIB),
    )


def test_s3_single_put_is_presigned_with_content_type():
    client = FakeS3()
    store = make_store(client)

    plan = store.prepare_upload(
        key="inputs/a.png",
        content_type="image/png",
        size=1024,
    )

    assert plan["mode"] == "single"
    assert plan["headers"] == {"Content-Type": "image/png"}
    assert plan["url"].startswith("https://s3.test/put_object/")
    _, operation, params, expires = client.calls[0]
    assert operation == "put_object"
    assert params["ContentType"] == "image/png"
    assert expires == 900


def test_s3_large_upload_uses_parallelizable_multipart_plan():
    client = FakeS3()
    store = make_store(client, multipart_threshold=5 * MIB, part_size=5 * MIB)

    plan = store.prepare_upload(
        key="inputs/video.mp4",
        content_type="video/mp4",
        size=12 * MIB,
    )

    assert plan["mode"] == "multipart"
    assert plan["upload_id"] == "upload-1"
    assert plan["part_size"] == 5 * MIB
    assert [part["part_number"] for part in plan["parts"]] == [1, 2, 3]
    assert all("upload_part" in part["url"] for part in plan["parts"])


def test_s3_upload_plan_adapts_part_size_and_concurrency_to_object_size():
    client = FakeS3()
    store = make_store(client)

    medium = store.prepare_upload(
        key="inputs/medium.mp4",
        content_type="video/mp4",
        size=20 * MIB,
    )
    large = store.prepare_upload(
        key="inputs/large.mp4",
        content_type="video/mp4",
        size=100 * MIB,
    )
    huge = store.prepare_upload(
        key="inputs/huge.mp4",
        content_type="video/mp4",
        size=300 * MIB,
    )

    assert (medium["part_size"], medium["concurrency"]) == (8 * MIB, 2)
    assert (large["part_size"], large["concurrency"]) == (16 * MIB, 4)
    assert (huge["part_size"], huge["concurrency"]) == (32 * MIB, 6)


def test_s3_multipart_completion_forwards_etags_and_verifies_size():
    client = FakeS3()
    store = make_store(client, multipart_threshold=5 * MIB, part_size=5 * MIB)
    key = "inputs/video.mp4"
    client.complete_size = 12 * MIB

    actual = store.complete_upload(
        key=key,
        expected_size=12 * MIB,
        upload_id="upload-1",
        parts=[
            {"part_number": 2, "etag": '"b"'},
            {"part_number": 1, "etag": '"a"'},
            {"part_number": 3, "etag": '"c"'},
        ],
    )

    assert actual == 12 * MIB
    complete = next(call for call in client.calls if call[0] == "complete_multipart_upload")
    forwarded = complete[1]["MultipartUpload"]["Parts"]
    assert forwarded == [
        {"PartNumber": 1, "ETag": '"a"'},
        {"PartNumber": 2, "ETag": '"b"'},
        {"PartNumber": 3, "ETag": '"c"'},
    ]


def test_s3_multipart_completion_is_idempotent_after_lost_response():
    client = FakeS3()
    store = make_store(client, multipart_threshold=5 * MIB, part_size=5 * MIB)
    key = "inputs/video.mp4"
    client.sizes[key] = 12 * MIB

    actual = store.complete_upload(
        key=key,
        expected_size=12 * MIB,
        upload_id="already-finished",
        parts=[{"part_number": 1, "etag": '"a"'}],
    )

    assert actual == 12 * MIB
    assert not any(call[0] == "complete_multipart_upload" for call in client.calls)


def test_s3_copy_is_server_side_and_delivery_urls_are_method_specific():
    client = FakeS3()
    store = make_store(client)
    client.sizes["outputs/job.mp4"] = 123

    size = store.copy("outputs/job.mp4", "inputs/copy.mp4")
    get_url = store.delivery_url("outputs/job.mp4", method="GET")
    head_url = store.delivery_url("outputs/job.mp4", method="HEAD")

    assert size == 123
    assert ("copy_object", "outputs/job.mp4", "inputs/copy.mp4") in client.calls
    assert "/get_object/" in get_url
    assert "/head_object/" in head_url


def test_s3_local_upload_and_download_support_concat_cache(tmp_path: Path):
    client = FakeS3()
    store = make_store(client)
    source = tmp_path / "combined.mp4"
    source.write_bytes(b"video-bytes")

    store.upload_local(source, "outputs/combined.mp4")
    target = tmp_path / "cache" / "combined.mp4"
    store.download_to("outputs/combined.mp4", target)

    assert target.read_bytes() == b"video-bytes"
    upload = next(call for call in client.calls if call[0] == "upload_fileobj")
    assert upload[3]["ContentType"] == "video/mp4"
    assert store.metrics()["upload"]["bytes"] == len(b"video-bytes")
    assert store.metrics()["download"]["bytes"] == len(b"video-bytes")


def test_client_upload_time_is_tracked_separately():
    client = FakeS3()
    store = make_store(client)

    store.record_client_upload(1.25)

    metrics = store.metrics()["upload"]
    assert metrics["client_count"] == 1
    assert metrics["client_seconds"] == 1.25


def test_create_media_store_maps_s3_environment(monkeypatch):
    client = FakeS3()
    captured = {}

    def make_client(service, **kwargs):
        captured.update(service=service, **kwargs)
        return client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=make_client))
    store = create_media_store(
        object(),
        {
            "LTX25_MEDIA_BACKEND": "s3",
            "LTX25_S3_BUCKET": "media",
            "LTX25_S3_ENDPOINT_URL": "https://s3.example.test",
            "AWS_ACCESS_KEY_ID": "key",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "LTX25_MEDIA_PRESIGN_SECONDS": "600",
            "LTX25_MEDIA_MULTIPART_THRESHOLD_MB": "80",
            "LTX25_MEDIA_PART_SIZE_MB": "20",
        },
    )

    assert store.backend == "s3"
    assert store.bucket == "media"
    assert store.presign_seconds == 600
    assert store.multipart_threshold == 80 * MIB
    assert store.part_size == 20 * MIB
    assert captured["service"] == "s3"
    assert captured["region_name"] == "auto"
    assert captured["config"].signature_version == "s3v4"
    assert captured["config"].s3["addressing_style"] == "path"
