from pathlib import Path

from ltx25.media_storage import MediaRef, MediaStorage


class FakeStore:
    backend = "s3"
    direct_upload = True
    direct_download = True

    def __init__(self, name, *, fail_prepare=False, fail_upload=False, upload_error=None):
        self.name = name
        self.fail_prepare = fail_prepare
        self.fail_upload = fail_upload
        self.upload_error = upload_error
        self.objects = {}
        self.aborted = []
        self.client_uploads = []

    def prepare_upload(self, *, key, content_type, size):
        if self.fail_prepare:
            raise OSError(f"{self.name} unavailable")
        return {"mode": "single", "method": "PUT", "url": f"https://{self.name}/{key}", "headers": {}}

    def complete_upload(self, *, key, expected_size, upload_id=None, parts=None):
        return expected_size

    def abort_upload(self, *, key, upload_id):
        self.aborted.append((key, upload_id))

    def upload_local(self, local_path: Path, key: str):
        if self.upload_error is not None:
            raise self.upload_error
        if self.fail_upload:
            raise OSError(f"{self.name} unavailable")
        self.objects[key] = local_path.read_bytes()

    def copy(self, source_key, destination_key):
        self.objects[destination_key] = self.objects[source_key]
        return len(self.objects[destination_key])

    def remove(self, key):
        self.objects.pop(key, None)

    def stat_size(self, key):
        return len(self.objects[key])

    def download_to(self, key, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[key])
        return target

    def delivery_url(self, key, *, method="GET"):
        return f"https://{self.name}/{key}?method={method}"

    def metrics(self):
        return {"upload": {"count": len(self.objects)}}

    def record_client_upload(self, seconds):
        self.client_uploads.append(seconds)


def test_prepare_uses_primary_and_returns_stable_store_id():
    primary = FakeStore("primary")
    fallback = FakeStore("fallback")
    storage = MediaStorage(stores={"r2": primary, "minio": fallback}, primary_id="r2", fallback_id="minio")
    plan = storage.prepare_upload(key="inputs/a.png", content_type="image/png", size=1)
    assert plan["store_id"] == "r2"
    assert plan["url"].startswith("https://primary/")


def test_prepare_falls_back_when_primary_prepare_really_fails():
    primary = FakeStore("primary", fail_prepare=True)
    fallback = FakeStore("fallback")
    storage = MediaStorage(stores={"r2": primary, "minio": fallback}, primary_id="r2", fallback_id="minio")
    plan = storage.prepare_upload(key="inputs/a.png", content_type="image/png", size=1)
    assert plan["store_id"] == "minio"
    assert storage.metrics()["routing"]["failover_count"] == 1


def test_upload_local_falls_back_and_returns_fallback_ref(tmp_path):
    primary = FakeStore("primary", fail_upload=True)
    fallback = FakeStore("fallback")
    storage = MediaStorage(stores={"r2": primary, "minio": fallback}, primary_id="r2", fallback_id="minio")
    source = tmp_path / "out.mp4"
    source.write_bytes(b"video")
    ref = storage.upload_local(source, "outputs/out.mp4")
    assert ref == MediaRef("minio", "outputs/out.mp4")
    assert fallback.objects["outputs/out.mp4"] == b"video"


def test_read_routes_only_to_persisted_store(tmp_path):
    primary = FakeStore("primary")
    fallback = FakeStore("fallback")
    fallback.objects["inputs/a.png"] = b"fallback"
    storage = MediaStorage(stores={"r2": primary, "minio": fallback}, primary_id="r2", fallback_id="minio")
    target = tmp_path / "a.png"
    storage.download_to(MediaRef("minio", "inputs/a.png"), target)
    assert target.read_bytes() == b"fallback"


def test_local_validation_error_does_not_fail_over(tmp_path):
    primary = FakeStore("primary", upload_error=ValueError("bad local state"))
    fallback = FakeStore("fallback")
    storage = MediaStorage(stores={"r2": primary, "minio": fallback}, primary_id="r2", fallback_id="minio")
    source = tmp_path / "out.mp4"
    source.write_bytes(b"video")
    import pytest
    with pytest.raises(ValueError, match="bad local state"):
        storage.upload_local(source, "outputs/out.mp4")
    assert not fallback.objects
    assert storage.metrics()["routing"]["failover_count"] == 0
