from copy import deepcopy
from types import SimpleNamespace
from threading import Event
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.modal_gateway import build_gateway


class Store:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return deepcopy(self.data.get(key))

    def put(self, key, value):
        self.data[key] = deepcopy(value)

    def items(self):
        return list(deepcopy(self.data).items())

    def pop(self, key, default=None):
        return self.data.pop(key, default)


@pytest.fixture
def gateway(tmp_path):
    store = Store()
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    worker = SimpleNamespace(generate=SimpleNamespace(spawn=lambda *_: SimpleNamespace(object_id="fc-test")))
    app = build_gateway(worker_cls=lambda: worker, job_store=store, state_volume=volume,
                        state_root=str(tmp_path), model_id="test")
    return TestClient(app), store, worker, volume, tmp_path


def test_fast_worker_state_is_not_overwritten(gateway):
    client, store, worker, _, _ = gateway

    def spawn(job_id, payload):
        record = store.get(f"job:{job_id}")
        record.update(status="completed", progress=1, video_url=f"/outputs/{job_id}.mp4")
        store.put(f"job:{job_id}", record)
        return SimpleNamespace(object_id="fc-test")

    worker.generate.spawn = spawn
    response = client.post("/api/jobs", json={"prompt": "A bird"})
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "completed"
    assert "call_id" not in job
    assert store.get(f"call:{job['id']}") == "fc-test"


def test_completed_job_cannot_be_interrupted(gateway):
    client, store, _, _, _ = gateway
    job = client.post("/api/jobs", json={"prompt": "A bird"}).json()
    record = store.get(f"job:{job['id']}")
    record["status"] = "completed"
    store.put(f"job:{job['id']}", record)
    assert client.post("/api/interrupt", json={"job_id": job["id"]}).json()["interrupted"] is False
    assert store.get(f"job:{job['id']}")["status"] == "completed"


def test_cancel_failure_does_not_claim_success(gateway, monkeypatch):
    client, store, _, _, _ = gateway
    job = client.post("/api/jobs", json={"prompt": "A bird"}).json()

    def fail(**kwargs):
        raise RuntimeError("connection lost")

    monkeypatch.setattr("modal.FunctionCall.from_id", lambda _: SimpleNamespace(cancel=fail))
    assert client.post("/api/interrupt", json={"job_id": job["id"]}).status_code == 502
    assert store.get(f"job:{job['id']}")["status"] == "queued"


def test_poll_transport_failure_does_not_fail_job(gateway, monkeypatch):
    import modal

    client, store, _, _, _ = gateway
    job = client.post("/api/jobs", json={"prompt": "A bird"}).json()

    def fail(**kwargs):
        raise modal.exception.ConnectionError("temporary outage")

    monkeypatch.setattr("modal.FunctionCall.from_id", lambda _: SimpleNamespace(get=fail))
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"
    assert store.get(f"job:{job['id']}")["status"] == "queued"


def test_output_range_and_head_do_not_reload_existing_file(gateway):
    client, _, _, volume, root = gateway
    path = root / "outputs" / "test.mp4"
    path.write_bytes(b"0123456789")
    reloads = []
    volume.reload = lambda: reloads.append(True)
    response = client.get("/outputs/test.mp4", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert client.head("/outputs/test.mp4").headers["content-length"] == "10"
    assert not reloads


def test_worker_timeout_is_terminal_not_an_unfinished_poll(gateway, monkeypatch):
    import modal

    client, _, _, _, _ = gateway
    job = client.post("/api/jobs", json={"prompt": "A bird"}).json()

    def fail(**kwargs):
        raise modal.exception.FunctionTimeoutError("GPU execution expired")

    monkeypatch.setattr("modal.FunctionCall.from_id", lambda _: SimpleNamespace(get=fail))
    result = client.get(f"/api/jobs/{job['id']}").json()
    assert result["status"] == "failed"
    assert "FunctionTimeoutError" in result["error"]


def test_slow_upload_keeps_health_responsive_and_blocks_reload(gateway):
    client, _, _, volume, root = gateway
    entered, release = Event(), Event()
    reloads = []

    def commit():
        entered.set()
        assert release.wait(5)

    volume.commit = commit
    volume.reload = lambda: reloads.append(True)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app), base_url="http://test") as http:
            # GIF is accepted without an external ffprobe dependency.
            upload = asyncio.create_task(http.post("/api/assets", files={"file": ("test.gif", b"GIF89a", "image/gif")}))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                listing = asyncio.create_task(http.get("/api/loras"))
                assert (await asyncio.wait_for(http.get("/api/health"), 1)).status_code == 200
                assert not reloads
            finally:
                release.set()
            assert (await upload).status_code == 201
            assert (await listing).status_code == 200
            assert reloads == [True]

    asyncio.run(run())
