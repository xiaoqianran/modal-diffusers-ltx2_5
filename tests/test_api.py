import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import ltx25.api as api_module
from ltx25.modal_client import QueueFullError
from ltx25.schemas import ConcatRequest, GenerateRequest, STILL_IMAGE_MODES


class FakeControl:
    model_id = "test/ltx25"
    gpu_idle_seconds = 600
    warm_lease_seconds = 90
    keep_gpu_warm = False

    def __init__(self, root: Path):
        self.root = root
        self.jobs = {}
        self.uploads = {}
        self.upload_intents = {}
        self.loras = []
        self.idle_windows = []
        self.warmups = 0
        self.delivery_url = None

    def warm_status(self):
        return {"state": "disabled"}

    def transfer_metrics(self):
        return {
            "upload": {"count": 0, "bytes": 0, "seconds": 0.0},
            "download": {"count": 0, "bytes": 0, "seconds": 0.0, "first_byte_seconds": 0.0, "cache_hits": 0},
            "copy": {"count": 0, "seconds": 0.0},
        }

    def media_info(self):
        return {"backend": "volume", "direct_upload": False, "direct_download": False}

    def queue_stats(self):
        queued = sum(job["status"] == "queued" for job in self.jobs.values())
        running = sum(job["status"] == "running" for job in self.jobs.values())
        return {"queued": queued, "running": running, "active": queued + running, "capacity": 4}

    def maintain_keep_warm(self):
        return False

    def set_idle_window(self, seconds):
        self.idle_windows.append(seconds)

    def start_warmup(self):
        self.warmups += 1
        return True

    def enable_keep_warm(self):
        self.keep_gpu_warm = True
        self.set_idle_window(self.gpu_idle_seconds)
        return self.start_warmup()

    def disable_keep_warm(self):
        self.keep_gpu_warm = False
        self.set_idle_window(2)

    @staticmethod
    def create_session():
        return 123456

    def create_job(self, request: GenerateRequest):
        now = datetime.now(timezone.utc).isoformat()
        job_id = uuid.uuid4().hex
        record = {
            "id": job_id,
            "session_number": request.session_number or self.create_session(),
            "status": "completed",
            "progress": 1.0,
            "error": None,
            "video_url": None if request.mode in STILL_IMAGE_MODES else f"/outputs/{job_id}.mp4",
            "image_url": f"/outputs/t2i_{job_id}.png" if request.mode in STILL_IMAGE_MODES else None,
            "request": request.model_dump(mode="json"),
            "created_at": now,
            "updated_at": now,
            "generation_seconds": 0.1,
            "peak_vram_gb": 1.0,
        }
        self.jobs[job_id] = record
        return record

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def list_jobs(self, session_number=None, limit=50):
        jobs = list(self.jobs.values())
        if session_number is not None:
            jobs = [job for job in jobs if job["session_number"] == session_number]
        return jobs[:limit]

    def list_job_summaries(self, session_number=None, limit=50):
        return [
            {
                **job,
                "request": {
                    key: job["request"].get(key)
                    for key in ("mode", "prompt", "width", "height", "num_frames", "fps", "steps", "upscale")
                },
            }
            for job in self.list_jobs(session_number, limit)
        ]

    def delete_job(self, job_id):
        return self.jobs.pop(job_id, None) is not None

    def interrupt(self, job_id):
        return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}

    def upload_file(self, local_path: Path, remote_path: str):
        self.uploads[remote_path] = local_path.read_bytes()
        return SimpleNamespace(store_id="volume", key=remote_path)
        return SimpleNamespace(store_id="volume", key=remote_path)

    def prepare_asset_upload(self, *, filename, content_type, size, kind, suffix):
        asset_id = uuid.uuid4().hex
        intent = {
            "asset_id": asset_id,
            "key": f"inputs/{asset_id}{suffix}",
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "kind": kind,
            "mode": "proxy",
        }
        self.upload_intents[asset_id] = intent
        return {
            "asset_id": asset_id,
            "filename": filename,
            "kind": kind,
            "size": size,
            "backend": "volume",
            "mode": "proxy",
            "method": "PUT",
            "url": f"/api/assets/{asset_id}/content",
            "headers": {"Content-Type": content_type or "application/octet-stream"},
            "parts": [],
        }

    def get_upload_intent(self, asset_id):
        return self.upload_intents.get(asset_id)

    def upload_proxy_file(self, asset_id, local_path):
        intent = self.upload_intents[asset_id]
        self.uploads[intent["key"]] = local_path.read_bytes()

    def complete_asset_upload(self, asset_id, parts=None, client_upload_seconds=None):
        intent = self.upload_intents.pop(asset_id)
        return {
            "id": asset_id,
            "kind": intent["kind"],
            "filename": intent["filename"],
            "size": intent["size"],
        }

    def abort_asset_upload(self, asset_id):
        intent = self.upload_intents.pop(asset_id, None)
        if not intent:
            return False
        self.uploads.pop(intent["key"], None)
        return True

    def register_asset(self, asset_id: str, remote_path: str, **metadata):
        pass

    def remove_input(self, remote_path: str):
        self.uploads.pop(remote_path, None)

    def cleanup_assets(self):
        return 0

    def download_output(self, filename: str):
        raise FileNotFoundError(filename)

    def output_delivery_url(self, filename: str, method="GET"):
        return self.delivery_url

    def copy_output_to_input(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job or not job.get("video_url"):
            raise FileNotFoundError(job_id)
        asset_id = uuid.uuid4().hex
        self.uploads[f"inputs/{asset_id}.mp4"] = b"server-copy"
        return {"id": asset_id, "kind": "video", "filename": f"{job_id}.mp4", "size": len(b"server-copy")}

    def list_loras(self):
        return self.loras


@pytest.fixture
def client(tmp_path, monkeypatch):
    control = FakeControl(tmp_path)
    monkeypatch.setattr(api_module, "modal_client", control)
    monkeypatch.setattr(api_module, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(api_module, "OUTPUT_CACHE", tmp_path / "cache" / "outputs")
    monkeypatch.setattr(api_module, "UPLOAD_CACHE", tmp_path / "cache" / "uploads")
    with TestClient(api_module.app) as test_client:
        yield test_client, control


def test_health_and_generation(client):
    test_client, _ = client
    health = test_client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["transport"] == "local-modal-sdk"
    assert "transfer_metrics" in health.json()

    response = test_client.post("/api/jobs", json={"prompt": "A crane flies over Tokyo"})
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "completed"
    assert job["video_url"].endswith(".mp4")

    assert test_client.get(f"/api/jobs/{job['id']}").status_code == 200
    status = test_client.get("/api/jobs/status", params={"session_number": job["session_number"]})
    assert status.status_code == 200
    assert status.json()[0]["request"]["prompt"] == "A crane flies over Tokyo"
    assert "negative_prompt" not in status.json()[0]["request"]

    reused = test_client.post(f"/api/assets/from-job/{job['id']}")
    assert reused.status_code == 201
    assert reused.json()["kind"] == "video"
    assert test_client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert test_client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_queue_full_maps_to_http_429(client):
    test_client, control = client
    control.create_job = lambda request: (_ for _ in ()).throw(
        QueueFullError("GPU queue is full (4/4)")
    )

    response = test_client.post("/api/jobs", json={"prompt": "queued"})

    assert response.status_code == 429
    assert "GPU queue is full" in response.json()["detail"]


def test_validation():
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", width=513)
    with pytest.raises(ValueError):
        GenerateRequest(mode="i2v", prompt="x")
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", num_frames=120)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", quality="high", width=1024, height=512)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", upscale=False, decoder="diffusion")

    temporal = GenerateRequest(prompt="x", upscale=False, temporal_upscale=True, decoder="diffusion")
    assert temporal.temporal_upscale is True

    direct_1080p = GenerateRequest(prompt="x", width=1920, height=1088, upscale=False)
    assert (direct_1080p.width, direct_1080p.height) == (1920, 1088)
    latent_1080p = GenerateRequest(prompt="x", width=960, height=544, upscale=True)
    assert latent_1080p.upscale is True
    pixel = GenerateRequest(prompt="x", upscale=True, upscale_method="pixel")
    assert pixel.upscale_method == "pixel"

    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", upscale=False, upscale_method="pixel")
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", mode="t2i", upscale_method="pixel")
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", width=1280, height=704, upscale=True)
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", loras=[{"id": "style.safetensors"}, {"id": "style.safetensors"}])
    with pytest.raises(ValueError):
        ConcatRequest(job_ids=["a" * 32, "a" * 32])


def test_render_options_and_modes():
    draft = GenerateRequest(prompt="x", quality="draft")
    assert draft.upscale is False

    retake = GenerateRequest(
        mode="retake",
        prompt="replace the action",
        conditions=[{"asset_id": "a" * 32, "kind": "video"}],
        retake_start=1,
        retake_end=3,
        upscale=True,
    )
    assert retake.upscale is False
    assert retake.decoder == "vae"

    extend = GenerateRequest(
        mode="extend",
        prompt="continue walking",
        conditions=[{"asset_id": "b" * 32, "kind": "video"}],
        extend_direction="start",
        extend_seconds=4,
        extend_context_seconds=2,
        upscale=True,
    )
    assert extend.upscale is False

    a2v = GenerateRequest(prompt="visualize the beat", mode="a2v", audio_asset_id="c" * 32)
    assert a2v.audio_duration is None

    iclora = GenerateRequest(
        prompt="Generated video: the character walks",
        mode="iclora",
        conditions=[{"asset_id": "d" * 32, "kind": "image", "index": 1}],
        loras=[{"id": "ingredients.safetensors", "strength": 1.4}],
    )
    assert iclora.upscale is False


def test_sessions_and_prompt_enhancer(client):
    test_client, _ = client
    session = test_client.post("/api/sessions")
    assert session.status_code == 201
    session_number = session.json()["session_number"]
    history = test_client.get("/api/jobs", params={"session_number": session_number})
    assert history.status_code == 200
    assert history.json() == []
    enhancer = test_client.post("/api/prompts/enhance", json={"prompt": "A person walks"})
    assert enhancer.status_code == 503


def test_warm_and_unload_toggle_keep_warm(client):
    test_client, control = client

    response = test_client.post("/api/admin/warm")
    assert response.status_code == 200
    assert control.keep_gpu_warm is True
    assert control.idle_windows[-1] == control.gpu_idle_seconds
    assert control.warmups == 1

    response = test_client.post("/api/admin/unload")
    assert response.status_code == 200
    assert control.keep_gpu_warm is False
    assert control.idle_windows[-1] == 2
    assert test_client.get("/api/health").json()["keep_gpu_warm"] is False


def test_app_shutdown_uses_two_second_idle_window(tmp_path, monkeypatch):
    control = FakeControl(tmp_path)
    control.keep_gpu_warm = True
    monkeypatch.setattr(api_module, "modal_client", control)
    monkeypatch.setattr(api_module, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(api_module, "OUTPUT_CACHE", tmp_path / "cache" / "outputs")
    monkeypatch.setattr(api_module, "UPLOAD_CACHE", tmp_path / "cache" / "uploads")

    with TestClient(api_module.app):
        assert control.idle_windows[-1] == control.gpu_idle_seconds

    assert control.idle_windows[-1] == 2


def test_upload_and_i2v_request(client):
    test_client, control = client
    image_file = BytesIO()
    Image.new("RGB", (64, 64), "red").save(image_file, format="PNG")

    response = test_client.post(
        "/api/assets",
        files={"file": ("first.png", image_file.getvalue(), "image/png")},
    )
    assert response.status_code == 201
    asset = response.json()
    assert asset["kind"] == "image"
    assert f"inputs/{asset['id']}.png" in control.uploads
    assert list(api_module.UPLOAD_CACHE.iterdir()) == []

    response = test_client.post(
        "/api/jobs",
        json={
            "mode": "i2v",
            "prompt": "The red frame begins to move",
            "conditions": [{"asset_id": asset["id"], "kind": "image", "index": 0, "strength": 1}],
        },
    )
    assert response.status_code == 202
    assert response.json()["request"]["mode"] == "i2v"


def test_prepared_upload_uses_binary_put_and_finalize(client):
    test_client, control = client
    image_file = BytesIO()
    Image.new("RGB", (32, 32), "green").save(image_file, format="PNG")
    payload = image_file.getvalue()

    prepared = test_client.post(
        "/api/assets/prepare",
        json={"filename": "direct.png", "size": len(payload), "content_type": "image/png"},
    )
    assert prepared.status_code == 201
    plan = prepared.json()
    assert plan["mode"] == "proxy"
    assert plan["backend"] == "volume"

    uploaded = test_client.put(plan["url"], content=payload, headers=plan["headers"])
    assert uploaded.status_code == 204

    completed = test_client.post(
        f"/api/assets/{plan['asset_id']}/complete",
        json={"parts": []},
    )
    assert completed.status_code == 200
    asset = completed.json()
    assert asset["id"] == plan["asset_id"]
    assert asset["kind"] == "image"
    assert f"inputs/{asset['id']}.png" in control.uploads


def test_upload_rolls_back_remote_file_when_asset_registration_fails(client):
    test_client, control = client
    image_file = BytesIO()
    Image.new("RGB", (64, 64), "blue").save(image_file, format="PNG")
    control.register_asset = lambda *_: (_ for _ in ()).throw(RuntimeError("dict unavailable"))

    response = test_client.post(
        "/api/assets",
        files={"file": ("first.png", image_file.getvalue(), "image/png")},
    )

    assert response.status_code == 502
    assert control.uploads == {}


def test_lora_listing(client):
    test_client, control = client
    control.loras = [{
        "id": "cinematic.safetensors",
        "name": "cinematic",
        "size": 7,
        "kind": "standard",
        "model_version": None,
        "reference_downscale_factor": None,
        "generic_iclora_compatible": False,
    }]
    response = test_client.get("/api/loras")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "cinematic.safetensors"


def test_rejects_unsupported_upload(client):
    test_client, _ = client
    response = test_client.post(
        "/api/assets",
        files={"file": ("notes.txt", b"not media", "text/plain")},
    )
    assert response.status_code == 415


def test_output_route_redirects_when_media_store_supports_direct_download(client):
    test_client, control = client
    control.delivery_url = "https://s3.test/signed-output"

    response = test_client.get("/outputs/test.mp4", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "https://s3.test/signed-output"
