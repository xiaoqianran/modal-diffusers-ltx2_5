from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from ltx25.schemas import GenerateRequest
from ltx25.modal_client import ActiveJobError, ModalClient


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


class Volume:
    def __init__(self):
        self.files = {}
        self.removed = []

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def remove_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        self.removed.append(path)
        del self.files[path]


def make_client(tmp_path: Path):
    control = ModalClient.__new__(ModalClient)
    control.app_name = "test"
    control.model_id = "test"
    control.gpu_idle_seconds = 600
    control.keep_gpu_warm = False
    control.cache_root = tmp_path
    control.output_cache = tmp_path / "outputs"
    control.upload_cache = tmp_path / "uploads"
    control.state_volume = Volume()
    control.job_store = Store()
    control.worker = SimpleNamespace(update_autoscaler=lambda **_: None)
    control.generate_fn = SimpleNamespace(spawn=lambda *_: SimpleNamespace(object_id="fc-test"))
    control.ready_fn = SimpleNamespace(spawn=lambda: None)
    control._warm_call = None
    control._output_lock = threading.Lock()
    return control


def test_create_job_does_not_overwrite_fast_worker_state(tmp_path):
    control = make_client(tmp_path)

    def spawn(job_id, payload):
        record = control.job_store.get(f"job:{job_id}")
        record.update(status="completed", progress=1.0, video_url=f"/outputs/{job_id}.mp4")
        control.job_store.put(f"job:{job_id}", record)
        return SimpleNamespace(object_id="fc-test")

    control.generate_fn.spawn = spawn
    job = control.create_job(GenerateRequest(prompt="A bird"))

    assert job["status"] == "completed"
    assert "call_id" not in job
    assert control.job_store.get(f"call:{job['id']}") == "fc-test"


def test_active_job_cannot_be_deleted(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="A bird"))

    with pytest.raises(ActiveJobError):
        control.delete_job(job["id"])


def test_download_output_is_cached_locally(tmp_path):
    control = make_client(tmp_path)
    control.state_volume.files["outputs/test.mp4"] = b"video"

    first = control.download_output("test.mp4")
    assert first.read_bytes() == b"video"

    del control.state_volume.files["outputs/test.mp4"]
    second = control.download_output("test.mp4")
    assert second == first
    assert second.read_bytes() == b"video"
