from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import ltx25.modal_client as modal_client_module
from ltx25.media_store import VolumeMediaStore
from ltx25.media_storage import MediaStorage
from ltx25.schemas import GenerateRequest
from ltx25.modal_client import (
    ActiveJobError,
    JobStateError,
    ModalClient,
    ModalOperationError,
    QueueFullError,
    SubmissionError,
)


class Store:
    def __init__(self):
        self.data = {}
        self.items_calls = 0

    def get(self, key):
        return deepcopy(self.data.get(key))

    def put(self, key, value):
        self.data[key] = deepcopy(value)

    def items(self):
        self.items_calls += 1
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

    def copy_files(self, src_paths, dst_path, recursive=False):
        assert len(src_paths) == 1
        source = src_paths[0]
        if source not in self.files:
            raise FileNotFoundError(source)
        self.files[dst_path] = self.files[source]

    def iterdir(self, path, recursive=False):
        if path not in self.files:
            raise FileNotFoundError(path)
        return [SimpleNamespace(path=path, size=len(self.files[path]))]

    class _Batch:
        def __init__(self, volume):
            self.volume = volume

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def put_file(self, local_path, remote_path):
            self.volume.files[remote_path] = Path(local_path).read_bytes()

    def batch_upload(self, force=False):
        return self._Batch(self)


def make_client(tmp_path: Path):
    control = ModalClient.__new__(ModalClient)
    control.app_name = "test"
    control.model_id = "test"
    control.gpu_idle_seconds = 600
    control.keep_gpu_warm = False
    control.warm_lease_seconds = 90
    control.max_queue_size = 4
    control._warm_lease_until = 0.0
    control.cache_root = tmp_path
    control.output_cache = tmp_path / "outputs"
    control.upload_cache = tmp_path / "uploads"
    control.state_volume = Volume()
    control.media_store = VolumeMediaStore(control.state_volume)
    control.media_storage = MediaStorage(stores={"volume": control.media_store}, primary_id="volume")
    control.job_store = Store()
    control.worker = SimpleNamespace(update_autoscaler=lambda **_: None)
    control.generate_fn = SimpleNamespace(spawn=lambda *_: SimpleNamespace(object_id="fc-test"))
    control.ready_fn = SimpleNamespace(spawn=lambda: None)
    control._warm_call = None
    control._warm_lock = threading.RLock()
    control._output_lock = threading.Lock()
    control._session_lock = threading.Lock()
    control._admission_lock = threading.RLock()
    control._queue_snapshot = {
        "queued": 0,
        "running": 0,
        "active": 0,
        "capacity": control.max_queue_size,
        "available": False,
    }
    control._active_index_initialized = False
    control.job_store.put(control._active_jobs_key(), [])
    return control


def test_list_loras_treats_missing_modal_directory_as_empty(tmp_path):
    control = make_client(tmp_path)

    class MissingModalDirectoryVolume(Volume):
        def iterdir(self, path, recursive=False):
            raise modal_client_module.modal.exception.NotFoundError("No such file or directory")

    control.state_volume = MissingModalDirectoryVolume()

    assert control.list_loras() == []


def test_constructor_does_not_hydrate_modal_state(monkeypatch, tmp_path):
    class NoIoStore:
        def get(self, *_args, **_kwargs):
            raise AssertionError("constructor must not read Modal Dict")

        def items(self):
            raise AssertionError("constructor must not scan Modal Dict")

        def put(self, *_args, **_kwargs):
            raise AssertionError("constructor must not write Modal Dict")

    volume = Volume()
    store = NoIoStore()
    media = MediaStorage(
        stores={"volume": VolumeMediaStore(volume)},
        primary_id="volume",
    )
    worker = SimpleNamespace(
        generate=SimpleNamespace(),
        ready=SimpleNamespace(),
    )

    monkeypatch.setenv("LTX25_LOCAL_CACHE", str(tmp_path))
    monkeypatch.setattr(modal_client_module.modal.Volume, "from_name", lambda *_a, **_k: volume)
    dict_calls = []

    def dict_from_name(*args, **kwargs):
        dict_calls.append((args, kwargs))
        return store

    monkeypatch.setattr(modal_client_module.modal.Dict, "from_name", dict_from_name)
    monkeypatch.setattr(modal_client_module.modal.Cls, "from_name", lambda *_a, **_k: lambda: worker)
    monkeypatch.setattr(modal_client_module, "create_media_storage", lambda _volume: media)

    control = ModalClient()

    assert control.job_store is store
    assert control._active_index_initialized is False
    assert dict_calls == [((modal_client_module.JOB_DICT_NAME,), {"create_if_missing": True})]


def test_active_index_is_lazy_backfilled_once_and_repairs_shape(tmp_path):
    control = make_client(tmp_path)
    control.job_store.pop(control._active_jobs_key(), None)
    control.job_store.put(
        "job:queued",
        {"id": "queued", "status": "queued"},
    )
    control.job_store.put(
        "job:done",
        {"id": "done", "status": "completed"},
    )
    control.job_store.items_calls = 0

    assert control.queue_stats()["active"] == 1
    assert control.job_store.get(control._active_jobs_key()) == ["queued"]
    assert control.job_store.items_calls == 1

    # Subsequent queue reads stay on the compact index and do not rescan.
    assert control.queue_stats()["active"] == 1
    assert control.job_store.items_calls == 1

    # A malformed persisted index is normalized without a global job scan.
    control._active_index_initialized = False
    control.job_store.put(control._active_jobs_key(), ["queued", "", "queued", 123])
    control.job_store.items_calls = 0
    control.queue_stats()
    assert control.job_store.get(control._active_jobs_key()) == ["queued"]
    assert control.job_store.items_calls == 0


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
    metrics = control.transfer_metrics()["download"]
    assert metrics["count"] == 1
    assert metrics["bytes"] == 5
    assert metrics["cache_hits"] == 1


def test_completed_output_can_be_reused_inside_volume_without_download(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="source", session_number=101))
    record = control.job_store.get(f"job:{job['id']}")
    record.update(status="completed", video_url=f"/outputs/{job['id']}.mp4")
    control.job_store.put(f"job:{job['id']}", record)
    control.state_volume.files[f"outputs/{job['id']}.mp4"] = b"video-data"

    asset = control.copy_output_to_input(job["id"])

    remote_path = control.job_store.get(f"asset:{asset['id']}")["key"]
    assert control.state_volume.files[remote_path] == b"video-data"
    assert asset["kind"] == "video"
    assert asset["size"] == len(b"video-data")
    assert control.transfer_metrics()["copy"]["count"] == 1
    assert control.transfer_metrics()["download"]["count"] == 0


def test_completed_image_output_can_be_reused_inside_volume_without_download(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="source image", mode="t2i", session_number=101))
    record = control.job_store.get(f"job:{job['id']}")
    record.update(status="completed", image_url=f"/outputs/{job['id']}.png")
    control.job_store.put(f"job:{job['id']}", record)
    control.state_volume.files[f"outputs/{job['id']}.png"] = b"image-data"

    asset = control.copy_output_to_input(job["id"])

    remote_path = control.job_store.get(f"asset:{asset['id']}")["key"]
    assert control.state_volume.files[remote_path] == b"image-data"
    assert remote_path.endswith(".png")
    assert asset["kind"] == "image"
    assert asset["size"] == len(b"image-data")
    assert control.transfer_metrics()["copy"]["count"] == 1
    assert control.transfer_metrics()["download"]["count"] == 0


def test_job_summary_keeps_render_fields_but_drops_heavy_request_details(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="summary", negative_prompt="very long negative", session_number=101))

    summary = control.list_job_summaries(101)[0]

    assert summary["id"] == job["id"]
    assert summary["request"]["prompt"] == "summary"
    assert summary["request"]["mode"] == "t2av"
    assert "negative_prompt" not in summary["request"]


def test_prepared_asset_finalize_is_idempotent(tmp_path):
    control = make_client(tmp_path)
    payload = tmp_path / "input.png"
    payload.write_bytes(b"media")
    plan = control.prepare_asset_upload(
        filename="input.png",
        content_type="image/png",
        size=5,
        kind="image",
        suffix=".png",
    )
    control.upload_proxy_file(plan["asset_id"], payload)

    first = control.complete_asset_upload(plan["asset_id"], client_upload_seconds=0.25)
    second = control.complete_asset_upload(plan["asset_id"], client_upload_seconds=9.0)

    assert second == first
    assert control.transfer_metrics()["upload"]["client_count"] == 1
    assert control.transfer_metrics()["upload"]["client_seconds"] == 0.25


def test_public_job_materializes_stable_url_from_internal_media_key(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="keyed", session_number=101))
    record = control.job_store.get(f"job:{job['id']}")
    record.update(status="completed", video_key=f"outputs/{job['id']}.mp4", video_url=None)
    control.job_store.put(f"job:{job['id']}", record)

    public = control.get_job(job["id"])

    assert public["video_url"] == f"/outputs/{job['id']}.mp4"
    assert "video_key" not in public


def test_keep_warm_dedupes_pending_call_and_unload_cancels_it(tmp_path):
    control = make_client(tmp_path)
    idle_windows = []
    spawned = []

    class PendingCall:
        cancelled = False

        def get(self, timeout=0):
            raise TimeoutError

        def cancel(self, terminate_containers=False):
            self.cancelled = True

    pending = PendingCall()
    control.worker = SimpleNamespace(
        update_autoscaler=lambda **kwargs: idle_windows.append(kwargs["scaledown_window"])
    )
    control.ready_fn = SimpleNamespace(spawn=lambda: spawned.append(pending) or pending)

    assert control.start_warmup() is False
    assert control.enable_keep_warm() is True
    assert idle_windows[-1] == control.gpu_idle_seconds
    assert len(spawned) == 1
    assert control.start_warmup() is False
    assert len(spawned) == 1

    control.disable_keep_warm()
    assert control.keep_gpu_warm is False
    assert pending.cancelled is True
    assert idle_windows[-1] == 2
    assert control.start_warmup() is False


def test_expired_warm_lease_scales_down_when_queue_is_idle(tmp_path):
    control = make_client(tmp_path)
    idle_windows = []
    control.keep_gpu_warm = True
    control._warm_lease_until = 0.0
    control.worker = SimpleNamespace(
        update_autoscaler=lambda **kwargs: idle_windows.append(kwargs["scaledown_window"])
    )

    assert control.maintain_keep_warm() is False
    assert control.keep_gpu_warm is False
    assert idle_windows[-1] == 2


def test_active_job_keeps_gpu_warm_after_studio_lease_expires(tmp_path):
    control = make_client(tmp_path)
    idle_windows = []
    control.worker = SimpleNamespace(
        update_autoscaler=lambda **kwargs: idle_windows.append(kwargs["scaledown_window"])
    )
    control.create_job(GenerateRequest(prompt="queued"))
    control._warm_lease_until = 0.0

    # The queued generation call itself starts/keeps the worker; no redundant
    # ready() call is spawned when only an active-job lease remains.
    assert control.maintain_keep_warm() is False
    assert control.keep_gpu_warm is True
    assert idle_windows[-1] == control.gpu_idle_seconds


def test_queue_capacity_is_enforced_before_spawn(tmp_path):
    control = make_client(tmp_path)
    control.max_queue_size = 2
    first = control.create_job(GenerateRequest(prompt="first"))
    control.create_job(GenerateRequest(prompt="second"))

    with pytest.raises(QueueFullError, match="GPU queue is full"):
        control.create_job(GenerateRequest(prompt="third"))

    record = control.job_store.get(f"job:{first['id']}")
    record["status"] = "completed"
    control.job_store.put(f"job:{first['id']}", record)
    third = control.create_job(GenerateRequest(prompt="third"))
    assert third["status"] == "queued"


def test_interrupt_without_id_prefers_running_job(tmp_path, monkeypatch):
    control = make_client(tmp_path)
    queued = control.create_job(GenerateRequest(prompt="queued"))
    running = control.create_job(GenerateRequest(prompt="running"))
    running_record = control.job_store.get(f"job:{running['id']}")
    running_record["status"] = "running"
    control.job_store.put(f"job:{running['id']}", running_record)

    cancelled = []

    class Call:
        def __init__(self, call_id):
            self.call_id = call_id

        def cancel(self, terminate_containers=False):
            cancelled.append(self.call_id)

    monkeypatch.setattr(
        modal_client_module.modal.FunctionCall,
        "from_id",
        lambda call_id: Call(call_id),
    )

    result = control.interrupt(None)

    assert result["interrupted"] is True
    assert result["current_job_id"] == running["id"]
    assert cancelled == ["fc-test"]
    assert control.job_store.get(f"job:{running['id']}")["status"] == "interrupted"
    assert control.job_store.get(f"cancel:{running['id']}")["requested_at"]
    assert control.job_store.get(f"job:{queued['id']}")["status"] == "queued"


def test_session_job_index_avoids_global_scan_after_creation(tmp_path):
    control = make_client(tmp_path)
    first = control.create_job(GenerateRequest(prompt="first", session_number=101))
    control.create_job(GenerateRequest(prompt="other", session_number=202))
    control.job_store.items_calls = 0

    jobs = control.list_jobs(session_number=101)

    assert [job["id"] for job in jobs] == [first["id"]]
    assert control.job_store.items_calls == 0


def test_delete_job_removes_session_and_cancel_indexes(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="done", session_number=101))
    record = control.job_store.get(f"job:{job['id']}")
    record["status"] = "completed"
    control.job_store.put(f"job:{job['id']}", record)
    control.job_store.put(f"cancel:{job['id']}", {"requested_at": "now"})

    assert control.delete_job(job["id"]) is True
    assert control.job_store.get("session:101:jobs") is None
    assert control.job_store.get(f"cancel:{job['id']}") is None


def test_asset_cleanup_preserves_active_inputs_then_removes_stale_terminal_inputs(tmp_path):
    control = make_client(tmp_path)
    asset_id = "c" * 32
    remote_path = f"inputs/{asset_id}.png"
    control.state_volume.files[remote_path] = b"image"
    control.register_asset(asset_id, remote_path)

    job = control.create_job(GenerateRequest(
        mode="i2v",
        prompt="move",
        session_number=101,
        conditions=[{"asset_id": asset_id, "kind": "image", "index": 0}],
    ))
    assert control.cleanup_assets(max_age_seconds=0) == 0
    assert remote_path in control.state_volume.files

    record = control.job_store.get(f"job:{job['id']}")
    record["status"] = "completed"
    control.job_store.put(f"job:{job['id']}", record)

    assert control.cleanup_assets(max_age_seconds=0) == 1
    assert remote_path not in control.state_volume.files
    assert control.job_store.get(f"asset:{asset_id}") is None


def test_interrupt_requires_call_id_for_active_job(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="queued"))
    control.job_store.pop(f"call:{job['id']}")

    with pytest.raises(JobStateError):
        control.interrupt(job["id"])


def test_interrupt_wraps_modal_cancel_failure(tmp_path, monkeypatch):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="queued"))

    class Call:
        def cancel(self, terminate_containers=False):
            raise RuntimeError("cancel failed")

    monkeypatch.setattr(
        modal_client_module.modal.FunctionCall,
        "from_id",
        lambda _call_id: Call(),
    )

    with pytest.raises(ModalOperationError):
        control.interrupt(job["id"])


def test_interrupt_without_id_uses_oldest_queued_when_nothing_is_running(tmp_path, monkeypatch):
    control = make_client(tmp_path)
    first = control.create_job(GenerateRequest(prompt="first"))
    second = control.create_job(GenerateRequest(prompt="second"))
    first_record = control.job_store.get(f"job:{first['id']}")
    second_record = control.job_store.get(f"job:{second['id']}")
    first_record["created_at"] = "2026-01-01T00:00:00+00:00"
    second_record["created_at"] = "2026-01-02T00:00:00+00:00"
    control.job_store.put(f"job:{first['id']}", first_record)
    control.job_store.put(f"job:{second['id']}", second_record)

    class Call:
        def cancel(self, terminate_containers=False):
            pass

    monkeypatch.setattr(
        modal_client_module.modal.FunctionCall,
        "from_id",
        lambda _call_id: Call(),
    )

    result = control.interrupt(None)

    assert result["current_job_id"] == first["id"]
    assert control.job_store.get(f"job:{first['id']}")["status"] == "interrupted"
    assert control.job_store.get(f"job:{second['id']}")["status"] == "queued"


def test_submission_failure_is_persisted(tmp_path):
    control = make_client(tmp_path)
    control.generate_fn = SimpleNamespace(
        spawn=lambda *_: (_ for _ in ()).throw(RuntimeError("submit failed"))
    )

    with pytest.raises(SubmissionError):
        control.create_job(GenerateRequest(prompt="fails"))

    jobs = [item for key, item in control.job_store.items() if key.startswith("job:")]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "failed"
    assert "submit failed" in jobs[0]["error"]


def test_submission_refreshes_stale_deployment_handle_once(tmp_path):
    control = make_client(tmp_path)
    attempts = []

    def stale_spawn(*_args):
        attempts.append("stale")
        raise modal_client_module.modal.exception.ConflictError("old deployment")

    control.generate_fn = SimpleNamespace(spawn=stale_spawn)

    def refresh():
        attempts.append("refresh")
        control.generate_fn = SimpleNamespace(
            spawn=lambda *_args: SimpleNamespace(object_id="fc-new")
        )

    control._refresh_remote_handles = refresh
    job = control.create_job(GenerateRequest(prompt="redeploy boundary"))

    assert job["status"] == "queued"
    assert attempts == ["stale", "refresh"]
    assert control.job_store.get(f"call:{job['id']}") == "fc-new"


def test_queue_snapshot_updates_without_health_remote_scan(tmp_path):
    control = make_client(tmp_path)
    job = control.create_job(GenerateRequest(prompt="snapshot"))

    snapshot = control.queue_snapshot()
    assert snapshot["available"] is True
    assert snapshot["active"] == 1
    assert snapshot["queued"] == 1

    record = control.job_store.get(f"job:{job['id']}")
    record["status"] = "completed"
    control.job_store.put(f"job:{job['id']}", record)
    control._remove_active_job(job["id"])

    snapshot = control.queue_snapshot()
    assert snapshot["active"] == 0
