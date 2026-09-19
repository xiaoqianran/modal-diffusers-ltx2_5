from copy import deepcopy

import modal_app


class Store:
    def __init__(self, data):
        self.data = deepcopy(data)

    def get(self, key):
        return deepcopy(self.data.get(key))

    def put(self, key, value):
        self.data[key] = deepcopy(value)


def test_cancel_request_wins_over_a_later_worker_completion(monkeypatch):
    job_id = "a" * 32
    store = Store(
        {
            f"cancel:{job_id}": {"requested_at": "2026-09-20T00:00:01+00:00"},
            f"job:{job_id}": {
                "id": job_id,
                "status": "completed",
                "updated_at": "2026-09-20T00:00:02+00:00",
                "error": None,
            },
        }
    )
    monkeypatch.setattr(modal_app, "job_store", store)

    result = modal_app._honor_interrupt(job_id, store.get(f"job:{job_id}"))

    assert result["status"] == "interrupted"
    assert store.get(f"job:{job_id}")["status"] == "interrupted"


def test_completion_that_precedes_cancel_request_is_preserved(monkeypatch):
    job_id = "b" * 32
    completed = {
        "id": job_id,
        "status": "completed",
        "updated_at": "2026-09-20T00:00:01+00:00",
        "error": None,
    }
    store = Store(
        {
            f"cancel:{job_id}": {"requested_at": "2026-09-20T00:00:02+00:00"},
            f"job:{job_id}": completed,
        }
    )
    monkeypatch.setattr(modal_app, "job_store", store)

    assert modal_app._honor_interrupt(job_id, completed) is None
    assert store.get(f"job:{job_id}")["status"] == "completed"
