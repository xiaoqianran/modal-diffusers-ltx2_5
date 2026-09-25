from ltx25.acceleration import cuda_graph


def test_repeated_overflow_shape_replaces_single_stale_capture(monkeypatch):
    class Module:
        def forward(self, **kwargs):
            return kwargs["shape"]

    class Graph:
        def replay(self):
            pass

    monkeypatch.setattr(cuda_graph.torch.cuda, "synchronize", lambda: None)

    runner = cuda_graph.ForwardGraphRunner(Module(), max_captures=1)
    old_key = runner._key({"shape": 1})
    runner._captures[old_key] = {"graph": Graph(), "inputs": {}, "outputs": 1}
    runner._capture = lambda kwargs: {
        "graph": Graph(),
        "inputs": {},
        "outputs": kwargs["shape"],
    }

    # First miss is deliberately eager so a one-off shape cannot evict the
    # resident graph.
    assert runner(shape=2) == 2
    assert runner.stats()["captures"] == 1
    assert runner.stats()["replacements"] == 0

    # Repeating the exact miss promotes it into the one safe resident slot.
    assert runner(shape=2) == 2
    stats = runner.stats()
    assert stats["captures"] == 1
    assert stats["replacements"] == 1
    assert stats["overflow_misses"] == 2

    # It now replays rather than falling back to eager.
    assert runner(shape=2) == 2
    assert runner.stats()["replays"] == 2
