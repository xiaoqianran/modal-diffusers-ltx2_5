"""Behavioral regressions for ported upstream graph admission."""
from ltx25.acceleration import cuda_graph


def test_large_shape_stays_eager_without_capture(monkeypatch):
    class Module:
        def forward(self, **kwargs):
            return kwargs["hidden_states"]
    class Shape:
        def __init__(self, tokens):
            self.shape = (1, tokens, 8)
            self.ndim = 3
    class TensorLike:
        pass
    monkeypatch.setattr(cuda_graph.torch, "Tensor", TensorLike)
    class FakeTensor(TensorLike, Shape):
        def __init__(self, tokens):
            Shape.__init__(self, tokens)
        @property
        def dtype(self):
            return "bf16"
        @property
        def device(self):
            return "cuda"
    runner = cuda_graph.ForwardGraphRunner(Module(), max_tokens=8192)
    runner._capture = lambda kwargs: (_ for _ in ()).throw(AssertionError("large graph captured"))
    sample = FakeTensor(8193)
    assert runner(hidden_states=sample) is sample
    assert runner.stats()["captures"] == 0
    assert runner.stats()["eager_shapes"] == 1
    assert runner(hidden_states=sample) is sample


def test_graph_limit_can_be_disabled():
    class Module:
        def forward(self, **kwargs):
            return 0
    runner = cuda_graph.ForwardGraphRunner(Module(), max_tokens=0)
    assert runner._max_tokens == 0
