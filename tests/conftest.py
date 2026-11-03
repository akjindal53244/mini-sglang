import pytest
import torch


@pytest.fixture(autouse=True)
def pageable_memory_without_a_gpu(monkeypatch):
    """Without a GPU, PyTorch cannot pin host memory. Pinning only speeds up
    host-to-device copies, so CPU-only runs can drop it and test the same logic."""
    if torch.cuda.is_available():
        return
    for name in ("empty", "tensor"):
        create = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, pin_memory=False, _c=create, **kw: _c(*a, **kw))
