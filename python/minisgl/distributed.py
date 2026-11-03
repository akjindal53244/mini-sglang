"""Rank information and GPU communication for tensor and pipeline parallelism.

Every collective is a plain torch.distributed call on the NCCL backend. PyTorch
supports capturing these calls, so decode CUDA graphs record them like kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None  # Shards each layer; trivial under PP.
_WORLD_INFO: DistributedInfo | None = None  # Every rank in the job, TP or PP.


def set_parallel_info(tp: DistributedInfo, world: DistributedInfo) -> None:
    global _TP_INFO, _WORLD_INFO
    _TP_INFO, _WORLD_INFO = tp, world


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def get_world_info() -> DistributedInfo:
    """This process among all ranks; a lone process before an engine starts."""
    return _WORLD_INFO or DistributedInfo(0, 1)


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum tensor-parallel partial results in place."""
    if get_tp_info().size > 1:
        dist.all_reduce(x)
    return x


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """Concatenate each rank's tensor along dimension zero."""
    tp_size = get_tp_info().size
    if tp_size == 1:
        return x
    output = x.new_empty((x.shape[0] * tp_size, *x.shape[1:]))
    dist.all_gather_into_tensor(output, x)
    return output


def stage_layers(num_layers: int, pp: DistributedInfo) -> range:
    """The contiguous decoder layers one pipeline stage owns; early stages take extras."""
    assert pp.size <= num_layers, f"{pp.size} pipeline stages need that many layers"
    base, extra = divmod(num_layers, pp.size)
    start = pp.rank * base + min(pp.rank, extra)
    return range(start, start + base + (pp.rank < extra))


def send(x: torch.Tensor, dst: int) -> None:
    dist.send(x, dst)


def recv(x: torch.Tensor, src: int) -> None:
    dist.recv(x, src)


def broadcast(x: torch.Tensor, src: int) -> None:
    """Copy src's tensor to every rank."""
    dist.broadcast(x, src)


def destroy_distributed() -> None:
    global _TP_INFO, _WORLD_INFO
    _TP_INFO = _WORLD_INFO = None
