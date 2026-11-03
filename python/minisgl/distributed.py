"""Rank information and GPU communication for tensor and pipeline parallelism."""

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


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


_PYNCCL = None


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum tensor-parallel partial results in place, using PyNCCL or PyTorch."""
    if get_tp_info().size > 1:
        if _PYNCCL is not None:
            _PYNCCL.all_reduce(x, "sum")
        else:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """Concatenate each rank's tensor along dimension zero."""
    tp_size = get_tp_info().size
    if tp_size == 1:
        return x
    output = x.new_empty((x.shape[0] * tp_size, *x.shape[1:]))
    if _PYNCCL is not None:
        _PYNCCL.all_gather(output, x)
    else:
        dist.all_gather_into_tensor(output, x)
    return output


def stage_layers(num_layers: int, pp: DistributedInfo) -> range:
    """The contiguous decoder layers one pipeline stage owns; early stages take extras."""
    assert pp.size <= num_layers, f"{pp.size} pipeline stages need that many layers"
    base, extra = divmod(num_layers, pp.size)
    start = pp.rank * base + min(pp.rank, extra)
    return range(start, start + base + (pp.rank < extra))


# Pipeline stages use PyNCCL alone: it enqueues on the current CUDA stream, which
# lets graphs capture these calls. Config rejects PP without it.
def send(x: torch.Tensor, dst: int) -> None:
    _PYNCCL.send(x, dst)


def recv(x: torch.Tensor, src: int) -> None:
    _PYNCCL.recv(x, src)


def broadcast(x: torch.Tensor, src: int) -> None:
    _PYNCCL.broadcast(x, src)


def enable_pynccl_distributed(
    world: DistributedInfo, cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    global _PYNCCL
    if world.size > 1:
        from minisgl.kernel import init_pynccl

        _PYNCCL = init_pynccl(
            tp_rank=world.rank,
            tp_size=world.size,
            tp_cpu_group=cpu_group,
            max_size_bytes=max_bytes,
        )


def destroy_distributed() -> None:
    global _PYNCCL, _TP_INFO
    _PYNCCL = None
    _TP_INFO = None
