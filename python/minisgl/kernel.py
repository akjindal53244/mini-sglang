"""Lazy bindings for the small C++/CUDA kernels in csrc/.

TVM FFI compiles a specialization on first use and caches its shared library.
Keep launch parameters here and the actual memory operations in the native files.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from minisgl.config import PYNCCL_MAX_BUFFER_SIZE

if TYPE_CHECKING:
    import torch

CSRC = Path(__file__).parent / "csrc"
INCLUDES = [str(CSRC / "include")]
CPP_FLAGS = ["-std=c++20", "-O3"]
CUDA_FLAGS = CPP_FLAGS + ["--expt-relaxed-constexpr"]


@cache
def _cuda_kernel(name: str, bytes_per_row: int, num_splits: int = 1):
    from tvm_ffi.cpp import load_inline

    # Index splits distribute a wide embedding row over several thread blocks.
    args = (bytes_per_row, num_splits) if name == "index" else (bytes_per_row,)
    args = tuple(str(arg).lower() for arg in args)
    function = "launch_index" if name == "index" else "launch_store"
    source = (
        f'#include "{CSRC / "jit" / (name + ".cu")}"\n'
        f'TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, ({function}<{", ".join(args)}>));'
    )
    return load_inline(
        "minisgl__" + "_".join((name, *args)),
        cuda_sources=[source],
        extra_cflags=CPP_FLAGS,
        extra_cuda_cflags=CUDA_FLAGS,
        extra_include_paths=INCLUDES,
    )


@cache
def _native_kernel(name: str):
    from tvm_ffi.cpp import load

    cuda = name == "pynccl"
    link_flags = []
    if cuda:
        # PyTorch's CUDA wheels ship libnccl.so.2 without the development
        # symlink required by -lnccl. Prefer that exact library when available.
        try:
            import nvidia.nccl
        except ImportError:
            link_flags = ["-lnccl"]
        else:
            library = Path(next(iter(nvidia.nccl.__path__))) / "lib/libnccl.so.2"
            link_flags = [str(library)] if library.is_file() else ["-lnccl"]
    filename = str(CSRC / "src" / (name + (".cu" if cuda else ".cpp")))
    return load(
        "minisgl__" + name,
        cpp_files=[] if cuda else [filename],
        cuda_files=[filename] if cuda else [],
        extra_cflags=CPP_FLAGS,
        extra_cuda_cflags=CUDA_FLAGS,
        extra_include_paths=INCLUDES,
        extra_ldflags=link_flags,
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Gather embedding rows; vocabulary shards return zeros for out-of-shard IDs."""
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])
    bytes_per_row = weights.shape[1] * weights.element_size()
    num_splits = 4 if bytes_per_row % 2048 == 0 else 2 if bytes_per_row % 1024 == 0 else 1
    _cuda_kernel("index", bytes_per_row, num_splits).launch(weights, indices, output, vocab_range)
    return output


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    # indices are physical token slots. Input K/V are strided QKV slices;
    # the kernel copies each into the corresponding dense cache row.
    k_cache, v_cache = k_cache.view(len(k_cache), -1), v_cache.view(len(v_cache), -1)
    bytes_per_row = k_cache.shape[1] * k_cache.element_size()
    _cuda_kernel("store", bytes_per_row).launch(k_cache, v_cache, indices, k, v)


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """Length of the equal prefix of two one-dimensional CPU token tensors."""
    return _native_kernel("radix").fast_compare_key(x, y)


@cache
def _nccl_type():
    import tvm_ffi

    # Register once: native NCCLWrapper objects expose all_reduce/all_gather.
    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class NCCLWrapper(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return NCCLWrapper


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
):
    import torch.distributed as dist

    module = _native_kernel("pynccl")
    # Gloo distributes the communicator ID; the resulting NCCL communicator
    # moves GPU tensors on the current CUDA stream, including during capture.
    ids = [module.create_nccl_uid() if tp_rank == 0 else None]
    dist.broadcast_object_list(ids, src=0, group=tp_cpu_group)
    return _nccl_type()(tp_rank, tp_size, min(max_size_bytes, PYNCCL_MAX_BUFFER_SIZE), ids[0])
