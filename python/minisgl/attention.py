"""Attention adapters: fi (FlashInfer FA2) and trtllm (NVIDIA TRT-LLM kernels)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from minisgl.config import FLASHINFER_USE_TENSOR_CORES
from minisgl.core import Batch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_ceil, div_even, init_logger

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.config import ModelConfig
    from minisgl.kvcache import MHAKVCache


class HybridBackend:
    """Choose a kernel by phase; CUDA graphs capture only the decode backend."""

    def __init__(
        self,
        prefill_backend: AttentionBackend,
        decode_backend: AttentionBackend,
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)


@dataclass
class CaptureBuffers:
    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
        )


logger = init_logger(__name__)


@dataclass
class FIMetadata:
    cu_seqlens_q_cpu: torch.Tensor
    cu_seqlens_k_cpu: torch.Tensor
    indices: torch.Tensor  # GPU physical token slots, concatenated across requests.
    last_page_len_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized: bool = False


class FlashInferBackend:
    def __init__(self, config: ModelConfig, kv_cache: MHAKVCache, page_table: torch.Tensor) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kv_cache
        self.page_table = page_table
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # Portable default; kernel alternatives are explicit backends.
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # Portable default; kernel alternatives are explicit backends.
        )

        # Both planners share workspace; last_event below guards its asynchronous reuse.
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrapper._int_workspace_buffer = self.int_workspace_buffer

        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        self.capture_bs: list[int] = []
        self.graph_wrappers: dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: CaptureBuffers | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=self.qo_head_local,
                num_kv_heads=self.kv_head_local,
                head_dim=self.config.head_dim,
                page_size=1,
                pos_encoding_mode="NONE",
                seq_lens=metadata.seq_lens_cpu,
                data_type=self.kvcache.dtype,
                q_data_type=self.kvcache.dtype,
                kv_data_type=self.kvcache.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=self.qo_head_local,
                num_kv_heads=self.kv_head_local,
                head_dim_qk=self.config.head_dim,
                page_size=1,
                pos_encoding_mode="NONE",
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=self.kvcache.dtype,
                kv_data_type=self.kvcache.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = 1 << (bs - 1).bit_length()
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
            # FI sees token-sized pages, independently of the allocator's page
            # size. Thus its indices are the original physical token offsets.
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k[layer_id], self.kvcache.v[layer_id])
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)  # [T, q_heads, head_dim]

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.num_new_tokens for req in reqs]
        seqlens_k = [req.scheduled_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all num_new_tokens = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = self.page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            indices=torch.cat([page_table[req.table_idx, : req.scheduled_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            seq_lens_cpu=seq_len_cpu,
            wrapper=self.decode_wrapper if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = CaptureBuffers.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (override := FLASHINFER_USE_TENSOR_CORES) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {override}")
            return override
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.page_table,
            last_page_len_buffer=capture.seq_lens[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)


@dataclass
class PagedAttentionMetadata:
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor


class TRTLLMBackend:
    """TRT-LLM reads whole allocator pages, so its metadata holds page numbers."""

    def __init__(
        self,
        config: ModelConfig,
        kv_cache: MHAKVCache,
        page_table: torch.Tensor,
        page_size: int,
    ):
        self.config = config
        self.kvcache = kv_cache
        self.page_table = page_table
        self.page_size = page_size
        self.capture: CaptureBuffers | None = None
        self.capture_bs: list[int] = []
        self.scale = config.head_dim**-0.5
        # XQA uses this workspace for synchronization counters; first use needs zeros.
        # https://docs.flashinfer.ai/generated/flashinfer.decode.xqa_batch_decode_with_kv_cache.html
        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        metadata = batch.attn_metadata
        assert isinstance(metadata, PagedAttentionMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k[layer_id], self.kvcache.v[layer_id])
        # Q is a strided slice of fused QKV, but these kernels read packed rows. A
        # single-request batch hides the mismatch; later rows would read garbage.
        q = q.contiguous()
        if batch.is_prefill:
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_q_len=metadata.max_seqlen_q,
                max_kv_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                cum_seq_lens_q=metadata.cu_seqlens_q,
                cum_seq_lens_kv=metadata.cu_seqlens_k,
                kv_layout="NHD",
                batch_size=batch.size,
                out_dtype=q.dtype,
            )
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens,
            max_seq_len=metadata.max_seqlen_k,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
            out_dtype=q.dtype,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.num_new_tokens for req in reqs]
        seqlens_k = [req.scheduled_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = self.page_table
        # The scheduler stores token slots. Keep one entry per physical page,
        # then convert its starting token offset to the kernel's page number.
        new_page_table = torch.stack(
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = PagedAttentionMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = CaptureBuffers.create(
            max_bs, div_ceil(max_seq_len, self.page_size), self.kvcache.device
        )
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = PagedAttentionMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, PagedAttentionMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def validate_attn_backend(backend: str, allow_auto: bool = True) -> str:
    if backend == "auto" and allow_auto:
        return backend
    names = backend.split(",")
    if len(names) not in (1, 2) or any(n not in ("fi", "trtllm") for n in names):
        raise ValueError("Attention backend must be fi, trtllm, or prefill,decode")
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
    kv_cache: MHAKVCache,
    page_table: torch.Tensor,
    page_size: int,
) -> AttentionBackend:
    validate_attn_backend(backend, allow_auto=False)
    names = backend.split(",")
    if len(names) == 2 and names[0] != names[1]:
        return HybridBackend(
            create_attention_backend(names[0], config, kv_cache, page_table, page_size),
            create_attention_backend(names[1], config, kv_cache, page_table, page_size),
        )
    if names[0] == "fi":
        return FlashInferBackend(config, kv_cache, page_table)
    return TRTLLMBackend(config, kv_cache, page_table, page_size)


# These types document the concrete choices; no plugin registry or abstract
# class is needed to dispatch two known kernels.
AttentionBackend = FlashInferBackend | TRTLLMBackend | HybridBackend
AttentionMetadata = FIMetadata | PagedAttentionMetadata
