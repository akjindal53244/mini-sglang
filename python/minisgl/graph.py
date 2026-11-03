"""Capture and replay decode batches with persistent CUDA graph buffers."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch
from minisgl.core import Batch, Request
from minisgl.distributed import get_world_info
from minisgl.utils import format_memory, init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import AttentionBackend
    from minisgl.models import DecoderForCausalLM

logger = init_logger(__name__)


def cuda_graph_sizes(max_bs: int) -> list[int]:
    """1, 2, 4, then every multiple of 8: a batch pads to the next captured size."""
    return [bs for bs in [1, 2, 4, *range(8, max_bs + 1, 8)] if bs <= max_bs]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: DecoderForCausalLM,
        attn_backend: AttentionBackend,
        cuda_graph_bs: list[int] | None,
        max_running_req: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Request,
    ) -> None:
        if cuda_graph_bs is None:
            # No batch exceeds max_running_req; larger graphs would only slow startup.
            cuda_graph_bs = cuda_graph_sizes(min(256, max_running_req))
        self.attn_backend = attn_backend
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.max_graph_bs = max(cuda_graph_bs, default=0)
        self.dummy_req = dummy_req
        self.graph_map: dict[int, torch.cuda.CUDAGraph] = {}
        self._capture_graphs(stream, device, model, max_seq_len, vocab_size)

    def _capture_graphs(self, stream, device, model: DecoderForCausalLM, max_seq_len, vocab_size):
        if self.max_graph_bs == 0:
            logger.info("CUDA graphs disabled")
            return
        # A graph replays fixed addresses. Each batch is copied into the first rows
        # of these [max_graph_bs] inputs, and every graph writes the same logits rows.
        size = self.max_graph_bs
        self.input_ids, self.positions, self.out_loc = (
            torch.zeros(size, dtype=torch.int32, device=device) for _ in range(3)
        )
        self.logits = None  # Only the last pipeline stage has a vocabulary head.
        if model.lm_head is not None:
            dtype = model.lm_head.weight.dtype
            self.logits = torch.empty(size, vocab_size, dtype=dtype, device=device)

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        logger.info(f"Capturing CUDA graphs for batch sizes {self.graph_bs_list}")
        free_memory = torch.cuda.mem_get_info(device)[0]

        sizes = sorted(self.graph_bs_list, reverse=True)  # The largest first sizes the pool.
        pool = None
        for bs in tqdm(sizes, desc="Capturing graphs", disable=not get_world_info().is_primary()):
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            batch.input_ids, batch.positions = self.input_ids[:bs], self.positions[:bs]
            batch.out_loc = self.out_loc[:bs]

            def run() -> None:
                logits = model(batch, self.attn_backend)
                if logits is not None:
                    self.logits[:bs] = logits

            run()  # A warmup outside the graph performs one-time kernel setup.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                run()
            if pool is None:
                pool = graph.pool()  # Smaller graphs reuse the largest graph's memory.
            self.graph_map[bs] = graph

        used = free_memory - torch.cuda.mem_get_info(device)[0]
        logger.info(f"CUDA graphs use {format_memory(used)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor | None:
        """Return logits [batch.size, vocab], or None on all but the last pipeline stage."""
        assert self.can_use_cuda_graph(batch)
        size = batch.padded_size
        self.input_ids[:size] = batch.input_ids
        self.positions[:size] = batch.positions
        self.out_loc[:size] = batch.out_loc
        self.attn_backend.prepare_for_replay(batch)
        self.graph_map[size].replay()
        return None if self.logits is None else self.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # Captured graphs may hold NCCL resources; release them before the communicator.
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
