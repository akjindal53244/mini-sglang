"""Capture and replay decode batches with persistent CUDA graph buffers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.core import Batch, Request
from minisgl.distributed import get_tp_info
from minisgl.utils import format_memory, get_free_memory, init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import AttentionBackend
    from minisgl.models import DecoderForCausalLM

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: list[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
    max_running_req: int,
) -> list[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    if cuda_graph_max_bs is None:
        cuda_graph_max_bs = 256 if free_memory > 80 * (1 << 30) else 160
    # No batch exceeds max_running_req; larger graphs would only slow startup.
    cuda_graph_max_bs = min(cuda_graph_max_bs, max_running_req)

    if cuda_graph_max_bs < 1:
        return []

    sizes = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in sizes if bs <= cuda_graph_max_bs]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: DecoderForCausalLM,
        attn_backend: AttentionBackend,
        cuda_graph_bs: list[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_running_req: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Request,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
            max_running_req=max_running_req,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: DecoderForCausalLM):
        self.graph_map: dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            logger.info("CUDA graphs disabled")
            return

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info(f"Free GPU memory before capturing CUDA graphs: {format_memory(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {format_memory(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)

            def run() -> None:
                logits = model(batch, self.attn_backend)
                if logits is not None:  # Only the last pipeline stage has logits.
                    self.buffer.logits[:bs] = logits

            run()  # A warmup outside the graph performs one-time kernel setup.
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                run()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info(f"Free GPU memory after capturing CUDA graphs: {format_memory(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

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
