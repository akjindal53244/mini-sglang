"""One GPU: initialize the model, execute a batch, and sample the next tokens."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import NamedTuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.config import Config
from minisgl.core import Batch, Request, SamplingParams
from minisgl.distributed import broadcast, destroy_distributed, set_parallel_info
from minisgl.graph import GraphRunner
from minisgl.kvcache import MHAKVCache
from minisgl.models import create_model, load_weights
from minisgl.utils import (
    align_ceil,
    compute_capability,
    div_ceil,
    div_even,
    format_memory,
    init_logger,
    torch_dtype,
)


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None


def make_device_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def prepare_sampling_args(batch: Batch, device: torch.device, vocab_size: int) -> BatchSamplingArgs:
    params = [r.sampling_params for r in batch.reqs]
    greedy = [p.is_greedy for p in params]
    if all(greedy):
        return BatchSamplingArgs(temperatures=None)

    MIN_P = MIN_T = 1e-6
    ts = [1.0 if p.is_greedy else max(p.temperature, MIN_T) for p in params]
    top_ks = [p.top_k if p.top_k >= 1 else vocab_size for p in params]
    top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
    temperatures = make_device_tensor(ts, torch.float32, device)
    top_k, top_p = None, None
    if any(k != vocab_size for k in top_ks):
        top_k = make_device_tensor(top_ks, torch.int32, device)
    if any(p < 1.0 for p in top_ps):
        top_p = make_device_tensor(top_ps, torch.float32, device)
    mask = make_device_tensor(greedy, torch.bool, device) if any(greedy) else None
    return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, greedy_mask=mask)


def sample(logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    with torch.cuda.nvtx.range("Sampler"):
        if args.temperatures is None:
            return logits.argmax(dim=-1)
        import flashinfer.sampling as sampling

        probs = sampling.softmax(
            logits.float(), args.temperatures, enable_pdl=compute_capability() >= (9, 0)
        )
        if args.top_k is None and args.top_p is None:
            tokens = sampling.sampling_from_probs(probs)
        elif args.top_p is None:
            tokens = sampling.top_k_sampling_from_probs(probs, args.top_k)
        elif args.top_k is None:
            tokens = sampling.top_p_sampling_from_probs(probs, args.top_p)
        else:
            tokens = sampling.top_k_top_p_sampling_from_probs(probs, args.top_k, args.top_p)
        if args.greedy_mask is not None:
            # A request with temperature=0 must remain exact argmax even when
            # batched with stochastic requests; tiny-temperature sampling can
            # choose randomly between tied maxima.
            tokens = torch.where(args.greedy_mask, logits.argmax(dim=-1), tokens)
        return tokens


logger = init_logger(__name__)


def largest_prefill(config: Config) -> list[int]:
    """Prompt lengths of the largest prefill batch the scheduler can form.

    Activations grow with tokens and logits with requests, so take both limits:
    the whole prefill budget, spread over every request row, with each prompt
    leaving room for one output token.
    """
    num_reqs = min(config.max_running_req, config.max_extend_tokens)
    total = min(config.max_extend_tokens, num_reqs * max(config.max_seq_len - 1, 1))
    base, extra = divmod(total, num_reqs)
    return [base + (i < extra) for i in range(num_reqs)]


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: Config):
        set_parallel_info(config.tp_info, config.world)
        self.config = config = resolve_engine_config(config)

        # One GPU per rank normally. With fewer GPUs than ranks, ranks share them.
        num_gpus = torch.cuda.device_count()
        self.ranks_per_gpu = div_ceil(config.world.size, num_gpus)
        self.device = torch.device(f"cuda:{config.world.rank % num_gpus}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        self.cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()
        logger.info(f"Free memory before loading model: {format_memory(init_free_memory)}")

        # Allocate parameters on the GPU, then copy each checkpoint shard into place.
        with torch.device(self.device), torch_dtype(config.dtype):
            self.model = create_model(config.model_config, pp_info=config.pp_info)
        if config.use_dummy_weight:
            for param in self.model.parameters():
                param.data.normal_()
        else:
            load_weights(self.model, config.model_path, config.model_config, config.pp_info)

        # Page-table rows cover every position the model allows; KV capacity may
        # turn out smaller. An int32 row is aligned to 128 bytes and holds physical
        # token slots, not page numbers. The extra row feeds graph padding.
        aligned_max_seq_len = align_ceil(config.max_seq_len, 32)
        self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len), dtype=torch.int32, device=self.device
        )
        # Size KV last: first run the largest prefill on a small pool and measure it.
        warmup_lengths = largest_prefill(config)
        self.kv_cache = MHAKVCache(
            num_kv_heads=config.model_config.num_kv_heads,
            num_layers=len(self.model.model.layers),  # Only this pipeline stage's layers.
            head_dim=config.model_config.head_dim,
            num_pages=div_ceil(max(warmup_lengths), config.page_size),
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
            config.page_size,
        )
        activation_memory = self._measure_prefill_memory(warmup_lengths)
        self.kv_cache.resize(0)  # Release the warm-up pool before measuring what is left.
        self.num_pages = self._determine_num_pages(init_free_memory, activation_memory, config)
        self.kv_cache.resize(self.num_pages + 1)  # +1 for the dummy page
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)

        post_free_memory = self._sync_get_memory()
        logger.info(f"Free memory after initialization: {format_memory(post_free_memory)}")

        # Graph padding rows decode one token from a page no request owns.
        self.dummy_req = Request(-1, [0], SamplingParams(max_tokens=1), config.max_running_req)
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            max_running_req=config.max_running_req,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: Config) -> torch.distributed.ProcessGroup | None:
        """NCCL moves GPU tensors; a Gloo group moves CPU requests and memory totals."""
        # A single GPU needs no process group, local TCP port, or CPU collective.
        world = config.world
        if world.size == 1:
            return None
        if self.ranks_per_gpu > 1:
            # NCCL rejects two ranks on one GPU ("Duplicate GPU detected") unless
            # they claim to be different hosts. They then talk over loopback TCP:
            # slow, but the same collectives and code path as real multi-GPU runs.
            os.environ["NCCL_HOSTID"] = f"minisgl-rank-{world.rank}"
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        torch.distributed.init_process_group(
            backend="nccl",
            rank=world.rank,
            world_size=world.size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        return torch.distributed.new_group(backend="gloo")

    @torch.inference_mode()
    def _measure_prefill_memory(self, lengths: list[int]) -> int:
        """Run the largest prefill and its sampling once; return the memory they needed."""
        # Stochastic settings with top-k and top-p take the sampler's largest path.
        params = SamplingParams(temperature=1.0, top_k=2, top_p=0.5, max_tokens=1)
        batch = Batch(
            [Request(-1, [0] * n, params, row) for row, n in enumerate(lengths)], "prefill"
        )
        batch.padded_reqs = batch.reqs
        # Every row maps position p to slot p, sharing the small warm-up pool.
        # Requests overwrite each other's KV, but only memory is measured here.
        self.page_table[: len(lengths), : max(lengths)] = torch.arange(max(lengths))
        positions = torch.cat([torch.arange(n, dtype=torch.int32) for n in lengths])
        batch.positions = batch.out_loc = positions.to(self.device)
        batch.input_ids = torch.zeros_like(batch.positions)
        batch.last_token_indices = torch.tensor(lengths).cumsum(0).sub(1).to(self.device)
        self.attn_backend.prepare_metadata(batch)
        args = prepare_sampling_args(batch, self.device, self.config.model_config.vocab_size)
        torch.cuda.synchronize(self.device)
        before = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        logits = self.model(batch, self.attn_backend)
        if logits is not None:  # Only the last pipeline stage samples.
            sample(logits, args)
        torch.cuda.synchronize(self.device)
        peak = torch.tensor(torch.cuda.max_memory_allocated(self.device) - before)
        if self.cpu_group is not None:  # Ranks must agree on one page count.
            torch.distributed.all_reduce(peak, torch.distributed.ReduceOp.MAX, self.cpu_group)
        logger.info(
            f"A {sum(lengths)}-token prefill of {len(lengths)} requests needs "
            f"{format_memory(int(peak))} of activations"
        )
        return int(peak)

    def _determine_num_pages(self, init_free: int, activation_memory: int, config: Config) -> int:
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            # Every stage needs the same page count, so size for the largest stage.
            * div_ceil(config.model_config.num_layers, config.pp_info.size)
        )
        num_pages = config.num_page_override
        if num_pages is None:
            # Weights, page tables, and workspaces are allocated; activations are not.
            # Both measurements are device-wide: share usage as well as budget.
            used_per_rank = (init_free - self._sync_get_memory()) / self.ranks_per_gpu
            budget_per_rank = config.memory_ratio * init_free / self.ranks_per_gpu
            num_pages = int(budget_per_rank - used_per_rank - activation_memory) // cache_per_page

        if num_pages < 1:
            raise ValueError(
                "No memory left for KV cache; increase memory_ratio or use a smaller model"
            )
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(
            f"Allocating {num_tokens} tokens for KV cache, K + V = {format_memory(real_kv_size)}"
        )
        return num_pages

    def _sync_get_memory(self) -> int:
        """The minimum free memory over ranks: all must fit the same page count."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free_memory = torch.tensor(torch.cuda.mem_get_info(self.device)[0])
        if self.cpu_group is not None:
            torch.distributed.all_reduce(
                free_memory, torch.distributed.ReduceOp.MIN, self.cpu_group
            )
        return int(free_memory)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        # logits: [batch.size, vocab], or None on all but the last pipeline stage.
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model(batch, self.attn_backend)

        for req in batch.reqs:
            req.advance_after_forward()

        # One rank samples: the last, the only pipeline stage with logits. Under TP
        # every rank has them, but independent samplers would agree only while their
        # random states stayed identical. Every rank needs the tokens: they feed the
        # next decode, and each rank's scheduler must retire the same work.
        world = self.config.world
        if world.rank == world.size - 1:
            next_tokens_gpu = sample(logits[: batch.size], args).to(torch.int32)
        else:
            next_tokens_gpu = torch.empty(batch.size, dtype=torch.int32, device=self.device)
        if world.size > 1:
            broadcast(next_tokens_gpu, src=world.size - 1)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group()
        destroy_distributed()


def resolve_engine_config(config: Config) -> Config:
    """Resolve hardware defaults without mutating the frozen caller configuration."""
    backend, page_size = config.attention_backend, config.page_size
    if backend == "auto":
        # TRT-LLM's prefill kernels target SM10x; SM12x (e.g. RTX PRO 6000) has none.
        is_sm10x = (10, 0) <= compute_capability() < (12, 0)
        backend = "trtllm" if is_sm10x else "fi"
        logger.info(f"Auto-selected attention backend: {backend}")

    if "trtllm" in backend and page_size not in [16, 32, 64]:
        page_size = 64
        logger.warning("Page size is overridden to 64 for TRTLLM backend")

    return replace(config, attention_backend=backend, page_size=page_size)
