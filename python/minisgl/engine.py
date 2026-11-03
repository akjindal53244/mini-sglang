"""One GPU: initialize the model, execute a batch, and sample the next tokens."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import NamedTuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.config import Config
from minisgl.core import Batch, Request
from minisgl.distributed import (
    broadcast,
    destroy_distributed,
    enable_pynccl_distributed,
    set_tp_info,
)
from minisgl.graph import GraphRunner
from minisgl.kvcache import MHAKVCache
from minisgl.models import create_model, load_weight
from minisgl.utils import (
    align_ceil,
    div_ceil,
    div_even,
    format_memory,
    get_free_memory,
    init_logger,
    is_compute_capability_at_least,
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
            logits.float(), args.temperatures, enable_pdl=is_compute_capability_at_least(9, 0)
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


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: Config):
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
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

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[0]
        logger.info(f"Free memory before loading model: {format_memory(init_free_memory)}")

        # Materialize checkpoint tensors without ever allocating random GPU weights.
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(
                config.model_config, rope_device=self.device, pp_info=config.pp_info
            )
        weights = self._load_weight_state_dict(config)
        if config.model_config.tie_word_embeddings:
            # assign=True retains tensor storage: both parameters share one
            # checkpoint allocation even when the Hub omits lm_head.weight.
            # Under PP the head may live on a stage that has no embedding.
            embedding = weights.get("model.embed_tokens.weight")
            if self.model.lm_head is not None and embedding is not None:
                weights["lm_head.weight"] = embedding
            if self.model.model.embed_tokens is None:
                weights.pop("model.embed_tokens.weight", None)
        self.model.load_state_dict(weights, assign=True)
        del weights

        # Reserve physical KV storage after measuring the loaded model's footprint.
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.kv_cache = MHAKVCache(
            num_kv_heads=config.model_config.num_kv_heads,
            num_layers=len(self.model.model.layers),  # Only this pipeline stage's layers.
            head_dim=config.model_config.head_dim,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # The extra row and page absorb graph padding. Each int32 row is aligned
        # to 128 bytes and holds physical token offsets, rather than page numbers.
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = align_ceil(self.max_seq_len, 32)
        self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
            config.page_size,
        )

        post_free_memory = self._sync_get_memory()[0]
        logger.info(f"Free memory after initialization: {format_memory(post_free_memory)}")

        self.dummy_req = Request(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_running_req=config.max_running_req,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: Config) -> torch.distributed.ProcessGroup | None:
        # A single GPU needs no process group, local TCP port, or CPU collective.
        world = config.world
        if world.size == 1:
            return None
        if self.ranks_per_gpu > 1:
            # NCCL rejects two ranks on one GPU ("Duplicate GPU detected") unless
            # they claim to be different hosts. They then talk over loopback TCP:
            # slow, but the same collectives and code path as real multi-GPU TP.
            os.environ["NCCL_HOSTID"] = f"minisgl-rank-{world.rank}"
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        if config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=world.rank,
                world_size=world.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_extend_tokens * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(world, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=world.rank,
                world_size=world.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: Config) -> dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        weights = load_weight(config.model_path, self.device, config.pp_info)
        return {k: v.to(self.dtype) for k, v in weights}

    def _determine_num_pages(self, old_free_memory: int, config: Config) -> int:
        new_free_memory = self._sync_get_memory()[0]
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
            model_memory = old_free_memory - new_free_memory
            budget = config.memory_ratio * old_free_memory / self.ranks_per_gpu
            available_memory = int(budget) - model_memory
            num_pages = available_memory // cache_per_page

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

    def _sync_get_memory(self) -> tuple[int, int]:
        """Use the minimum free memory: all TP ranks must fit the same page count."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        if self.tp_cpu_group is None:
            return free_memory, free_memory
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {format_memory(min_free_memory)}, max {format_memory(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        # logits: [batch.size, vocab], or None on all but the last pipeline stage.
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model(batch, self.attn_backend)

        for req in batch.reqs:
            req.advance_after_forward()

        # Only the last pipeline stage has logits. Every rank needs the tokens: they
        # feed the next decode, and each rank's scheduler must retire the same work.
        pp = self.config.pp_info
        if pp.rank == pp.size - 1:
            next_tokens_gpu = sample(logits[: batch.size], args).to(torch.int32)
        else:
            next_tokens_gpu = torch.empty(batch.size, dtype=torch.int32, device=self.device)
        if pp.size > 1:
            broadcast(next_tokens_gpu, src=pp.size - 1)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        if self.tp_cpu_group is not None:
            torch.distributed.destroy_process_group()
        destroy_distributed()


def resolve_engine_config(config: Config) -> Config:
    """Resolve hardware defaults without mutating the frozen caller configuration."""
    backend, page_size = config.attention_backend, config.page_size
    if backend == "auto":
        # TRT-LLM's prefill kernels target SM10x; SM12x (e.g. RTX PRO 6000) has none.
        is_sm10x = is_compute_capability_at_least(10, 0) and not (
            is_compute_capability_at_least(12, 0)
        )
        backend = "trtllm" if is_sm10x else "fi"
        logger.info(f"Auto-selected attention backend: {backend}")

    if "trtllm" in backend and page_size not in [16, 32, 64]:
        page_size = 64
        logger.warning("Page size is overridden to 64 for TRTLLM backend")

    return replace(config, attention_backend=backend, page_size=page_size)
