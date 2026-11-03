"""One configuration for the model, scheduler, and local serving processes."""

from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config
from transformers import PretrainedConfig


def env_bool(name: str, default: bool | None = False) -> bool | None:
    value = os.getenv("MINISGL_" + name)
    return default if value is None else value.lower() in ("1", "true", "yes")


def env_bytes(name: str, default: int) -> int:
    value = os.getenv("MINISGL_" + name)
    if value is None:
        return default
    value = value.strip().upper().removesuffix("B")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(value[:-1]) * units[value[-1]]) if value[-1:] in units else int(value)


FLASHINFER_USE_TENSOR_CORES = env_bool("FLASHINFER_USE_TENSOR_CORES", None)
PYNCCL_MAX_BUFFER_SIZE = env_bytes("PYNCCL_MAX_BUFFER_SIZE", 1024**3)


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    max_position: int
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    architecture: str

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        num_experts = (
            getattr(config, "num_local_experts", None) or getattr(config, "num_experts", 0) or 0
        )
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", None) or ["LlamaForCausalLM"]

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and rope_scaling is not None:
            rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            raise ValueError("Model config must specify rope_theta directly or in rope_scaling")

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            max_position=config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            architecture=architectures[0],
        )


@dataclass(frozen=True)
class Config:
    model_path: str
    dtype: torch.dtype = torch.bfloat16
    tp_info: DistributedInfo = field(default_factory=lambda: DistributedInfo(0, 1))
    pp_info: DistributedInfo = field(default_factory=lambda: DistributedInfo(0, 1))

    # GPU execution. A page is the allocator's unit; attention may expose a
    # different page layout over the same physical K/V storage.
    attention_backend: str = "auto"
    page_size: int = 1
    memory_ratio: float = 0.9
    max_running_req: int = 256
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    overlap: bool = not env_bool("DISABLE_OVERLAP_SCHEDULING")
    cuda_graph_bs: list[int] | None = None
    cuda_graph_max_bs: int | None = None  # 0 disables graphs; None selects by memory
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    distributed_timeout: float = 60.0

    # Serving. The offline LLM API uses no sockets or worker subprocesses.
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    shell: bool = False
    _ipc_id: str = field(default_factory=lambda: f"{os.getpid()}-{uuid.uuid4().hex[:8]}")

    def __post_init__(self):
        for name in ("page_size", "max_running_req", "max_extend_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_seq_len_override", "num_page_override"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.memory_ratio <= 1:
            raise ValueError("memory_ratio must be in (0, 1]")
        if self.cache_type not in ("radix", "naive"):
            raise ValueError("cache_type must be radix or naive")
        if self.dtype not in (torch.bfloat16, torch.float16):
            # FlashInfer's RoPE and PyNCCL only dispatch on 16-bit floats.
            raise ValueError("dtype must be bfloat16 or float16")
        if self.tp_info.size > 1 and self.pp_info.size > 1:
            raise ValueError("TP and PP are alternatives here: use --tp or --pp, not both")
        if self.pp_info.size > 1 and not self.use_pynccl:
            raise ValueError("PP needs PyNCCL: only it sends on the CUDA stream graphs capture")

    @property
    def world(self) -> DistributedInfo:
        """Every rank in this job. TP and PP are alternatives, so one of them is trivial."""
        return self.pp_info if self.pp_info.size > 1 else self.tp_info

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self):
        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        # RoPE is indexed by position with no GPU bounds check: stay inside its table.
        limit = self.model_config.max_position
        return min(self.max_seq_len_override or limit, limit)

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"

    def ipc(self, channel: str) -> str:
        # dataclasses.replace keeps this ID when creating each TP rank's config.
        return f"ipc:///tmp/minisgl-{self._ipc_id}-{channel}"


def parse_args(args: list[str]) -> Config:
    from minisgl.attention import validate_attn_backend

    parser = argparse.ArgumentParser(description="Mini-SGLang inference", allow_abbrev=False)
    parser.add_argument("--model", "--model-path", dest="model_path", required=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    parser.add_argument("--tp", "--tp-size", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pp", "--pp-size", "--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--host", dest="server_host", default=Config.server_host)
    parser.add_argument("--port", dest="server_port", type=int, default=Config.server_port)
    parser.add_argument("--shell", "--shell-mode", action="store_true")
    parser.add_argument(
        "--max-running-requests", dest="max_running_req", type=int, default=Config.max_running_req
    )
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        dest="max_extend_tokens",
        type=int,
        default=Config.max_extend_tokens,
        help="Tokens per prefill batch",
    )
    parser.add_argument("--max-seq-len-override", type=int, help="Cap prompt + generated tokens")
    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        help="Use a fixed-size KV pool instead of a memory ratio",
    )
    parser.add_argument("--page-size", type=int, default=Config.page_size)
    parser.add_argument("--memory-ratio", type=float, default=Config.memory_ratio)
    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        help="Largest captured decode batch; 0 disables graphs",
    )
    parser.add_argument(
        "--disable-overlap",
        dest="overlap",
        action="store_false",
        default=Config.overlap,
        help="Process each result before the next forward",
    )
    parser.add_argument("--cache-type", "--cache", choices=["radix", "naive"], default="radix")
    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default="auto",
        help="fi, trtllm, or prefill,decode (e.g. fi,trtllm)",
    )
    parser.add_argument("--dummy-weight", dest="use_dummy_weight", action="store_true")
    parser.add_argument("--disable-pynccl", dest="use_pynccl", action="store_false")
    parser.add_argument(
        "--model-source", choices=["huggingface", "modelscope"], default="huggingface"
    )
    options = vars(parser.parse_args(args))
    options["model_path"] = os.path.expanduser(options["model_path"])
    if options.pop("model_source") == "modelscope" and not os.path.isdir(options["model_path"]):
        try:
            from modelscope import snapshot_download
        except ImportError as error:
            raise ValueError(
                "Install minisgl[modelscope] to use --model-source modelscope"
            ) from error
        ignore = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"] if options["use_dummy_weight"] else []
        options["model_path"] = snapshot_download(options["model_path"], ignore_patterns=ignore)
    dtype = options["dtype"]
    if dtype == "auto":
        dtype = cached_load_hf_config(options["model_path"]).dtype
        dtype = torch.float16 if dtype in (None, torch.float32, "float32") else dtype
    options["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    options["tp_info"] = DistributedInfo(0, options.pop("tp"))
    options["pp_info"] = DistributedInfo(0, options.pop("pp"))
    if options["shell"]:
        options.update(cuda_graph_max_bs=1, max_running_req=1)
    return Config(**options)
