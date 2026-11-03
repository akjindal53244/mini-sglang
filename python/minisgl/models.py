"""Llama-like decoders and their checkpoint loader.

Read RotaryAttention -> DecoderLayer -> DecoderModel -> DecoderForCausalLM.
T is the number of new tokens across the batch; decode has one per request.
The model receives its batch and attention backend explicitly, as in a normal
PyTorch forward pass. Scheduling and KV allocation stay outside the model.

Pipeline parallelism cuts the decoder by depth: a stage holds only some layers
and their KV, then hands hidden states to the next stage. Tensor parallelism cuts
every layer by width instead; see the layers in layers.py.
"""

from __future__ import annotations

import glob
import os
import re
from functools import partial
from typing import TYPE_CHECKING

import safetensors
import torch
from minisgl.config import ModelConfig
from minisgl.core import Batch
from minisgl.distributed import (
    DistributedInfo,
    get_tp_info,
    get_world_info,
    recv,
    send,
    stage_layers,
)
from minisgl.layers import (
    MoELayer,
    ParallelLMHead,
    ResidualRMSNorm,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    VocabParallelEmbedding,
    copy_into,
    load_column_shard,
    load_whole,
    silu_and_mul,
)
from minisgl.utils import div_even, download_hf_weight
from torch import nn
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import AttentionBackend


class GatedFeedForward(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}")
        local_size = div_even(config.intermediate_size, get_tp_info().size)
        # Packed [local gate, local up]: the loader shards each before placing it.
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * local_size, bias=False)
        self.gate_up_proj.weight.weight_loader = partial(load_column_shard, local_size=local_size)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size)

    @torch.cuda.nvtx.range("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(x)))


class MoEFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = MoELayer(
            config.num_experts,
            config.num_experts_per_tok,
            config.hidden_size,
            config.moe_intermediate_size,
            config.norm_topk_prob,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.experts(x, self.gate(x))


class RotaryAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int, attention_bias: bool, qk_norm: bool):
        super().__init__()
        tp_size = get_tp_info().size
        self.num_q_heads = div_even(config.num_qo_heads, tp_size)
        # TP may exceed the number of KV heads: adjacent ranks then replicate a head.
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.head_dim, self.layer_id = config.head_dim, layer_id
        self.total_kv_heads = config.num_kv_heads
        q_size, kv_size = self.num_q_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=attention_bias)
        for param in self.qkv_proj.parameters():
            param.weight_loader = self._load_qkv
        self.o_proj = RowParallelLinear(config.num_qo_heads * self.head_dim, config.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if qk_norm else None

    def _load_qkv(self, param: nn.Parameter, value: torch.Tensor, shard: str) -> int:
        """Place this rank's heads of q_proj, k_proj, or v_proj in the packed [Q | K | V]."""
        tp = get_tp_info()
        if shard == "q":
            return copy_into(param.data, value.chunk(tp.size)[tp.rank])
        if self.total_kv_heads < tp.size:  # Adjacent ranks replicate one KV head.
            head = tp.rank * self.total_kv_heads // tp.size
            value = value[head * self.head_dim : (head + 1) * self.head_dim]
        else:
            value = value.chunk(tp.size)[tp.rank]
        q_size, kv_size = self.num_q_heads * self.head_dim, self.num_kv_heads * self.head_dim
        return copy_into(param.data, value, start=q_size + (kv_size if shard == "v" else 0))

    @torch.cuda.nvtx.range("Attention")
    def forward(
        self, x: torch.Tensor, batch: Batch, backend: AttentionBackend, rope: RotaryEmbedding
    ) -> torch.Tensor:
        q_size, kv_size = self.num_q_heads * self.head_dim, self.num_kv_heads * self.head_dim
        # [T, H] -> [T, Q + K + V] -> strided views of [T, Q], [T, K], [T, V].
        # K and V may be smaller than Q: that is grouped-query attention. Q/K norm is per head.
        q, k, v = self.qkv_proj(x).split([q_size, kv_size, kv_size], dim=-1)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_q_heads, self.head_dim))
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        q, k = rope.forward(batch.positions, q, k)
        q = q.view(-1, self.num_q_heads, self.head_dim)  # [T, q_heads, head_dim]
        # The backend writes K/V at batch.out_loc, then attends over each prefix.
        output = backend.forward(q, k, v, self.layer_id, batch)  # [T, q_heads, head_dim]
        return self.o_proj(output.view(-1, q_size))  # [T, H]


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        attention_bias: bool,
        qk_norm: bool,
        use_moe: bool,
    ):
        super().__init__()
        self.self_attn = RotaryAttention(config, layer_id, attention_bias, qk_norm)
        self.mlp = MoEFeedForward(config) if use_moe else GatedFeedForward(config)
        self.input_layernorm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layer_id = layer_id

    def forward(self, x, residual, batch, backend, rope):
        with torch.cuda.nvtx.range(f"Layer_{self.layer_id}"):
            # The residual travels separately so addition + the next RMSNorm
            # share one kernel. ResidualRMSNorm documents the in-place updates.
            x, residual = self.input_layernorm(x, residual)
            x = self.self_attn(x, batch, backend, rope)
            x, residual = self.post_attention_layernorm(x, residual)
            return self.mlp(x), residual


class DecoderModel(nn.Module):
    def __init__(self, config: ModelConfig, pp: DistributedInfo, **options):
        super().__init__()
        self.pp = pp
        first, last = pp.rank == 0, pp.rank == pp.size - 1
        num_local_layers = len(stage_layers(config.num_layers, pp))
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size) if first else None
        )
        # Local indices: each stage's KV pool holds only these layers.
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i, **options) for i in range(num_local_layers)]
        )
        self.norm = ResidualRMSNorm(config.hidden_size, config.rms_norm_eps) if last else None

    def forward(self, batch: Batch, backend: AttentionBackend, rope: RotaryEmbedding):
        rank, last = self.pp.rank, self.pp.size - 1
        if rank == 0:
            x, residual = self.embed_tokens(batch.input_ids), None
        else:
            # Every rank runs the same scheduler, so T needs no shape exchange.
            weight = self.layers[0].input_layernorm.weight
            x = weight.new_empty(len(batch.positions), len(weight))  # [T, H]
            residual = weight.new_empty(len(batch.positions), len(weight))  # [T, H]
            recv(x, rank - 1)
            recv(residual, rank - 1)
        for layer in self.layers:
            x, residual = layer(x, residual, batch, backend, rope)
        if rank < last:
            send(x, rank + 1)
            send(residual, rank + 1)
            return None
        return self.norm(x, residual)[0]


class DecoderForCausalLM(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        pp: DistributedInfo,
        attention_bias=False,
        qk_norm=False,
        use_moe=False,
    ):
        super().__init__()
        self.model = DecoderModel(
            config, pp, attention_bias=attention_bias, qk_norm=qk_norm, use_moe=use_moe
        )
        is_last = pp.rank == pp.size - 1
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size) if is_last else None
        embedding = self.model.embed_tokens
        if config.tie_word_embeddings and self.lm_head is not None and embedding is not None:
            self.lm_head.weight = embedding.weight  # One parameter, loaded once.
        # Built inside the model's torch.device context and shared by every layer.
        self.rope = RotaryEmbedding(config)

    def forward(self, batch: Batch, backend: AttentionBackend) -> torch.Tensor | None:
        x = self.model(batch, backend, self.rope)
        if x is None:  # A later pipeline stage produces the logits.
            return None
        if batch.is_prefill:
            # Prefill produces T hidden states, but only the last token of each
            # request needs a vocabulary projection. Intermediate chunks discard it.
            x = x[batch.last_token_indices].contiguous()
        return self.lm_head(x)


# Checkpoint projections that this decoder packs into one parameter.
_PACKED = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<proj>gate|up|down)_proj\.weight$"
)
_LAYER_PATTERN = re.compile(r"^model\.layers\.(?P<idx>\d+)\.(?P<name>.+)$")


def _stage_key(key: str, config: ModelConfig, pp: DistributedInfo) -> str | None:
    """Map a checkpoint key to this pipeline stage, or None if this stage skips it."""
    if (match := _LAYER_PATTERN.match(key)) is not None:
        layers = stage_layers(config.num_layers, pp)
        index = int(match.group("idx"))
        return (
            f"model.layers.{index - layers.start}.{match.group('name')}"
            if index in layers
            else None
        )
    first, last = pp.rank == 0, pp.rank == pp.size - 1
    tied = config.tie_word_embeddings
    if "embed_tokens" in key:
        if first:  # With one stage, a tied head shares this very parameter.
            return key
        return "lm_head.weight" if last and tied else None
    if key.startswith("lm_head") and tied:
        return None  # Tied: the head was loaded from embed_tokens.
    return key if last else None  # The final norm and lm_head.


def _packed_target(key: str) -> tuple[str, object]:
    """Checkpoint key -> (parameter name, shard passed to its weight_loader)."""
    if (match := _EXPERT_PATTERN.match(key)) is not None:
        expert, proj = int(match.group("idx")), match.group("proj")
        if proj == "down":
            return f"{match.group('prefix')}.down_proj", (expert, None)
        return f"{match.group('prefix')}.gate_up_proj", (expert, 0 if proj == "gate" else 1)
    for part, (packed, shard) in _PACKED.items():
        if f".{part}." in key:
            return key.replace(f".{part}.", f".{packed}."), shard
    return key, None


def load_weights(model: nn.Module, model_path: str, config: ModelConfig, pp: DistributedInfo):
    """Copy each checkpoint tensor's shard into its parameter, then check none was missed."""
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    # Mistral's consolidated.safetensors repeats the weights under other names.
    files = [f for f in files if not os.path.basename(f).startswith("consolidated")]
    params = dict(model.named_parameters())
    written, seen = dict.fromkeys(params, 0), set()
    device = str(next(model.parameters()).device)
    for file in tqdm(files, desc="Loading weights", disable=not get_world_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=device) as f:
            for name in f.keys():
                # Multimodal checkpoints: keep the language model, skip vision weights.
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                key = _stage_key(name.removeprefix("language_model."), config, pp)
                if key is None:
                    continue
                target, shard = _packed_target(key)
                if target not in params:
                    raise KeyError(f"Checkpoint tensor {name} has no parameter {target}")
                # A repeated slice could stand in for a missing one of equal size.
                if (target, shard) in seen:
                    raise ValueError(f"Checkpoint tensor {name} is loaded twice")
                seen.add((target, shard))
                loader = getattr(params[target], "weight_loader", load_whole)
                written[target] += loader(params[target], f.get_tensor(name), shard)
    missing = [name for name, param in params.items() if written[name] != param.numel()]
    if missing:
        raise ValueError(f"Checkpoint did not fill {len(missing)} parameters, e.g. {missing[:3]}")


# All supported families share the same pre-norm decoder. Keep the actual
# differences visible here instead of scattering identical model classes.
_ARCHITECTURES = {
    "LlamaForCausalLM": {},
    "Qwen2ForCausalLM": {"attention_bias": True},
    "Qwen3ForCausalLM": {"qk_norm": True},
    "Qwen3MoeForCausalLM": {"qk_norm": True, "use_moe": True},
    "MistralForCausalLM": {},
    "Mistral3ForConditionalGeneration": {},  # text decoder only
}


def create_model(
    model_config: ModelConfig, pp_info: DistributedInfo = DistributedInfo(0, 1)
) -> DecoderForCausalLM:
    architecture = model_config.architecture
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    options = _ARCHITECTURES[architecture]
    model = DecoderForCausalLM(model_config, pp=pp_info, **options)
    return model.eval().requires_grad_(False)
