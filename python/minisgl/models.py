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
import re
from typing import TYPE_CHECKING, Iterator

import safetensors
import torch
from minisgl.config import ModelConfig
from minisgl.core import Batch
from minisgl.distributed import DistributedInfo, get_tp_info, recv, send, stage_layers
from minisgl.layers import (
    MoELayer,
    ParallelLMHead,
    ResidualRMSNorm,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    create_rope,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import AttentionBackend
    from minisgl.layers import RotaryEmbedding


class GatedFeedForward(nn.Module):
    """SwiGLU/GEGLU: down(activation(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        local_size = div_even(config.intermediate_size, get_tp_info().size)
        # Shard gate and up separately when loading, then pack [local gate, local up].
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * local_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size)
        self.activation = {"silu": silu_and_mul, "gelu": gelu_and_mul}[config.hidden_act]

    @torch.cuda.nvtx.range("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_up_proj(x)))


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
        q_size, kv_size = self.num_q_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=attention_bias)
        self.o_proj = RowParallelLinear(config.num_qo_heads * self.head_dim, config.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if qk_norm else None

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
        rope_device="cpu",
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
        self.rope = create_rope(config, rope_device)

    def forward(self, batch: Batch, backend: AttentionBackend) -> torch.Tensor | None:
        x = self.model(batch, backend, self.rope)
        if x is None:  # A later pipeline stage produces the logits.
            return None
        if batch.is_prefill:
            # Prefill produces T hidden states, but only the last token of each
            # request needs a vocabulary projection. Intermediate chunks discard it.
            x = x[batch.last_token_indices].contiguous()
        return self.lm_head(x)


_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGED_PROJECTIONS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_LAYER_PATTERN = re.compile(r"^model\.layers\.(?P<idx>\d+)\.(?P<name>.+)$")


def _stage_key(key: str, config: ModelConfig, pp: DistributedInfo) -> str | None:
    """Map a checkpoint key to this pipeline stage, or None if another stage owns it."""
    if (match := _LAYER_PATTERN.match(key)) is not None:
        layers = stage_layers(config.num_layers, pp)
        index = int(match.group("idx"))
        return (
            f"model.layers.{index - layers.start}.{match.group('name')}"
            if index in layers
            else None
        )
    first, last = pp.rank == 0, pp.rank == pp.size - 1
    if "embed_tokens" in key:  # A tied head on the last stage needs these weights too.
        return key if first or (last and config.tie_word_embeddings) else None
    return key if last else None  # The final norm and lm_head.


def _shard_tensor(key: str, value: torch.Tensor, rank: int, world_size: int, num_kv_heads: int):
    """Extract this rank's shard, copying slices to release the full checkpoint tensor."""
    if any(sub in key for sub in _SPLIT_DIM_0):
        is_kv_proj = any(sub in key for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads < world_size:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = rank * num_kv_heads // world_size
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(world_size, dim=0)[rank].clone()
    elif any(sub in key for sub in _SPLIT_DIM_1):
        return value.chunk(world_size, dim=1)[rank].clone()
    elif "lm_head" in key or "embed_tokens" in key:
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, world_size)
        vocab_start_idx = rank * num_embeddings_per_partition
        vocab_end_idx = min((rank + 1) * num_embeddings_per_partition, num_embeddings)
        shard = value[vocab_start_idx:vocab_end_idx, :].clone()
        # Every rank's embedding/head has the same local shape, even when the
        # vocabulary does not divide TP evenly. Padded logits are trimmed later.
        padding = num_embeddings_per_partition - len(shard)
        return F.pad(shard, (0, 0, 0, padding)) if padding else shard
    else:
        return value


def _get_merge_info(key: str):
    for merged, parts in _MERGED_PROJECTIONS.items():
        for part in parts:
            if f".{part}." in key:
                return key.replace(f".{part}.", f".{merged}."), part, parts
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(
    model_path: str, device: torch.device, pp: DistributedInfo
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield checkpoint tensors on device, sharded and packed for fused projections.

    Completed groups are yielded immediately; incomplete projections/experts
    stay in buffers until their remaining checkpoint tensors have been read.
    """

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Shard each projection BEFORE packing QKV or gate/up. Splitting a packed
    # tensor evenly would assign ranks the wrong heads/features.
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}
    expert_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                key = _stage_key(name.removeprefix("language_model."), config, pp)
                if key is None:
                    continue
                raw = f.get_tensor(name)
                name = key
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"


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
    model_config: ModelConfig, rope_device="cpu", pp_info: DistributedInfo = DistributedInfo(0, 1)
) -> DecoderForCausalLM:
    architecture = model_config.architecture
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    options = _ARCHITECTURES[architecture]
    model = DecoderForCausalLM(model_config, rope_device, pp=pp_info, **options)
    return model.eval().requires_grad_(False)
