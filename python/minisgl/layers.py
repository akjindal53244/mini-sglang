"""GPU building blocks: RoPE, fused RMSNorm, and tensor-parallel layers.

Model weights use ordinary nn.Module/Parameter objects. A sharded or packed
parameter carries a ``weight_loader(param, tensor, shard)`` that copies its part
of a checkpoint tensor into place and returns the number of elements written.
The layer that knows a layout is the one that loads it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from minisgl.distributed import all_gather, all_reduce, get_tp_info
from minisgl.utils import div_ceil, div_even
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from minisgl.config import ModelConfig


def copy_into(dest: torch.Tensor, value: torch.Tensor, start: int = 0, dim: int = 0) -> int:
    """Copy value into dest[start : start + len] along dim; return elements written."""
    dest = dest.narrow(dim, start, value.shape[dim])
    assert dest.shape == value.shape, (
        f"checkpoint shape {tuple(value.shape)} != {tuple(dest.shape)}"
    )
    dest.copy_(value)
    return dest.numel()


def load_whole(param: nn.Parameter, value: torch.Tensor, shard=None) -> int:
    return copy_into(param.data, value)


def load_column_shard(param: nn.Parameter, value: torch.Tensor, shard: int, local_size: int) -> int:
    """gate_up: each rank keeps rows tp.rank of gate and of up, packed [gate, up]."""
    tp = get_tp_info()
    return copy_into(param.data, value.chunk(tp.size)[tp.rank], start=shard * local_size)


def load_row_shard(param: nn.Parameter, value: torch.Tensor, shard=None) -> int:
    tp = get_tp_info()
    return copy_into(param.data, value.chunk(tp.size, dim=1)[tp.rank], dim=1)


def vocab_range(vocab_size: int) -> tuple[int, int, int]:
    """(local rows, first owned ID, end of owned IDs). The last shard may be padded."""
    tp = get_tp_info()
    local_size = div_ceil(vocab_size, tp.size)
    start = local_size * tp.rank
    return local_size, start, min(start + local_size, vocab_size)


def load_vocab_shard(param: nn.Parameter, value: torch.Tensor, shard=None) -> int:
    _, start, end = vocab_range(len(value))
    param.data[end - start :].zero_()  # Padding rows: zero embeddings, trimmed logits.
    return copy_into(param.data, value[start:end]) + param.data[end - start :].numel()


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def rope_frequencies(config: ModelConfig) -> tuple[torch.Tensor, float]:
    """Inverse frequencies [head_dim // 2] and the Q/K amplitude, as Transformers defines them."""
    dim, base = config.head_dim, config.rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    scaling = config.rope_scaling or {"rope_type": "default"}
    match scaling["rope_type"]:
        case "default":
            return inv_freq, 1.0

        case "llama3":  # Slow down only the low frequencies, with a smooth transition band.
            factor, old_context = scaling["factor"], scaling["original_max_position_embeddings"]
            low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
            wave_len = 2 * math.pi / inv_freq
            if low == high:  # No transition band.
                return torch.where(wave_len < old_context / high, inv_freq, inv_freq / factor), 1.0
            smooth = torch.clamp((old_context / wave_len - low) / (high - low), 0, 1)
            return ((1 - smooth) / factor + smooth) * inv_freq, 1.0

        case "yarn":  # Interpolate low frequencies, keep high ones, and scale attention.
            factor = scaling["factor"]
            old_context = scaling.get("original_max_position_embeddings") or config.max_position

            def mscale(multiplier: float = 1.0) -> float:
                return 1.0 if factor <= 1 else 1.0 + 0.1 * multiplier * math.log(factor)

            amplitude = scaling.get("attention_factor")
            if amplitude is None:
                m, m_all = scaling.get("mscale"), scaling.get("mscale_all_dim")
                amplitude = mscale(m) / mscale(m_all) if m and m_all else mscale()

            def correction_dim(rotations: float) -> float:  # Dimension turning this many times.
                return (
                    dim * math.log(old_context / (rotations * 2 * math.pi)) / (2 * math.log(base))
                )

            low = correction_dim(scaling.get("beta_fast") or 32.0)
            high = correction_dim(scaling.get("beta_slow") or 1.0)
            if scaling.get("truncate", True):
                low, high = math.floor(low), math.ceil(high)
            low, high = max(low, 0), min(high, dim - 1)
            ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (
                high - low if high != low else 1e-3
            )
            ramp = torch.clamp(ramp, 0, 1)
            return (inv_freq / factor) * ramp + inv_freq * (1 - ramp), amplitude

    raise ValueError(f"Unsupported rope_scaling: {scaling}")


class RotaryEmbedding:
    """Rotate Q and K in place by position, using one FP32 cos/sin table per model."""

    def __init__(self, config: ModelConfig):
        assert config.head_dim in (64, 128, 256, 512), "FlashInfer RoPE head sizes"
        self.head_dim = config.head_dim
        inv_freq, amplitude = rope_frequencies(config)
        positions = torch.arange(config.max_position, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)  # [max_position, head_dim // 2]
        # Cos, then sin, for each frequency. YaRN scales Q and K through this table:
        # omitting its amplitude changes attention scores even at position zero.
        self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * amplitude

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        # positions: [T]. query/key: [T, heads * head_dim], rotated in place.
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_dim,
            cos_sin_cache=self.cos_sin_cache,
        )
        return query, key


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from flashinfer import rmsnorm

        return rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        from flashinfer import rmsnorm

        rmsnorm(x, self.weight, self.eps, out=x)


class ResidualRMSNorm(RMSNorm):
    """Return (normalized branch, accumulated residual).

    First call: preserve x as the residual. Later calls: the fused kernel
    updates residual += x and overwrites x with RMSNorm(residual).
    """

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        from flashinfer import fused_add_rmsnorm

        if residual is None:
            return super().forward(x), x
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class RowParallelLinear(nn.Linear):
    """Each rank multiplies its input shard; the partial outputs are summed."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__(div_even(input_size, get_tp_info().size), output_size, bias=False)
        self.weight.weight_loader = load_row_shard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [T, input_size / TP] -> [T, output_size], summed across ranks.
        return all_reduce(F.linear(x, self.weight))


class VocabParallelEmbedding(nn.Embedding):
    """Ranks own disjoint vocabulary rows; each zeros the IDs it does not own."""

    def __init__(self, vocab_size: int, hidden_size: int):
        local_size, self.start, self.end = vocab_range(vocab_size)
        super().__init__(local_size, hidden_size)
        self.weight.weight_loader = load_vocab_shard
        self.tp_size = get_tp_info().size

    @torch.cuda.nvtx.range("Embedding")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [T] -> [T, H]. Exactly one rank owns each ID, so the sum recovers its row.
        if self.tp_size == 1:
            return F.embedding(input_ids, self.weight)
        owned = (input_ids >= self.start) & (input_ids < self.end)
        output = F.embedding((input_ids - self.start) * owned, self.weight)
        return all_reduce(output * owned.unsqueeze(1))


class ParallelLMHead(nn.Linear):
    """Project to a vocabulary shard, then restore batch-major vocabulary order."""

    def __init__(self, vocab_size: int, hidden_size: int):
        self.tp_size = get_tp_info().size
        self.vocab_size = vocab_size
        super().__init__(hidden_size, vocab_range(vocab_size)[0], bias=False)
        self.weight.weight_loader = load_vocab_shard

    @torch.cuda.nvtx.range("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.weight)  # [token, H] -> [token, local_vocab]
        if self.tp_size == 1:
            return logits
        # all_gather stacks [rank, token, local_vocab]. Sampling needs
        # [token, rank, local_vocab], with any padded vocabulary rows removed.
        gathered = all_gather(logits).view(self.tp_size, *logits.shape)
        return gathered.transpose(0, 1).reshape(len(x), -1)[:, : self.vocab_size]


class MoELayer(nn.Module):
    """Experts shard intermediate features; every rank sees the same router."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        local_size = div_even(intermediate_size, get_tp_info().size)
        self.top_k, self.renormalize = top_k, renormalize
        # Stacked per expert. gate_up: [experts, 2 * local, H], down: [experts, H, local].
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * local_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, local_size))
        self.gate_up_proj.weight_loader = self.down_proj.weight_loader = self._load_expert
        self.local_size = local_size

    def _load_expert(self, param: nn.Parameter, value: torch.Tensor, shard) -> int:
        """shard = (expert, 0 gate / 1 up / None down): shard like the dense MLP."""
        expert, part = shard
        if part is None:
            return load_row_shard(param.data[expert], value)
        return load_column_shard(param.data[expert], value, part, self.local_size)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        from minisgl.moe import fused_moe

        output = fused_moe(
            hidden_states,
            self.gate_up_proj,
            self.down_proj,
            router_logits,
            self.top_k,
            self.renormalize,
        )
        return all_reduce(output)
