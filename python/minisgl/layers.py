"""GPU building blocks: RoPE, fused RMSNorm, and tensor-parallel layers.

Model weights use ordinary nn.Module/Parameter objects. Kernels below change
how an operation runs, not how its checkpoint tensors are named.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import torch
from minisgl.distributed import all_gather, all_reduce, get_tp_info
from minisgl.utils import div_ceil, div_even
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from minisgl.config import ModelConfig


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


class RotaryEmbedding:
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
        attention_factor: float = 1.0,
    ) -> None:
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # [max_position, head_size]: cos then sin for each of head_size // 2 frequencies.
        # Computed once per model in FP32; shared by all decoder layers.
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        # YaRN scales both Q and K through the rotation table. Omitting this
        # amplitude changes attention scores even at position zero.
        if attention_factor != 1.0:
            self._cos_sin_cache.mul_(attention_factor)
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: [T]. query/key: [T, heads * head_size], rotated in place.
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast") or 32.0
            beta_slow: float = rope_scaling.get("beta_slow") or 1.0
            orig_max_pos: int = rope_scaling.get("original_max_position_embeddings") or max_position

            def get_mscale(multiplier: float = 1.0) -> float:
                return 1.0 if factor <= 1 else 1.0 + 0.1 * multiplier * math.log(factor)

            attention_factor = rope_scaling.get("attention_factor")
            if attention_factor is None:
                mscale, mscale_all = rope_scaling.get("mscale"), rope_scaling.get("mscale_all_dim")
                attention_factor = (
                    get_mscale(mscale) / get_mscale(mscale_all)
                    if mscale and mscale_all
                    else get_mscale()
                )

            def _find_correction_dim(num_rotations: float) -> float:
                return (
                    rotary_dim
                    * math.log(orig_max_pos / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low, high = _find_correction_dim(beta_fast), _find_correction_dim(beta_slow)
            if rope_scaling.get("truncate", True):
                low, high = math.floor(low), math.ceil(high)
            low, high = max(low, 0), min(high, rotary_dim - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low)
                    / (high - low if high != low else 1e-3),
                    0,
                    1,
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(
                head_dim,
                rotary_dim,
                max_position,
                base,
                post_process,
                attention_factor=attention_factor,
            )

    raise ValueError(f"Unsupported {rope_scaling = }")


def create_rope(config: ModelConfig, device: torch.device | str) -> RotaryEmbedding:
    # Parameters are initially on meta, but RoPE needs actual numbers. Compute
    # its shared table on the execution device rather than using global state.
    with torch.device(device):
        return _get_rope(
            config.head_dim,
            config.head_dim,
            config.max_position,
            config.rope_theta,
            config.rope_scaling,
        )


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
    """Each rank multiplies its input shard; sum partial outputs before bias."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        super().__init__(div_even(input_size, get_tp_info().size), output_size, bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [T, input_size / TP] -> [T, output_size], summed across ranks.
        output = all_reduce(F.linear(x, self.weight))
        return output if self.bias is None else output + self.bias


class VocabParallelEmbedding(nn.Embedding):
    """Ranks own disjoint vocabulary rows; the indexing kernel zeros misses."""

    def __init__(self, vocab_size: int, hidden_size: int):
        tp = get_tp_info()
        local_size = div_ceil(vocab_size, tp.size)
        super().__init__(local_size, hidden_size)
        start = local_size * tp.rank
        self.vocab_range = (start, min(start + local_size, vocab_size) - start)
        self.tp_size = tp.size

    @torch.cuda.nvtx.range("Embedding")
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [T] -> [T, H]. Exactly one rank owns each ID, so the sum recovers its row.
        from minisgl.kernel import indexing

        output = indexing(
            self.weight, input_ids, vocab_range=self.vocab_range if self.tp_size > 1 else None
        )
        return all_reduce(output) if self.tp_size > 1 else output


class ParallelLMHead(nn.Linear):
    """Project to a vocabulary shard, then restore batch-major vocabulary order."""

    def __init__(self, vocab_size: int, hidden_size: int):
        self.tp_size = get_tp_info().size
        self.vocab_size = vocab_size
        super().__init__(hidden_size, div_ceil(vocab_size, self.tp_size), bias=False)

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
