from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import AttentionMetadata
    from minisgl.kvcache import CacheHandle


@dataclass
class SamplingParams:
    """Per-request generation settings; ``max_tokens`` counts newly sampled tokens."""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0 or self.top_k == 1


@dataclass(eq=False)
class Request:
    """A request's CPU history and the lengths used to schedule GPU work.

    ``input_ids`` contains the prompt and output tokens already copied to CPU.
    ``scheduled_len`` includes output tokens produced by submitted forwards,
    even when their CPU copies have not been consumed yet. ``cached_len`` is
    the length whose KV writes have been submitted on the engine stream.

    Example: after prefill of a 4-token prompt, cached_len=4 and
    scheduled_len=5. The sampled fifth token has no KV until the next forward.
    Identity-based equality lets the scheduler keep active requests in a set.
    """

    input_ids: torch.Tensor  # 1-D CPU int32 token IDs
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: CacheHandle
    is_prefill_chunk: bool = False
    prompt_len: int = field(init=False)
    scheduled_len: int = field(init=False)
    max_seq_len: int = field(init=False)
    is_finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu and self.input_ids.ndim == 1
        self.prompt_len = len(self.input_ids)
        self.scheduled_len = self.prompt_len
        self.max_seq_len = self.prompt_len + self.output_len
        assert 0 <= self.cached_len < self.scheduled_len <= self.max_seq_len

    @property
    def remaining_tokens(self) -> int:
        return self.max_seq_len - self.scheduled_len

    @property
    def num_new_tokens(self) -> int:
        return self.scheduled_len - self.cached_len

    def advance_after_forward(self) -> None:
        """Advance logical GPU state without waiting for the sampled CPU token."""
        self.cached_len = self.scheduled_len
        self.scheduled_len += 1

    def append_output_token(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def num_output_tokens(self) -> int:
        """Count delivered tokens, independently of in-flight GPU work."""
        return len(self.input_ids) - self.prompt_len

    @property
    def can_decode(self) -> bool:
        return not self.is_prefill_chunk and not self.is_finished and self.remaining_tokens > 0

    def __repr__(self) -> str:
        return (
            f"Request(uid={self.uid}, table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, scheduled_len={self.scheduled_len}, "
            f"max_seq_len={self.max_seq_len})"
        )


@dataclass
class Batch:
    """Ragged tokens flattened across requests, with padding only for decode graphs.

    input_ids/positions/out_loc have one entry per newly processed token.
    ``positions`` are logical sequence offsets; ``out_loc`` are physical KV
    token slots. These two address spaces must never be interchanged.
    """

    reqs: list[Request]
    phase: Literal["prefill", "decode"]
    # Filled by the scheduler before submitting GPU work.
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    last_token_indices: torch.Tensor = field(init=False)  # Prefill rows selected for logits.
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: list[Request] = field(init=False)
    # Filled by the selected attention backend.
    attn_metadata: AttentionMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)
