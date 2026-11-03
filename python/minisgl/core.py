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
    """One request, from admission to its last token.

    ``input_ids`` is the prompt plus every output token already copied to CPU.
    Two lengths follow the GPU instead: ``cached_len`` tokens have KV writes
    submitted, and ``scheduled_len`` tokens are, or are being, computed. A prefill
    chunk ends before ``prompt_len``; its sampled token is discarded.

    Example: after prefill of a 4-token prompt, cached_len=4 and scheduled_len=5.
    The sampled fifth token has no KV until the next forward, and its CPU copy may
    still be in flight. Identity-based equality keeps requests hashable.
    """

    uid: int
    input_ids: list[int]
    sampling_params: SamplingParams
    table_idx: int = -1  # Row in page_table and next_tokens, assigned at admission.
    cache_handle: CacheHandle | None = None
    cached_len: int = 0
    prompt_len: int = field(init=False)
    scheduled_len: int = field(init=False)
    max_seq_len: int = field(init=False)
    is_finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.prompt_len = self.scheduled_len = len(self.input_ids)
        self.max_seq_len = self.prompt_len + self.sampling_params.max_tokens

    @property
    def is_prefill_chunk(self) -> bool:
        return self.scheduled_len < self.prompt_len

    @property
    def remaining_tokens(self) -> int:
        return self.max_seq_len - self.scheduled_len

    @property
    def num_new_tokens(self) -> int:
        return self.scheduled_len - self.cached_len

    def advance_after_forward(self) -> None:
        """Advance logical GPU state without waiting for the sampled CPU token."""
        chunk = self.is_prefill_chunk
        self.cached_len = self.scheduled_len
        if not chunk:  # The sampled token is real: the next forward computes it.
            self.scheduled_len += 1

    @property
    def num_output_tokens(self) -> int:
        """Count delivered tokens, independently of in-flight GPU work."""
        return len(self.input_ids) - self.prompt_len

    @property
    def can_decode(self) -> bool:
        return not self.is_prefill_chunk and not self.is_finished and self.remaining_tokens > 0

    def __repr__(self) -> str:
        return (
            f"Request(uid={self.uid}, table_idx={self.table_idx}, cached_len={self.cached_len}, "
            f"scheduled_len={self.scheduled_len}, max_seq_len={self.max_seq_len})"
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
