"""Admission and one scheduling step. Transport belongs to server.py or llm.py.

Start at Scheduler.step: process inputs, choose prefill or decode, submit GPU
work, then retire a completed batch. The same step serves online and offline use.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple

import torch
from minisgl.config import Config
from minisgl.core import Batch, Request, SamplingParams
from minisgl.engine import Engine, prepare_sampling_args
from minisgl.kvcache import KVCacheManager
from minisgl.message import (
    AbortMsg,
    BatchMsg,
    CancelledMsg,
    DetokenizeMsg,
    ErrorMsg,
    Message,
    UserMsg,
)
from minisgl.utils import align_ceil, load_eos_token_ids

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


@dataclass
class PendingRequest:
    uid: int
    input_ids: torch.Tensor  # Full prompt stays on CPU while chunks run on GPU.
    sampling_params: SamplingParams
    chunked_req: Request | None = None


class ScheduledBatch(NamedTuple):
    # Keep index tensors alive until GPU gathers/scatters have completed. They
    # are made on the metadata stream and consumed on the engine stream.
    batch: Batch
    sampling_args: BatchSamplingArgs
    input_indices: tuple[torch.Tensor, torch.Tensor]
    output_indices: tuple[torch.Tensor, torch.Tensor]


class InFlightBatch(NamedTuple):
    scheduled: ScheduledBatch
    output: ForwardOutput


class Scheduler:
    def __init__(self, config: Config):
        self.engine = Engine(config)
        self.config = self.engine.config  # Resolved backend and allocation page size.
        self.device = self.engine.device
        self.scheduler_stream = torch.cuda.Stream(device=self.device)
        # [requests + 1, max_seq_len], like page_table: row r, position p -> token ID.
        self.token_pool = torch.zeros_like(self.engine.page_table)
        self.free_rows = list(range(config.max_running_req))
        self.cache_manager = KVCacheManager(
            self.engine.num_pages, self.config.page_size, self.engine.page_table, config.cache_type
        )
        self.eos_token_ids = load_eos_token_ids(config.model_path, config.hf_config)
        self.pending: list[PendingRequest] = []
        self.running: set[Request] = set()
        self.in_flight: InFlightBatch | None = None

    @property
    def idle(self) -> bool:
        # A final GPU-to-CPU copy can still be pending after decode becomes empty.
        return not (self.pending or self.running or self.in_flight)

    @torch.inference_mode()
    def step(self, messages: list[Message]) -> list[Message]:
        """Submit one batch and return tokens ready for the caller to deliver."""
        stream = self.scheduler_stream if self.config.overlap else self.engine.stream
        with torch.cuda.stream(stream):
            # Reused request rows must not race the preceding forward's token
            # writes. This GPU dependency still lets CPU admission run ahead.
            if self.config.overlap:
                stream.wait_stream(self.engine.stream)
            replies = []
            for message in messages:
                replies.extend(self._handle_message(message))
            scheduled = self._schedule_next_batch()
            current = None
            if scheduled is not None:
                with torch.cuda.stream(self.engine.stream):
                    if self.config.overlap:
                        self.engine.stream.wait_stream(stream)
                    current = InFlightBatch(scheduled, self._run_batch(scheduled))

            # Overlap queues the next forward BEFORE waiting for the prior CPU
            # token copy. Serial mode consumes this forward in the same step.
            completed = self.in_flight if self.config.overlap else current
            replies.extend(self._process_completed_batch(completed))
            self.in_flight = current if self.config.overlap else None
            return replies

    def _handle_message(self, message: Message) -> list[Message]:
        if isinstance(message, BatchMsg):
            return [error for child in message.data for error in self._handle_message(child)]
        if isinstance(message, AbortMsg):
            self._abort_request(message.uid)
            return [CancelledMsg(message.uid)]
        elif isinstance(message, UserMsg):
            available_output = self.engine.max_seq_len - len(message.input_ids)
            if not len(message.input_ids) or available_output < 1:
                return [
                    ErrorMsg(
                        message.uid,
                        "Prompt must contain tokens and leave room for output "
                        f"within the {self.engine.max_seq_len}-token context limit.",
                    )
                ]
            # Do not mutate SamplingParams: callers often reuse one instance
            # for several prompts with different remaining context lengths.
            params = replace(
                message.sampling_params,
                max_tokens=min(message.sampling_params.max_tokens, available_output),
            )
            self.pending.append(PendingRequest(message.uid, message.input_ids, params))
        else:
            raise TypeError(f"Unknown scheduler message: {type(message).__name__}")
        return []

    def _future_kv_tokens(self, cached_len: int, target_len: int) -> int:
        # Reserve whole pages, not just token counts. The last sampled token
        # needs no KV: it is never fed through the model again.
        page_size = self.config.page_size
        return align_ceil(target_len, page_size) - align_ceil(cached_len, page_size)

    def _prefill_batch(self) -> Batch | None:
        budget = self.config.max_extend_tokens
        reserved = sum(
            self._future_kv_tokens(r.cached_len, r.max_seq_len - 1) for r in self.running
        )
        requests, chunks = [], []
        for pending in self.pending:
            if budget == 0:
                break
            ids, params = pending.input_ids, pending.sampling_params
            previous = pending.chunked_req
            if previous is not None:
                handle, row, cached_len = (
                    previous.cache_handle,
                    previous.table_idx,
                    previous.cached_len,
                )
            else:
                if not self.free_rows:
                    break
                # Cache KV, not logits: recompute at least the final prompt token.
                handle = self.cache_manager.prefix_cache.match_prefix(ids[:-1])
                cached_len = handle.cached_len
                needed = self._future_kv_tokens(cached_len, len(ids) + params.max_tokens - 1)
                if needed + reserved > self.cache_manager.available_tokens:
                    break
                self.cache_manager.lock(handle)
                # The match was evictable before locking, so capacity changed.
                if needed + reserved > self.cache_manager.available_tokens:
                    self.cache_manager.unlock(handle)
                    break
                row = self.free_rows.pop()
                if cached_len:  # token_pool needs no prefix: only new tokens are read.
                    self.engine.page_table[row, :cached_len].copy_(handle.get_matched_indices())

            reserved += self._future_kv_tokens(cached_len, len(ids) + params.max_tokens - 1)
            end = min(cached_len + budget, len(ids))
            self.token_pool[row, cached_len:end].copy_(
                ids[cached_len:end].pin_memory(), non_blocking=True
            )
            request = Request(
                input_ids=ids[:end],
                table_idx=row,
                cached_len=cached_len,
                output_len=params.max_tokens,
                uid=pending.uid,
                sampling_params=params,
                cache_handle=handle,
                is_prefill_chunk=end < len(ids),
            )
            budget -= end - cached_len
            pending.chunked_req = request if request.is_prefill_chunk else None
            if request.is_prefill_chunk:
                chunks.append(pending)
            requests.append(request)

        self.pending = chunks + self.pending[len(requests) :]
        return Batch(requests, "prefill") if requests else None

    def _schedule_next_batch(self) -> ScheduledBatch | None:
        batch = self._prefill_batch()
        if batch is None and self.running:
            # A Python set's order differs across TP processes. Every rank must
            # use the same row order before executing collective operations.
            batch = Batch(sorted(self.running, key=lambda request: request.uid), "decode")
        return self._prepare_batch(batch) if batch is not None else None

    def _prepare_batch(self, batch: Batch) -> ScheduledBatch:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions, rows = _token_positions_and_rows(batch, self.device)  # Both [T].
        input_indices = (rows, batch.positions.long())
        output_indices = _output_indices(batch, self.device)
        batch.out_loc = self.engine.page_table[input_indices]  # [T] physical token slots
        if batch.is_prefill:
            # The model needs one logit row per request, regardless of the
            # attention library's private metadata representation.
            ends = accumulate(r.num_new_tokens for r in batch.reqs)
            batch.last_token_indices = torch.tensor(
                [end - 1 for end in ends], dtype=torch.int64, pin_memory=True
            ).to(self.device, non_blocking=True)
        self.engine.attn_backend.prepare_metadata(batch)
        args = prepare_sampling_args(batch, self.device, self.config.model_config.vocab_size)
        return ScheduledBatch(batch, args, input_indices, output_indices)

    def _run_batch(self, scheduled: ScheduledBatch) -> ForwardOutput:
        batch = scheduled.batch
        batch.input_ids = self.token_pool[scheduled.input_indices]
        output = self.engine.forward_batch(batch, scheduled.sampling_args)
        # Feed the next decode directly from GPU IDs, without waiting for D2H.
        self.token_pool[scheduled.output_indices] = output.next_tokens_gpu
        self.running = {r for r in self.running.union(batch.reqs) if r.can_decode}
        return output

    def _process_completed_batch(self, completed: InFlightBatch | None) -> list[DetokenizeMsg]:
        if completed is None:
            return []
        completed.output.copy_done_event.synchronize()
        replies = []
        with self.cache_manager.defer_page_frees():
            for index, request in enumerate(completed.scheduled.batch.reqs):
                # Overlap may submit a lookahead before EOS/cancellation is
                # observed. Intermediate chunks and retired requests never reply.
                if request.is_prefill_chunk or request.is_finished:
                    continue
                token = completed.output.next_tokens_cpu[index]
                request.append_output_token(token.unsqueeze(0))
                token_id, reason = int(token), None
                if not request.sampling_params.ignore_eos and token_id in self.eos_token_ids:
                    reason = "stop"
                elif request.num_output_tokens >= request.output_len:
                    reason = "length"
                # Only delivered CPU tokens count toward the output budget;
                # scheduled_len may already include the following forward.
                finished = reason is not None
                replies.append(DetokenizeMsg(request.uid, token_id, finished, reason))
                if finished:
                    self._free_request(request)
                elif completed.scheduled.batch.is_prefill:
                    self.cache_manager.cache_req(request, finished=False)
        return replies

    def _free_request(self, request: Request) -> None:
        request.is_finished = True
        self.running.discard(request)
        self.free_rows.append(request.table_idx)
        self.cache_manager.cache_req(request, finished=True)

    def _abort_request(self, uid: int) -> None:
        for index, pending in enumerate(self.pending):
            if pending.uid == uid:
                self.pending.pop(index)
                if pending.chunked_req is not None:
                    self._free_request(pending.chunked_req)
                return
        candidates = list(self.running)
        if self.in_flight is not None:
            candidates += self.in_flight.scheduled.batch.reqs
        for request in candidates:
            if request.uid == uid and not request.is_finished:
                self._free_request(request)
                return

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.engine.shutdown()


def _token_positions_and_rows(batch: Batch, device: torch.device):
    size = sum(r.num_new_tokens for r in batch.padded_reqs)
    positions = torch.empty(size, dtype=torch.int32, pin_memory=True)
    rows = torch.empty(size, dtype=torch.int64, pin_memory=True)
    offset = 0
    for request in batch.padded_reqs:
        end = offset + request.num_new_tokens
        torch.arange(request.cached_len, request.scheduled_len, out=positions[offset:end])
        rows[offset:end].fill_(request.table_idx)
        offset = end
    return positions.to(device, non_blocking=True), rows.to(device, non_blocking=True)


def _output_indices(batch: Batch, device: torch.device):
    rows = torch.tensor([r.table_idx for r in batch.reqs], dtype=torch.int64, pin_memory=True)
    # Intermediate chunks discard sampled IDs into the last column, away from
    # the next chunk's prompt tokens. Dummy rows never produce an output.
    positions = torch.tensor(
        [r.scheduled_len if r.can_decode else -1 for r in batch.reqs],
        dtype=torch.int64,
        pin_memory=True,
    )
    return rows.to(device, non_blocking=True), positions.to(device, non_blocking=True)
