"""Admission and one scheduling step. Transport belongs to server.py or llm.py.

Start at Scheduler.step: process inputs, choose prefill or decode, submit GPU
work, then retire a completed batch. The same step serves online and offline use.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import accumulate, chain
from typing import TYPE_CHECKING, NamedTuple

import torch
from minisgl.config import Config
from minisgl.core import Batch, Request
from minisgl.engine import Engine, make_device_tensor, prepare_sampling_args
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


class ScheduledBatch(NamedTuple):
    batch: Batch
    sampling_args: BatchSamplingArgs
    # [padded_size] request rows. Made on the metadata stream but read later on the
    # engine stream, so it must stay alive until this batch completes.
    rows: torch.Tensor
    # Rows whose sampled token is output. Record it now: with overlap, a chunked
    # request has already moved on to its next chunk when this batch completes.
    emits: list[bool]


class InFlightBatch(NamedTuple):
    scheduled: ScheduledBatch
    output: ForwardOutput


class Scheduler:
    def __init__(self, config: Config):
        self.engine = Engine(config)
        self.config = self.engine.config  # Resolved backend and allocation page size.
        self.device = self.engine.device
        self.scheduler_stream = torch.cuda.Stream(device=self.device)
        # [requests + 1]: each row's latest sampled token, which its next decode reads
        # on the GPU without waiting for the CPU copy. The extra row feeds graph padding.
        rows = len(self.engine.page_table)
        self.next_tokens = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.free_rows = list(range(config.max_running_req))
        self.cache_manager = KVCacheManager(
            self.engine.num_pages, self.config.page_size, self.engine.page_table, config.cache_type
        )
        self.eos_token_ids = load_eos_token_ids(config.model_path, config.hf_config)
        self.pending: list[Request] = []  # Waiting, or partway through a chunked prefill.
        self.running: list[Request] = []  # Decoding. Lists keep the same order on every rank.
        self.in_flight: InFlightBatch | None = None

    @property
    def idle(self) -> bool:
        # A final GPU-to-CPU copy can still be pending after decode becomes empty.
        return not (self.pending or self.running or self.in_flight)

    @torch.inference_mode()
    def step(self, messages: list[Message]) -> list[Message]:
        """Submit one batch and return tokens ready for the caller to deliver."""
        # With overlap, batch metadata is copied on a side stream while the previous
        # forward runs. That forward reads only its own copies, never page_table.
        stream = self.scheduler_stream if self.config.overlap else self.engine.stream
        with torch.cuda.stream(stream):
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
            self.pending.append(Request(message.uid, list(message.input_ids), params))
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
        scheduled = []
        for request in self.pending:
            if budget == 0 or (request.table_idx < 0 and not self._admit(request, reserved)):
                break
            reserved += self._future_kv_tokens(request.cached_len, request.max_seq_len - 1)
            request.scheduled_len = min(request.cached_len + budget, request.prompt_len)
            budget -= request.num_new_tokens
            scheduled.append(request)
        # An unfinished prompt keeps its place at the front of the queue.
        chunks = [r for r in scheduled if r.is_prefill_chunk]
        self.pending = chunks + self.pending[len(scheduled) :]
        return Batch(scheduled, "prefill") if scheduled else None

    def _admit(self, request: Request, reserved: int) -> bool:
        """Give a new request a row and its cached prefix, if its KV budget fits."""
        if not self.free_rows:
            return False
        # Cache KV, not logits: recompute at least the final prompt token.
        prefix = torch.tensor(request.input_ids[:-1], dtype=torch.int32)
        handle = self.cache_manager.prefix_cache.match_prefix(prefix)
        needed = self._future_kv_tokens(handle.cached_len, request.max_seq_len - 1)
        if needed + reserved > self.cache_manager.available_tokens:
            return False
        self.cache_manager.lock(handle)
        # The match was evictable before locking, so capacity changed.
        if needed + reserved > self.cache_manager.available_tokens:
            self.cache_manager.unlock(handle)
            return False
        request.table_idx, request.cache_handle = self.free_rows.pop(), handle
        request.cached_len = handle.cached_len
        if handle.cached_len:  # Point the row at the shared prefix's KV slots.
            row = self.engine.page_table[request.table_idx, : handle.cached_len]
            row.copy_(handle.get_matched_indices())
        return True

    def _schedule_next_batch(self) -> ScheduledBatch | None:
        batch = self._prefill_batch()
        if batch is None and self.running:
            batch = Batch(list(self.running), "decode")
        return self._prepare_batch(batch) if batch is not None else None

    def _prepare_batch(self, batch: Batch) -> ScheduledBatch:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        device = self.device
        batch.positions, token_rows = _token_positions_and_rows(batch, device)  # Both [T].
        batch.out_loc = self.engine.page_table[token_rows, batch.positions.long()]  # KV slots
        rows = make_device_tensor([r.table_idx for r in batch.padded_reqs], torch.int64, device)
        if batch.is_prefill:  # Prompt tokens come from CPU; decode reads next_tokens instead.
            chunks = (r.input_ids[r.cached_len : r.scheduled_len] for r in batch.reqs)
            batch.input_ids = make_device_tensor(list(chain(*chunks)), torch.int32, device)
            # The model needs one logit row per request, regardless of the
            # attention library's private metadata representation.
            ends = accumulate(r.num_new_tokens for r in batch.reqs)
            batch.last_token_indices = make_device_tensor(
                [e - 1 for e in ends], torch.int64, device
            )
        self.engine.attn_backend.prepare_metadata(batch)
        args = prepare_sampling_args(batch, device, self.config.model_config.vocab_size)
        emits = [not r.is_prefill_chunk for r in batch.reqs]
        return ScheduledBatch(batch, args, rows, emits)

    def _run_batch(self, scheduled: ScheduledBatch) -> ForwardOutput:
        batch, rows = scheduled.batch, scheduled.rows
        if batch.is_decode:  # The previous samples may not have reached the CPU yet.
            batch.input_ids = self.next_tokens[rows]
        output = self.engine.forward_batch(batch, scheduled.sampling_args)
        # A prefill chunk's sample is discarded, and harmless here: the request's
        # final chunk overwrites this slot before any decode reads it.
        self.next_tokens[rows[: batch.size]] = output.next_tokens_gpu
        # dict.fromkeys drops duplicates and keeps order, identically on every rank.
        self.running = [r for r in dict.fromkeys(self.running + batch.reqs) if r.can_decode]
        return output

    def _process_completed_batch(self, completed: InFlightBatch | None) -> list[DetokenizeMsg]:
        if completed is None:
            return []
        completed.output.copy_done_event.synchronize()
        replies = []
        with self.cache_manager.defer_page_frees():
            scheduled = completed.scheduled
            for index, (request, emits) in enumerate(zip(scheduled.batch.reqs, scheduled.emits)):
                # Overlap may submit a lookahead before EOS/cancellation is
                # observed. Intermediate chunks and retired requests never reply.
                if not emits or request.is_finished:
                    continue
                token_id, reason = int(completed.output.next_tokens_cpu[index]), None
                request.input_ids.append(token_id)
                if not request.sampling_params.ignore_eos and token_id in self.eos_token_ids:
                    reason = "stop"
                elif request.num_output_tokens >= request.sampling_params.max_tokens:
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
        if request in self.running:
            self.running.remove(request)
        self.free_rows.append(request.table_idx)
        self.cache_manager.cache_req(request, finished=True)

    def _abort_request(self, uid: int) -> None:
        for index, pending in enumerate(self.pending):
            if pending.uid == uid:
                self.pending.pop(index)
                if pending.table_idx >= 0:  # Admitted: a chunked prefill is underway.
                    self._free_request(pending)
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
