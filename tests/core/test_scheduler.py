"""CPU admission regressions: real page accounting, without loading a model."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.config import Config
from minisgl.core import SamplingParams
from minisgl.kvcache import KVCacheManager
from minisgl.message import AbortMsg, CancelledMsg, ErrorMsg, UserMsg
from minisgl.scheduler import Scheduler


@pytest.fixture
def scheduler(monkeypatch):
    # A real CPU page table keeps admission, chunking, locks, and frees observable.
    scheduler = object.__new__(Scheduler)
    scheduler.config = Config("unused", page_size=4, max_running_req=8, max_extend_tokens=16)
    table = torch.zeros((8, 32), dtype=torch.int32)
    scheduler.engine = SimpleNamespace(page_table=table, max_seq_len=16)
    scheduler.cache_manager = KVCacheManager(4, 4, table, "radix")
    scheduler.free_rows = list(range(8))
    scheduler.pending, scheduler.running, scheduler.in_flight = [], [], None
    return scheduler


def test_admission_reserves_whole_pages_for_short_requests(scheduler):
    for uid in range(7):
        scheduler._handle_message(UserMsg(uid, [uid], SamplingParams(max_tokens=1)))
    batch = scheduler._prefill_batch()
    assert len(batch.reqs) == 4  # Four physical pages, not sixteen one-token slots.
    assert len(scheduler.pending) == 3
    scheduler.cache_manager.allocate_paged(batch.reqs)
    assert scheduler.cache_manager.free_pages.numel() == 0
    for request in batch.reqs:
        request.advance_after_forward()
        scheduler._free_request(request)
    scheduler.cache_manager.check_integrity()


def test_chunked_prefill_keeps_one_row_and_emits_only_after_the_full_prompt(scheduler):
    object.__setattr__(scheduler.config, "max_extend_tokens", 3)
    ids = torch.arange(7, dtype=torch.int32)
    scheduler._handle_message(UserMsg(0, ids.tolist(), SamplingParams(max_tokens=2)))
    rows, chunk_lengths, intermediate = [], [], []
    while scheduler.pending:
        request = scheduler._prefill_batch().reqs[0]
        rows.append(request.table_idx)
        chunk_lengths.append(request.num_new_tokens)
        intermediate.append(request.is_prefill_chunk)
        scheduler.cache_manager.allocate_paged([request])
        request.advance_after_forward()
    assert chunk_lengths == [3, 3, 1]
    assert intermediate == [True, True, False]
    assert len(set(rows)) == 1
    scheduler._free_request(request)
    scheduler.cache_manager.check_integrity()


def test_matching_prefix_is_locked_and_mapped_before_the_row_is_used(scheduler):
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    pages = scheduler.cache_manager._allocate(1)
    locations = scheduler.cache_manager._page_to_token(pages)
    scheduler.cache_manager.prefix_cache.insert_prefix(ids, locations)
    scheduler._handle_message(UserMsg(0, [1, 2, 3, 4, 5], SamplingParams(max_tokens=2)))
    request = scheduler._prefill_batch().reqs[0]
    assert request.cached_len == 4
    assert scheduler.cache_manager.prefix_cache.protected_size == 4
    assert torch.equal(scheduler.engine.page_table[request.table_idx, :4], locations)
    assert request.num_new_tokens == 1  # Only the token after the cached prefix is computed.


def test_abort_during_chunking_releases_its_row_and_pages_once(scheduler):
    object.__setattr__(scheduler.config, "max_extend_tokens", 3)
    scheduler._handle_message(UserMsg(0, list(range(7)), SamplingParams(max_tokens=2)))
    request = scheduler._prefill_batch().reqs[0]
    scheduler.cache_manager.allocate_paged([request])
    request.advance_after_forward()
    assert scheduler._handle_message(AbortMsg(0)) == [CancelledMsg(0)]
    assert scheduler.idle and request.is_finished
    scheduler._handle_message(AbortMsg(0))
    assert len(set(scheduler.free_rows)) == len(scheduler.free_rows) == 8
    scheduler.cache_manager.check_integrity()


def test_context_clipping_does_not_mutate_shared_sampling_settings(scheduler):
    params = SamplingParams(max_tokens=10)
    scheduler._handle_message(UserMsg(0, list(range(14)), params))
    scheduler._handle_message(UserMsg(1, list(range(3)), params))
    assert [request.sampling_params.max_tokens for request in scheduler.pending] == [2, 10]
    assert params.max_tokens == 10


@pytest.mark.parametrize("length", [0, 16, 17])
def test_invalid_prompt_returns_an_error_instead_of_leaving_a_waiting_client(scheduler, length):
    replies = scheduler._handle_message(UserMsg(42, list(range(length)), SamplingParams()))
    assert len(replies) == 1 and isinstance(replies[0], ErrorMsg)
    assert replies[0].uid == 42
    assert scheduler.idle
