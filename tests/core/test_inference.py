"""CPU regressions for state, cache ownership, checkpoint layout, and API cleanup.

No CUDA device or downloaded model is needed. Native radix comparison uses the
normal TVM FFI binding, so a C++ compiler is required.
"""

import asyncio
import signal
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from minisgl.config import Config
from minisgl.core import Batch, Request, SamplingParams
from minisgl.distributed import DistributedInfo, stage_layers
from minisgl.engine import ForwardOutput, prepare_sampling_args, resolve_engine_config, sample
from minisgl.graph import _determine_cuda_graph_bs
from minisgl.kvcache import KVCacheManager
from minisgl.layers import ParallelLMHead, RowParallelLinear
from minisgl.llm import LLM
from minisgl.message import AbortMsg, DetokenizeMsg, TokenizeMsg, UserReply
from minisgl.models import _shard_tensor, _stage_key, create_model
from minisgl.scheduler import InFlightBatch, Scheduler
from minisgl.server import (
    FrontendManager,
    OpenAICompletionRequest,
    WorkerUnavailable,
    _chat_stream,
    _listen,
    _receive_requests,
    create_app,
)
from minisgl.tokenizer import DetokenizeManager, tokenize
from minisgl.utils import cached_load_hf_config, load_eos_token_ids


class TestAttentionBackendSelection(unittest.TestCase):
    def adjust_config(self, capability, **kwargs):
        config = Config(
            model_path="unused",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            **kwargs,
        )
        # Avoid downloading a model config in this CPU-only configuration test.
        object.__setattr__(config, "model_config", SimpleNamespace(is_moe=False))
        with (
            patch("minisgl.utils.get_compute_capability", return_value=capability),
            patch("minisgl.engine.logger"),
        ):
            resolved = resolve_engine_config(config)
        self.assertEqual(config.attention_backend, kwargs.get("attention_backend", "auto"))
        self.assertEqual(config.page_size, kwargs.get("page_size", 1))
        return resolved

    def test_auto_backend_by_gpu_architecture(self):
        cases = [
            ((8, 0), "fi", 1),
            ((8, 9), "fi", 1),
            ((9, 0), "fi", 1),
            ((10, 0), "trtllm", 64),
            ((10, 3), "trtllm", 64),
            ((12, 0), "fi", 1),  # RTX PRO 6000 / GeForce Blackwell
            ((12, 1), "fi", 1),
        ]
        for capability, backend, page_size in cases:
            with self.subTest(capability=capability):
                config = self.adjust_config(capability)
                self.assertEqual(config.attention_backend, backend)
                self.assertEqual(config.page_size, page_size)

    def test_sm120_preserves_explicit_backend(self):
        for backend in ("fi", "trtllm", "fi,trtllm"):
            with self.subTest(backend=backend):
                config = self.adjust_config((12, 0), attention_backend=backend)
                self.assertEqual(config.attention_backend, backend)

    def test_sm120_flashinfer_preserves_requested_page_size(self):
        config = self.adjust_config((12, 0), page_size=16)
        self.assertEqual(config.attention_backend, "fi")
        self.assertEqual(config.page_size, 16)


def test_context_limit_never_exceeds_the_rope_table():
    # RoPE is indexed by position with no bounds check on the GPU.
    config = Config("unused", max_seq_len_override=4096)
    object.__setattr__(config, "model_config", SimpleNamespace(max_position=1024))
    assert config.max_seq_len == 1024


def test_unsupported_dtype_fails_before_reaching_a_kernel():
    with pytest.raises(ValueError, match="dtype"):
        Config("unused", dtype=torch.float32)


def test_tp_and_pp_are_alternatives_and_pp_requires_pynccl():
    with pytest.raises(ValueError, match="not both"):
        Config("unused", tp_info=DistributedInfo(0, 2), pp_info=DistributedInfo(0, 2))
    with pytest.raises(ValueError, match="PyNCCL"):
        Config("unused", pp_info=DistributedInfo(0, 2), use_pynccl=False)


@pytest.mark.parametrize("num_layers,stages", [(28, 2), (28, 3), (7, 3), (4, 4)])
def test_pipeline_stages_cover_every_layer_exactly_once(num_layers, stages):
    ranges = [stage_layers(num_layers, DistributedInfo(s, stages)) for s in range(stages)]
    assert [layer for r in ranges for layer in r] == list(range(num_layers))
    assert max(map(len, ranges)) - min(map(len, ranges)) <= 1


def test_pipeline_stages_load_only_their_own_weights_under_local_names():
    config = SimpleNamespace(num_layers=4, tie_word_embeddings=True)
    stages = [DistributedInfo(0, 2), DistributedInfo(1, 2)]
    expected = {
        "model.embed_tokens.weight": ["model.embed_tokens.weight"] * 2,  # Tied to last stage.
        "model.layers.1.mlp.up_proj.weight": ["model.layers.1.mlp.up_proj.weight", None],
        "model.layers.3.mlp.up_proj.weight": [None, "model.layers.1.mlp.up_proj.weight"],
        "model.norm.weight": [None, "model.norm.weight"],
        "lm_head.weight": [None, "lm_head.weight"],
    }
    for key, owners in expected.items():
        assert [_stage_key(key, config, pp) for pp in stages] == owners
    config.tie_word_embeddings = False
    assert _stage_key("model.embed_tokens.weight", config, stages[1]) is None


def make_request(output_len=2):
    return Request(
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=42,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,
    )


def make_scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler.eos_token_ids = {99}
    scheduler.running = set()
    scheduler.free_rows = []
    scheduler.pending = []
    scheduler.in_flight = None
    scheduler.cache_manager = SimpleNamespace(defer_page_frees=nullcontext, cache_req=Mock())
    return scheduler


def deliver(scheduler, request, token, phase="decode"):
    output = ForwardOutput(None, torch.tensor([token]), SimpleNamespace(synchronize=lambda: None))
    scheduled = SimpleNamespace(batch=Batch([request], phase))
    return scheduler._process_completed_batch(InFlightBatch(scheduled, output))


def test_overlap_delivers_the_full_output_budget_before_finishing():
    request = make_request()
    scheduler = make_scheduler()
    request.advance_after_forward()  # prefill: sampled token 10
    request.advance_after_forward()  # decode: sampled token 11, CPU still behind
    assert not request.can_decode
    first_reply = deliver(scheduler, request, 10, "prefill")
    assert not first_reply[0].finished
    assert request.num_output_tokens == 1
    assert scheduler.free_rows == []
    last_reply = deliver(scheduler, request, 11)
    assert last_reply[0].finished and last_reply[0].finish_reason == "length"
    assert request.num_output_tokens == 2
    assert request.is_finished
    assert scheduler.free_rows == [0]


@pytest.mark.parametrize("output_len", [1, 2, 5])
def test_serial_completion_has_the_same_token_budget(output_len):
    request = make_request(output_len)
    scheduler = make_scheduler()
    for i in range(output_len):
        request.advance_after_forward()
        reply = deliver(scheduler, request, 10 + i)
        assert reply[0].finished == (i == output_len - 1)
    assert scheduler.free_rows == [0]


def test_eos_discards_the_overlapped_lookahead_and_frees_once():
    request = make_request(output_len=5)
    scheduler = make_scheduler()
    request.advance_after_forward()
    request.advance_after_forward()
    reply = deliver(scheduler, request, 99)[0]
    assert reply.finished and reply.finish_reason == "stop"
    assert deliver(scheduler, request, 10) == []
    assert request.num_output_tokens == 1
    assert scheduler.free_rows == [0]


def test_ignored_eos_counts_toward_the_output_budget():
    request = make_request(output_len=2)
    request.sampling_params.ignore_eos = True
    scheduler = make_scheduler()
    request.advance_after_forward()
    assert deliver(scheduler, request, 99)[0].finish_reason is None
    request.advance_after_forward()
    assert deliver(scheduler, request, 99)[0].finish_reason == "length"
    assert request.input_ids[-2:].tolist() == [99, 99]


def test_mixed_sampling_keeps_greedy_tied_maxima_deterministic(monkeypatch):
    import flashinfer.sampling as sampling

    def cpu_tensor(data, dtype, device):
        return torch.tensor(data, dtype=dtype)

    monkeypatch.setattr("minisgl.engine.make_device_tensor", cpu_tensor)
    requests = [
        SimpleNamespace(sampling_params=params)
        for params in (SamplingParams(temperature=0, top_p=0.5), SamplingParams(temperature=1))
    ]
    args = prepare_sampling_args(Batch(requests, "decode"), torch.device("cpu"), 3)
    logits = torch.tensor([[5.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    monkeypatch.setattr(sampling, "softmax", lambda values, *a, **kw: values.softmax(-1))
    monkeypatch.setattr(sampling, "top_p_sampling_from_probs", lambda *a: torch.tensor([1, 2]))
    # The sampler deliberately selects the *other* tied maximum for row 0.
    # Exact greedy semantics require argmax's first index, even in this batch.
    assert sample(logits, args).tolist() == [0, 2]


def test_aborted_inflight_request_does_not_reply_or_free_twice():
    request = make_request(output_len=5)
    scheduler = make_scheduler()
    scheduler.running.add(request)
    request.advance_after_forward()
    scheduler._handle_message(AbortMsg(uid=request.uid))
    assert deliver(scheduler, request, 10) == []
    assert scheduler.free_rows == [0]


def test_empty_output_budget_is_rejected():
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)


def test_trtllm_decode_preserves_query_rows_from_a_fused_qkv_projection(monkeypatch):
    import flashinfer.decode
    from minisgl.attention import PagedAttentionMetadata, TRTLLMBackend

    # Each request occupies a full QKV row, so Q's batch stride exceeds its width.
    qkv = torch.arange(24, dtype=torch.float32).view(2, 12)
    q, k, v = qkv.split(4, dim=-1)
    q = q.view(2, 2, 2)
    assert not q.is_contiguous()

    def dense_decode(*, query, **kwargs):
        # Model a kernel that reads packed [request, head, feature] rows.
        heads, head_dim = query.shape[1:]
        return query.as_strided(query.shape, (heads * head_dim, head_dim, 1)).clone()

    monkeypatch.setattr(flashinfer.decode, "trtllm_batch_decode_with_kv_cache", dense_decode)
    backend = object.__new__(TRTLLMBackend)
    backend.workspace_buffer = torch.empty(0)
    backend.scale = 1.0
    backend.kvcache = SimpleNamespace(store_kv=Mock(), k=[k], v=[v])
    batch = Batch([], "decode")
    batch.out_loc = torch.tensor([0, 1])
    batch.attn_metadata = PagedAttentionMetadata(
        cu_seqlens_k=torch.tensor([0, 1, 2]),
        cu_seqlens_q=torch.tensor([0, 1, 2]),
        cache_seqlens=torch.tensor([1, 1]),
        max_seqlen_k=1,
        max_seqlen_q=1,
        page_table=torch.tensor([[0], [1]]),
    )
    output = backend.forward(q, k, v, 0, batch)
    assert torch.equal(output, q)
    backend.kvcache.store_kv.assert_called_once_with(k, v, batch.out_loc, 0)


def test_graph_page_table_keeps_a_partial_final_page():
    from minisgl.attention import TRTLLMBackend

    backend = object.__new__(TRTLLMBackend)
    backend.kvcache = SimpleNamespace(device=torch.device("cpu"))
    backend.page_size, backend.capture = 64, None
    backend.init_capture_graph(max_seq_len=32, bs_list=[1, 4])
    assert backend.capture.page_table.shape == (4, 1)


@pytest.fixture
def cache(monkeypatch):
    table = torch.arange(16, dtype=torch.int32).view(1, 16)
    return KVCacheManager(4, 4, table, cache_type="radix")


def test_empty_prefix_has_valid_empty_indices(cache):
    handle = cache.prefix_cache.match_prefix(torch.tensor([], dtype=torch.int32))
    assert handle.cached_len == 0
    assert handle.get_matched_indices().numel() == 0


def test_split_locks_and_eviction_keep_accounting_consistent(cache):
    tree = cache.prefix_cache
    ids = torch.arange(8, dtype=torch.int32)
    _, full_handle = tree.insert_prefix(ids, ids.clone())
    prefix = tree.match_prefix(ids[:4])
    tree.lock_handle(prefix)
    tree.check_integrity()
    assert tree.protected_size == 4
    evicted = tree.evict(4)
    assert evicted.tolist() == [4, 5, 6, 7]
    tree.check_integrity()
    tree.lock_handle(prefix, unlock=True)
    tree.check_integrity()


def test_integrity_check_detects_corrupt_counters(cache):
    cache.prefix_cache.evictable_size = 4
    with pytest.raises(RuntimeError, match="accounting"):
        cache.prefix_cache.check_integrity()


def test_deferred_frees_flush_on_exception_and_reject_nesting(cache):
    cache._allocate(4)
    with pytest.raises(ValueError, match="test"):
        with cache.defer_page_frees():
            cache._free(torch.arange(8, dtype=torch.int32))
            assert cache.free_pages.numel() == 0
            with pytest.raises(RuntimeError, match="Nested"):
                with cache.defer_page_frees():
                    pass
            raise ValueError("test")
    assert cache.free_pages.tolist() == [0, 4]
    cache._free(torch.arange(8, 12, dtype=torch.int32))
    assert cache.free_pages.tolist() == [0, 4, 8]


def test_cancellation_caches_only_known_token_ids_and_returns_unknown_kv(cache):
    cache._allocate(2)
    ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    request = SimpleNamespace(
        input_ids=ids,
        cached_len=5,
        table_idx=0,
        cache_handle=cache.prefix_cache.match_prefix(ids[:0]),
    )
    cache.cache_req(request, finished=True)
    cache.check_integrity()
    assert cache.prefix_cache.total_size == 4
    assert sorted(cache.free_pages.tolist()) == [4, 8, 12]


@torch.no_grad()
def test_row_parallel_bias_is_added_after_all_reduce():
    with patch("minisgl.layers.get_tp_info", return_value=DistributedInfo(0, 2)):
        layer = RowParallelLinear(4, 3, has_bias=True)
    layer.weight.fill_(1)
    layer.bias.fill_(3)
    with patch("minisgl.layers.all_reduce", side_effect=lambda x: 2 * x):
        result = layer.forward(torch.ones(1, 2))
    assert torch.equal(result, torch.full((1, 3), 7.0))


@torch.no_grad()
def test_vocabulary_gather_restores_batch_order_before_trimming_padding(monkeypatch):
    monkeypatch.setattr("minisgl.layers.get_tp_info", lambda: DistributedInfo(0, 2))
    head = ParallelLMHead(5, 2)
    # all_gather concatenates [rank, batch, local_vocab], rather than
    # [batch, rank, local_vocab]. Row 1 exposes an incorrect plain reshape.
    gathered = torch.tensor(
        [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [13.0, 14.0, 0.0], [23.0, 24.0, 0.0]]
    )
    monkeypatch.setattr("minisgl.layers.all_gather", lambda value: gathered)
    assert head(torch.ones(2, 2)).tolist() == [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]


@pytest.mark.parametrize("backend", ["single", "torch", "pynccl"])
def test_collectives_preserve_single_rank_and_select_the_configured_transport(monkeypatch, backend):
    import minisgl.distributed as distributed

    size = 1 if backend == "single" else 2
    monkeypatch.setattr(distributed, "_TP_INFO", DistributedInfo(0, size))

    def reduce(tensor, *args, **kwargs):
        tensor.mul_(size)

    def gather(output, tensor):
        output.copy_(tensor.repeat(size, 1))

    communicator = SimpleNamespace(
        all_reduce=Mock(side_effect=reduce), all_gather=Mock(side_effect=gather)
    )
    monkeypatch.setattr(distributed, "_PYNCCL", communicator if backend == "pynccl" else None)
    torch_reduce, torch_gather = Mock(side_effect=reduce), Mock(side_effect=gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", torch_reduce)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", torch_gather)

    value = torch.ones(2, 3)
    assert distributed.all_reduce(value) is value
    assert torch.equal(value, torch.full((2, 3), float(size)))
    assert torch.equal(distributed.all_gather(value), value.repeat(size, 1))
    if backend == "pynccl":
        communicator.all_reduce.assert_called_once()
        communicator.all_gather.assert_called_once()
        torch_reduce.assert_not_called()
        torch_gather.assert_not_called()
    elif backend == "torch":
        torch_reduce.assert_called_once()
        torch_gather.assert_called_once()
    else:
        torch_reduce.assert_not_called()
        torch_gather.assert_not_called()


def test_uneven_vocabulary_shards_have_matching_padded_shapes():
    weights = torch.arange(10).view(5, 2)
    shards = [_shard_tensor("lm_head.weight", weights, rank, 2, 1) for rank in range(2)]
    assert [tuple(shard.shape) for shard in shards] == [(3, 2), (3, 2)]
    assert torch.equal(torch.cat(shards)[:5], weights)
    assert torch.equal(shards[-1][-1], torch.zeros(2, dtype=weights.dtype))


def test_kv_head_replication_keeps_adjacent_ranks_on_the_same_head():
    weights = torch.arange(16).view(8, 2)
    for rank in range(8):
        shard = _shard_tensor("self_attn.k_proj.weight", weights, rank, 8, 4)
        head = rank // 2
        assert torch.equal(shard, weights[2 * head : 2 * head + 2])


def test_terminal_reply_releases_frontend_state_before_consumer_breaks():
    async def check():
        state = FrontendManager(SimpleNamespace(), None, None)
        uid = state.new_request()
        state.pending[uid].put_nowait(UserReply(uid, "done", True, "stop"))
        replies = state.responses(uid)
        reply = await anext(replies)
        assert reply.finished
        assert uid not in state.pending
        await replies.aclose()

    asyncio.run(check())


def test_idle_tp_primary_wakes_so_other_ranks_never_hit_the_gloo_timeout(monkeypatch):
    # Other ranks sit in a Gloo broadcast. If the primary blocks waiting for users,
    # that broadcast times out and an idle server dies after a minute.
    receiver = SimpleNamespace(socket=Mock(), empty=lambda: True, get_raw=Mock())
    monkeypatch.setattr(torch.distributed, "broadcast", Mock())
    assert _receive_requests(receiver, object(), primary=True, idle=True) == []
    receiver.socket.poll.assert_called_once_with(timeout=1000)
    torch.distributed.broadcast.assert_called_once()
    receiver.get_raw.assert_not_called()


def test_single_gpu_idle_primary_sleeps_until_work_arrives():
    receiver = SimpleNamespace(socket=Mock(), empty=lambda: True, get_raw=Mock())
    assert _receive_requests(receiver, None, primary=True, idle=True) == []
    receiver.socket.poll.assert_called_once_with(timeout=None)


def test_dead_worker_fails_waiting_clients_instead_of_hanging(monkeypatch):
    stop_serving = Mock()
    monkeypatch.setattr("minisgl.server.signal.raise_signal", stop_serving)

    async def check():
        dead = SimpleNamespace(is_alive=lambda: False, exitcode=1)
        frontend = FrontendManager(SimpleNamespace(), None, None, workers=[dead])
        uid = frontend.new_request()
        await frontend.watch_workers()
        with pytest.raises(WorkerUnavailable, match="worker"):
            await anext(frontend.responses(uid))

    asyncio.run(check())
    stop_serving.assert_called_once()


def test_ctrl_c_reaching_the_workers_is_not_reported_as_a_crash(monkeypatch):
    # Clean exits still fail waiters, but must not log a crash or send a second SIGINT.
    stop_serving = Mock()
    monkeypatch.setattr("minisgl.server.signal.raise_signal", stop_serving)

    async def check():
        stopped = SimpleNamespace(is_alive=lambda: False, exitcode=0)
        running = SimpleNamespace(is_alive=lambda: True, exitcode=None)
        frontend = FrontendManager(SimpleNamespace(), None, None, workers=[stopped, running])
        uid = frontend.new_request()
        await frontend.watch_workers()
        assert not frontend.worker_crashed
        with pytest.raises(WorkerUnavailable):
            await anext(frontend.responses(uid))

    asyncio.run(check())
    stop_serving.assert_called_once_with(signal.SIGTERM)


@pytest.mark.parametrize("crashed", [False, True])
def test_http_shutdown_is_bounded_and_only_a_crash_exits_nonzero(monkeypatch, crashed):
    import minisgl.server as server

    ready = Mock()
    ready.get.return_value = "ready"
    context = SimpleNamespace(
        Queue=lambda: ready,
        Process=lambda **_: Mock(exitcode=0, is_alive=Mock(return_value=False)),
    )
    served = {}

    def serve(frontend, **kwargs):
        served.update(kwargs)
        frontend.worker_crashed = crashed

    monkeypatch.setattr(server, "parse_args", lambda _: Config("unused"))
    monkeypatch.setattr(server, "ZmqAsyncQueue", Mock())
    monkeypatch.setattr(server.mp, "get_context", lambda _: context)
    monkeypatch.setattr(server, "create_app", lambda frontend: frontend)
    monkeypatch.setattr(server.uvicorn, "run", serve)
    if crashed:
        with pytest.raises(SystemExit) as exit_info:
            server.launch_server()
        assert exit_info.value.code == 1
    else:
        server.launch_server()
    assert served["timeout_graceful_shutdown"] is not None


def test_worker_failure_rejects_a_request_registered_after_the_monitor_ran(monkeypatch):
    monkeypatch.setattr("minisgl.server.signal.raise_signal", Mock())

    async def check():
        sender = SimpleNamespace(put=AsyncMock())
        frontend = FrontendManager(
            SimpleNamespace(),
            sender,
            None,
            workers=[SimpleNamespace(is_alive=lambda: False, exitcode=1)],
        )
        await frontend.watch_workers()
        with pytest.raises(WorkerUnavailable):
            await frontend.submit("late HTTP body", SamplingParams())
        sender.put.assert_not_awaited()
        assert frontend.pending == {}

    asyncio.run(check())


def test_worker_failure_interrupts_a_submission_waiting_on_a_dead_tokenizer(monkeypatch):
    monkeypatch.setattr("minisgl.server.signal.raise_signal", Mock())

    async def check():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def blocked_send(message):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        frontend = FrontendManager(
            SimpleNamespace(),
            SimpleNamespace(put=blocked_send),
            None,
            workers=[SimpleNamespace(is_alive=lambda: False, exitcode=1)],
        )
        submitting = asyncio.create_task(frontend.submit("hello", SamplingParams()))
        await entered.wait()
        await frontend.watch_workers()
        with pytest.raises(WorkerUnavailable):
            await asyncio.wait_for(submitting, timeout=1)
        assert cancelled.is_set() and frontend.pending == {} and not frontend.background

    asyncio.run(check())


def test_shutdown_cancels_background_sends_before_closing_their_socket():
    async def check():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def blocked_send(message):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def close_sender():
            assert cancelled.is_set()

        sender = SimpleNamespace(put=blocked_send, stop=close_sender)
        receiver = SimpleNamespace(get=asyncio.Queue().get, stop=Mock())
        frontend = FrontendManager(SimpleNamespace(), sender, receiver)
        async with _listen(frontend):
            frontend._send_abort(42)
            await entered.wait()
        assert not frontend.background

    asyncio.run(check())


def test_worker_crash_in_an_open_stream_is_a_server_error():
    async def check():
        import json

        from minisgl.message import ErrorMsg

        frontend = FrontendManager(SimpleNamespace(), None, None)
        uid = frontend.new_request()
        frontend.worker_failed.set()
        frontend.pending[uid].put_nowait(ErrorMsg(uid, "worker died"))
        chunks = [chunk async for chunk in _chat_stream(frontend, uid, "model")]
        assert json.loads(chunks[0][6:])["error"]["type"] == "server_error"
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert frontend.pending == {}

    asyncio.run(check())


@pytest.mark.parametrize(
    "path,stream",
    [("/v1/chat/completions", False), ("/v1/chat/completions", True), ("/generate", True)],
)
def test_late_http_requests_get_503_before_any_stream_headers(path, stream):
    import json

    async def check():
        sender = SimpleNamespace(put=AsyncMock())
        frontend = FrontendManager(SimpleNamespace(model_path="model"), sender, None)
        frontend.worker_failed.set()
        app = create_app(frontend)
        body = json.dumps(
            {"model": "model", "prompt": "hello", "max_tokens": 2, "stream": stream}
        ).encode()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": path,
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "server": ("localhost", 80),
            "client": ("test", 1),
        }
        await app(scope, receive, send)
        assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 503
        assert b"worker exited" in sent[1]["body"]
        sender.put.assert_not_awaited()
        assert frontend.pending == {}

    asyncio.run(check())


def test_api_rejects_zero_output_budget():
    with pytest.raises(ValueError):
        OpenAICompletionRequest(model="test", prompt="hello", max_tokens=0)


@pytest.mark.parametrize(
    "settings",
    [{"n": 2}, {"stop": ["END"]}, {"tools": []}, {"presence_penalty": 1}, {"frequency_penalty": 1}],
)
def test_api_rejects_unimplemented_settings(settings):
    with pytest.raises(ValueError):
        OpenAICompletionRequest(model="test", prompt="hello", **settings)


def test_cancelled_consumer_releases_frontend_state_and_aborts_the_request():
    async def check():
        sender = SimpleNamespace(put=AsyncMock())
        frontend = FrontendManager(SimpleNamespace(), sender, None)
        uid = frontend.new_request()
        replies = frontend.responses(uid)
        waiting = asyncio.create_task(anext(replies))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await asyncio.sleep(0)  # Let the separate abort task send its message.
        assert uid not in frontend.pending
        sender.put.assert_awaited_once_with(AbortMsg(uid))

    asyncio.run(check())


def test_chat_template_token_ids_are_not_tokenized_twice():
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    text = [{"role": "user", "content": "hello"}]
    ids = tokenize(tokenizer, TokenizeMsg(0, text, SamplingParams()))
    assert ids.dtype == torch.int32 and ids.tolist() == [1, 2, 3]
    tokenizer.apply_chat_template.assert_called_once_with(
        text, tokenize=True, add_generation_prompt=True
    )
    tokenizer.encode.assert_not_called()


def test_bad_chat_does_not_kill_the_shared_tokenizer_worker(monkeypatch):
    from minisgl.tokenizer import tokenize_worker

    tokenizer = SimpleNamespace(
        apply_chat_template=Mock(side_effect=ValueError("invalid roles")), encode=lambda text: [7]
    )
    receiver, backend, frontend = Mock(), Mock(), Mock()
    receiver.get.side_effect = [
        TokenizeMsg(0, [{"role": "user"}], SamplingParams()),
        TokenizeMsg(1, "valid", SamplingParams()),
        KeyboardInterrupt(),
    ]
    monkeypatch.setattr("minisgl.tokenizer.load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(
        "minisgl.tokenizer.ZmqQueue", Mock(side_effect=[receiver, backend, frontend])
    )
    tokenize_worker(Config("unused"), Mock())
    error = frontend.put.call_args.args[0].data[0]
    assert error.uid == 0 and "invalid roles" in error.error
    request = backend.put.call_args.args[0].data[0]
    assert request.uid == 1 and request.input_ids.tolist() == [7]
    for socket in (receiver, backend, frontend):
        socket.stop.assert_called_once()


@pytest.mark.parametrize("reason,expected", [("stop", ""), ("length", "99")])
def test_detokenizer_drops_only_a_stopping_eos(reason, expected):
    tokenizer = SimpleNamespace(batch_decode=lambda rows: ["".join(map(str, row)) for row in rows])
    decoder = DetokenizeManager(tokenizer)
    assert decoder.detokenize([DetokenizeMsg(0, 99, True, reason)]) == [expected]
    assert decoder.decode_map == {}


def test_incremental_decoding_waits_for_a_complete_utf8_character():
    texts = {(): "", (1,): "�", (1, 2): "é"}
    tokenizer = SimpleNamespace(batch_decode=lambda rows: [texts[tuple(row)] for row in rows])
    decoder = DetokenizeManager(tokenizer)
    assert decoder.detokenize([DetokenizeMsg(0, 1, False)]) == [""]
    assert decoder.detokenize([DetokenizeMsg(0, 2, True, "length")]) == ["é"]
    assert decoder.decode_map == {}


def test_final_token_flushes_text_held_for_an_incomplete_character():
    # A length limit can end the stream inside a multi-byte character.
    texts = {(): "", (1,): "ab", (1, 2): "ab cd�"}
    tokenizer = SimpleNamespace(batch_decode=lambda rows: [texts[tuple(row)] for row in rows])
    decoder = DetokenizeManager(tokenizer)
    assert decoder.detokenize([DetokenizeMsg(0, 1, False)]) == ["ab"]
    assert decoder.detokenize([DetokenizeMsg(0, 2, True, "length")]) == [" cd�"]
    assert decoder.decode_map == {}


def test_benchmark_starts_before_headers_and_ignores_control_chunks(monkeypatch):
    from minisgl.benchmark import benchmark_one

    ticks, seen = iter([1.0, 2.0, 3.0]), []

    def clock():
        value = next(ticks)
        seen.append(value)
        return value

    monkeypatch.setattr("minisgl.benchmark.time.perf_counter", clock)

    async def response():
        for content in [None, "", "hello", " world", None]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    async def create(**kwargs):
        assert seen == [1.0]  # The HTTP wait belongs inside request latency.
        return response()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    timing = asyncio.run(benchmark_one(client, "hello", 8, "model"))
    assert (timing.started, timing.first_text, timing.finished) == (1.0, 2.0, 3.0)
    assert timing.requested_output_tokens == 8


def test_offline_api_rejects_mismatched_prompt_and_parameter_counts():
    llm = object.__new__(LLM)
    with pytest.raises(ValueError, match="one SamplingParams per prompt"):
        llm.generate(["first", "second"], [SamplingParams()])


def test_config_copies_preserve_revision_and_do_not_share_mutable_state():
    original = SimpleNamespace(_commit_hash="checkpoint-revision", nested={"value": [1]})
    with patch("minisgl.utils._load_hf_config", return_value=original):
        config = cached_load_hf_config("unused")
    config.nested["value"].append(2)
    assert original.nested == {"value": [1]}
    assert config._commit_hash == "checkpoint-revision"


def test_stop_tokens_include_generation_config(tmp_path):
    # Qwen3's config.json names <|im_end|>; only generation_config adds <|endoftext|>.
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [151645, 151643]}')
    config = SimpleNamespace(eos_token_id=151645)
    assert load_eos_token_ids(str(tmp_path), config) == {151645, 151643}


def test_stop_tokens_work_without_generation_config(tmp_path):
    config = SimpleNamespace(eos_token_id=[1, 2])
    assert load_eos_token_ids(str(tmp_path), config) == {1, 2}


def test_legacy_chat_template_loads_from_a_local_snapshot(monkeypatch, tmp_path):
    from minisgl.utils import load_tokenizer

    tokenizer = SimpleNamespace(chat_template=None)
    monkeypatch.setattr("minisgl.utils.AutoTokenizer.from_pretrained", lambda path: tokenizer)
    hub = Mock(side_effect=AssertionError("A local snapshot must not fetch its template from Hub"))
    monkeypatch.setattr("minisgl.utils.hf_hub_download", hub)
    (tmp_path / "chat_template.json").write_text('{"chat_template": "{{ messages }}"}')
    assert load_tokenizer(str(tmp_path)).chat_template == "{{ messages }}"
    hub.assert_not_called()


def test_corrupt_legacy_chat_template_is_not_silently_ignored(monkeypatch, tmp_path):
    import json

    from minisgl.utils import load_tokenizer

    monkeypatch.setattr(
        "minisgl.utils.AutoTokenizer.from_pretrained",
        lambda path: SimpleNamespace(chat_template=None),
    )
    (tmp_path / "chat_template.json").write_text("invalid JSON")
    with pytest.raises(json.JSONDecodeError):
        load_tokenizer(str(tmp_path))


@pytest.mark.parametrize("limit,expected", [(0, []), (1, [1]), (3, [1, 2]), (8, [1, 2, 4, 8])])
def test_cuda_graph_sizes_respect_the_requested_limit(limit, expected):
    assert _determine_cuda_graph_bs(None, limit, 96 << 30, 256) == expected


def test_cuda_graphs_stop_at_the_largest_possible_batch():
    assert _determine_cuda_graph_bs(None, None, 96 << 30, 8) == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "architecture",
    [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "MistralForCausalLM",
        "Mistral3ForConditionalGeneration",
    ],
)
def test_shared_decoder_preserves_checkpoint_keys_and_shapes(monkeypatch, architecture):
    monkeypatch.setattr("minisgl.distributed._TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr("minisgl.models.create_rope", lambda *args: None)
    config = SimpleNamespace(
        architecture=architecture,
        num_layers=2,
        vocab_size=10,
        hidden_size=128,
        intermediate_size=256,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=64,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=128,
        norm_topk_prob=True,
    )
    with torch.device("meta"):
        model = create_model(config)
    expected = {
        "model.embed_tokens.weight": (10, 128),
        "lm_head.weight": (10, 128),
        "model.norm.weight": (128,),
    }
    for i in range(2):
        prefix = f"model.layers.{i}."
        expected.update(
            {
                prefix + name: shape
                for name, shape in {
                    "self_attn.qkv_proj.weight": (256, 128),
                    "self_attn.o_proj.weight": (128, 128),
                    "input_layernorm.weight": (128,),
                    "post_attention_layernorm.weight": (128,),
                }.items()
            }
        )
        if architecture.startswith("Qwen3"):
            expected[prefix + "self_attn.q_norm.weight"] = (64,)
            expected[prefix + "self_attn.k_norm.weight"] = (64,)
        if architecture == "Qwen2ForCausalLM":
            expected[prefix + "self_attn.qkv_proj.bias"] = (256,)
        if architecture == "Qwen3MoeForCausalLM":
            expected[prefix + "mlp.gate.weight"] = (2, 128)
            expected[prefix + "mlp.experts.gate_up_proj"] = (2, 256, 128)
            expected[prefix + "mlp.experts.down_proj"] = (2, 128, 128)
        else:
            expected[prefix + "mlp.gate_up_proj.weight"] = (512, 128)
            expected[prefix + "mlp.down_proj.weight"] = (128, 256)
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == expected
