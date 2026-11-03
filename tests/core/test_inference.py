"""CPU regressions for state, cache ownership, checkpoint layout, and API cleanup.

No downloaded model is needed.
"""

import asyncio
import signal
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from minisgl.config import Config, ModelConfig
from minisgl.core import Batch, Request, SamplingParams
from minisgl.distributed import DistributedInfo, stage_layers
from minisgl.engine import (
    ForwardOutput,
    largest_prefill,
    prepare_sampling_args,
    resolve_engine_config,
    sample,
)
from minisgl.graph import GraphRunner, cuda_graph_sizes
from minisgl.kvcache import KVCacheManager
from minisgl.layers import ParallelLMHead
from minisgl.llm import LLM
from minisgl.message import AbortMsg, DetokenizeMsg, TokenizeMsg, UserReply
from minisgl.models import _stage_key, create_model, load_weights
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
        with (
            patch("minisgl.engine.compute_capability", return_value=capability),
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


def test_tp_and_pp_are_alternatives():
    with pytest.raises(ValueError, match="not both"):
        Config("unused", tp_info=DistributedInfo(0, 2), pp_info=DistributedInfo(0, 2))


def test_active_sliding_windows_are_rejected_rather_than_attending_too_far():
    from transformers import MistralConfig, Qwen2Config

    small = {"hidden_size": 128, "num_attention_heads": 2, "max_position_embeddings": 64}
    with pytest.raises(ValueError, match="Sliding-window"):
        ModelConfig.from_hf(MistralConfig(sliding_window=8, **small))
    ModelConfig.from_hf(MistralConfig(sliding_window=None, **small))
    ModelConfig.from_hf(MistralConfig(sliding_window=64, **small))  # Covers the whole context.
    ModelConfig.from_hf(Qwen2Config(sliding_window=8, use_sliding_window=False, **small))


@pytest.mark.parametrize(
    "max_running_req,context,expected",
    [(32, 64, [63] * 32), (3, 64, [63, 63, 63]), (1, 64, [63]), (3, 8192, [683, 683, 682])],
)
def test_warmup_covers_the_largest_batch_the_scheduler_can_form(max_running_req, context, expected):
    # Several short prompts can fill the prefill budget when one alone cannot.
    config = Config("unused", max_running_req=max_running_req, max_extend_tokens=2048)
    object.__setattr__(config, "model_config", SimpleNamespace(max_position=context))
    assert largest_prefill(config) == expected


@pytest.mark.parametrize("num_layers,stages", [(28, 2), (28, 3), (7, 3), (4, 4)])
def test_pipeline_stages_cover_every_layer_exactly_once(num_layers, stages):
    ranges = [stage_layers(num_layers, DistributedInfo(s, stages)) for s in range(stages)]
    assert [layer for r in ranges for layer in r] == list(range(num_layers))
    assert max(map(len, ranges)) - min(map(len, ranges)) <= 1


def test_pipeline_stages_load_only_their_own_weights_under_local_names():
    config = SimpleNamespace(num_layers=4, tie_word_embeddings=True)
    stages = [DistributedInfo(0, 2), DistributedInfo(1, 2)]
    expected = {
        # Tied: the last stage has no embedding, so its head loads this tensor.
        "model.embed_tokens.weight": ["model.embed_tokens.weight", "lm_head.weight"],
        "model.layers.1.mlp.up_proj.weight": ["model.layers.1.mlp.up_proj.weight", None],
        "model.layers.3.mlp.up_proj.weight": [None, "model.layers.1.mlp.up_proj.weight"],
        "model.norm.weight": [None, "model.norm.weight"],
        "lm_head.weight": [None, None],  # A tied head never loads a second copy.
    }
    for key, owners in expected.items():
        assert [_stage_key(key, config, pp) for pp in stages] == owners
    config.tie_word_embeddings = False
    assert _stage_key("model.embed_tokens.weight", config, stages[1]) is None
    assert _stage_key("lm_head.weight", config, stages[1]) == "lm_head.weight"


def make_request(output_len=2):
    return Request(42, [1, 2, 3], SamplingParams(max_tokens=output_len), table_idx=0)


def make_scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler.eos_token_ids = {99}
    scheduler.running = []
    scheduler.free_rows = []
    scheduler.pending = []
    scheduler.in_flight = None
    scheduler.cache_manager = SimpleNamespace(defer_page_frees=nullcontext, cache_req=Mock())
    return scheduler


def deliver(scheduler, request, token, phase="decode"):
    output = ForwardOutput(None, torch.tensor([token]), SimpleNamespace(synchronize=lambda: None))
    scheduled = SimpleNamespace(batch=Batch([request], phase), emits=[True])
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
    assert request.input_ids[-2:] == [99, 99]


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
    scheduler.running.append(request)
    request.advance_after_forward()
    scheduler._handle_message(AbortMsg(uid=request.uid))
    assert deliver(scheduler, request, 10) == []
    assert scheduler.free_rows == [0]


@pytest.mark.parametrize("rank", [0, 1])
def test_only_the_last_tp_rank_samples_and_every_rank_receives_its_tokens(monkeypatch, rank):
    import minisgl.engine as engine

    eng = object.__new__(engine.Engine)
    eng.config = Config("unused", tp_info=DistributedInfo(rank, 2))
    eng.device, eng.stream = torch.device("cpu"), "engine stream"
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "engine stream")
    eng.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda batch: False)
    eng.model, eng.attn_backend = Mock(return_value=torch.zeros(1, 4)), None
    sampler, broadcast = Mock(return_value=torch.tensor([3])), Mock()
    monkeypatch.setattr(engine, "sample", sampler)
    monkeypatch.setattr(engine, "broadcast", broadcast)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None))
    eng.forward_batch(Batch([make_request()], "decode"), None)
    assert sampler.called == (rank == 1)
    assert broadcast.call_args.kwargs["src"] == 1


def test_empty_output_budget_is_rejected():
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)


def test_trtllm_workspace_starts_with_zeroed_synchronization_counters(monkeypatch):
    from minisgl.attention import TRTLLMBackend

    empty = torch.empty

    def dirty_empty(*args, **kwargs):
        # Simulate recycled memory containing stale semaphore values.
        return empty(*args, **kwargs).fill_(1)

    monkeypatch.setattr(torch, "empty", dirty_empty)
    backend = TRTLLMBackend(
        SimpleNamespace(head_dim=64),
        SimpleNamespace(device=torch.device("cpu")),
        torch.zeros((1, 1), dtype=torch.int32),
        page_size=64,
    )
    assert torch.count_nonzero(backend.workspace_buffer).item() == 0


@pytest.mark.parametrize("ranks_per_gpu", [1, 2])
def test_automatic_kv_sizing_shares_device_memory_once(ranks_per_gpu):
    from minisgl.engine import Engine

    config = Config("unused", tp_info=DistributedInfo(0, ranks_per_gpu), memory_ratio=0.9)
    model = SimpleNamespace(head_dim=128, num_kv_heads=4, num_layers=48)
    object.__setattr__(config, "model_config", model)
    engine = object.__new__(Engine)
    engine.ranks_per_gpu, engine.dtype = ranks_per_gpu, config.dtype
    initial_free, current_free, activations = 94 << 30, 34 << 30, 1 << 30
    engine._sync_get_memory = lambda: current_free

    pages = engine._determine_num_pages(initial_free, activations, config)
    page_bytes = 2 * 48 * (4 // ranks_per_gpu) * 128 * config.dtype.itemsize
    # All ranks together must leave room for the ratio's headroom and each one's
    # activation peak. The remaining capacity should miss that bound by <1 page/rank.
    remaining = current_free - pages * page_bytes * ranks_per_gpu
    reserved = (1 - config.memory_ratio) * initial_free + activations * ranks_per_gpu
    assert 0 <= remaining - reserved < page_bytes * ranks_per_gpu


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


@pytest.mark.parametrize("size", [1, 2])
def test_collectives_reduce_in_place_and_skip_a_single_rank(monkeypatch, size):
    import minisgl.distributed as distributed

    monkeypatch.setattr(distributed, "_TP_INFO", DistributedInfo(0, size))
    torch_reduce = Mock(side_effect=lambda tensor: tensor.mul_(size))
    torch_gather = Mock(side_effect=lambda output, tensor: output.copy_(tensor.repeat(size, 1)))
    monkeypatch.setattr(torch.distributed, "all_reduce", torch_reduce)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", torch_gather)

    value = torch.ones(2, 3)
    assert distributed.all_reduce(value) is value
    assert torch.equal(value, torch.full((2, 3), float(size)))
    assert torch.equal(distributed.all_gather(value), value.repeat(size, 1))
    assert torch_reduce.called == torch_gather.called == (size > 1)


def test_uneven_vocabulary_shards_are_padded_with_zero_rows(monkeypatch):
    from minisgl.layers import load_vocab_shard

    weights = torch.arange(10.0).view(5, 2)
    shards = []
    for rank in range(2):
        monkeypatch.setattr(
            "minisgl.layers.get_tp_info", lambda rank=rank: DistributedInfo(rank, 2)
        )
        param = torch.nn.Parameter(torch.full((3, 2), -1.0))
        assert load_vocab_shard(param, weights) == param.numel()
        shards.append(param.data)
    assert torch.equal(torch.cat(shards)[:5], weights)
    assert torch.equal(shards[-1][-1], torch.zeros(2))


def test_packed_loaders_place_each_rank_shard(monkeypatch):
    from minisgl.models import GatedFeedForward, RotaryAttention

    config = SimpleNamespace(
        num_qo_heads=8, num_kv_heads=4, head_dim=2, hidden_size=4, intermediate_size=4,
        hidden_act="silu", rms_norm_eps=1e-6,
    )  # fmt: skip
    k_proj = torch.arange(32.0).view(8, 4)  # Four KV heads of two rows each.
    gate, up = torch.arange(16.0).view(4, 4), -torch.arange(16.0).view(4, 4)
    for rank in range(8):
        monkeypatch.setattr("minisgl.distributed._TP_INFO", DistributedInfo(rank, 8))
        attention = RotaryAttention(config, 0, attention_bias=False, qk_norm=False)
        weight = attention.qkv_proj.weight  # [Q 2 | K 2 | V 2] rows: one head each.
        weight.weight_loader(weight, k_proj, "k")
        head = rank // 2  # Eight ranks share four KV heads: adjacent ranks replicate one.
        assert torch.equal(weight.data[2:4], k_proj[2 * head : 2 * head + 2])
    monkeypatch.setattr("minisgl.distributed._TP_INFO", DistributedInfo(1, 2))
    mlp = GatedFeedForward(config)
    weight = mlp.gate_up_proj.weight  # [local gate 2 | local up 2] rows on rank 1.
    weight.weight_loader(weight, gate, 0)
    weight.weight_loader(weight, up, 1)
    assert torch.equal(weight.data, torch.cat([gate[2:], up[2:]]))


def _tiny_llama(tie):
    return SimpleNamespace(
        architecture="LlamaForCausalLM", num_layers=1, vocab_size=8, hidden_size=4,
        intermediate_size=8, num_qo_heads=2, num_kv_heads=1, head_dim=2, rms_norm_eps=1e-6,
        hidden_act="silu", tie_word_embeddings=tie, num_experts=0, num_experts_per_tok=0,
        moe_intermediate_size=0, norm_topk_prob=False,
    )  # fmt: skip


def _llama_checkpoint(tie):
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "model.norm.weight": torch.randn(4),
        "model.layers.0.input_layernorm.weight": torch.randn(4),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(2, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(4, 4),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 8),
    }
    if not tie:
        tensors["lm_head.weight"] = torch.randn(8, 4)
    return tensors


@pytest.mark.parametrize("tie", [True, False])
def test_weight_loading_fills_every_parameter_or_fails(monkeypatch, tmp_path, tie):
    from safetensors.torch import save_file

    monkeypatch.setattr("minisgl.distributed._TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr("minisgl.models.RotaryEmbedding", lambda config: None)
    tensors = _llama_checkpoint(tie)
    save_file(tensors, tmp_path / "model.safetensors")
    model = create_model(_tiny_llama(tie))
    load_weights(model, str(tmp_path), _tiny_llama(tie), DistributedInfo(0, 1))
    head = tensors["model.embed_tokens.weight" if tie else "lm_head.weight"]
    assert torch.equal(model.lm_head.weight, head)
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tie
    packed = model.model.layers[0].self_attn.qkv_proj.weight
    assert torch.equal(packed[4:6], tensors["model.layers.0.self_attn.k_proj.weight"])

    def load():
        model = create_model(_tiny_llama(tie))
        load_weights(model, str(tmp_path), _tiny_llama(tie), DistributedInfo(0, 1))

    gate = "model.layers.0.mlp.gate_proj.weight"
    del tensors["model.layers.0.mlp.up_proj.weight"]  # Half of gate_up stays unloaded.
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="did not fill"):
        load()
    # A repeated gate slice in another file must not stand in for the missing up slice.
    save_file({gate: tensors[gate]}, tmp_path / "model-extra.safetensors")
    with pytest.raises(ValueError, match="loaded twice"):
        load()


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


@pytest.mark.parametrize("stream", [False, True])
def test_late_http_requests_get_503_before_any_stream_headers(stream):
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
            "path": "/v1/chat/completions",
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
    assert ids == [1, 2, 3]
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
    assert request.uid == 1 and request.input_ids == [7]
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
    assert cuda_graph_sizes(limit) == expected


def test_cuda_graphs_stop_at_the_largest_possible_batch(monkeypatch):
    monkeypatch.setattr(GraphRunner, "_capture_graphs", lambda *args: None)
    runner = GraphRunner(None, None, None, None, None, 8, 64, 10, None)
    assert runner.graph_bs_list == [1, 2, 4, 8]


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
    monkeypatch.setattr("minisgl.models.RotaryEmbedding", lambda config: None)
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
