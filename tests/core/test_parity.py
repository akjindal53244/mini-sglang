"""The parity gate must fail on corruption, even after greedy sequences diverge."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "parity_helpers", Path(__file__).parents[1] / "parity.py"
)
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


@pytest.fixture
def reference():
    return {
        "format_version": parity.FORMAT_VERSION,
        "engine": "hf",
        "model": "same-model",
        "model_commit": "same-revision",
        "dtype": "bfloat16",
        "prompts": [[1, 2]],
        "continuations": [[3, 4, 5]],
        "greedy": [[3, 4, 5]],
        "logprobs": [[-1.0] * 4],
    }


def compare(tmp_path, reference, candidate, max_error=0.75):
    paths = [tmp_path / "hf.json", tmp_path / "mini.json"]
    for path, report in zip(paths, (reference, candidate)):
        path.write_text(json.dumps(report))
    return parity.compare(list(map(str, paths)), max_median=0.06, max_error=max_error)


def test_asymmetric_confidence_error_is_not_filtered_out(tmp_path, reference):
    candidate = copy.deepcopy(reference)
    reference["logprobs"][0][0] = -10.0
    candidate["logprobs"][0][0] = -0.01
    assert compare(tmp_path, reference, candidate) == 1


def test_median_keeps_positions_where_only_one_engine_is_certain(tmp_path, reference):
    # Below every maximum limit, so only the median can see this. With max(a, b)
    # the filter drops all four positions and the median is vacuously zero.
    reference["logprobs"] = [[-0.15] * 4]
    candidate = copy.deepcopy(reference)
    candidate["logprobs"] = [[-0.01] * 4]
    assert compare(tmp_path, reference, candidate) == 1


def test_decode_error_too_small_for_the_prefill_limit_still_fails(tmp_path, reference):
    # Measured: an off-by-one decode RoPE position moved Qwen3-0.6B by only 0.246.
    candidate = copy.deepcopy(reference)
    candidate["logprobs"][0][-1] += 0.25
    assert compare(tmp_path, reference, candidate) == 1


def test_decode_limit_skips_the_first_continuation_token(tmp_path, reference):
    # forced[0] is scored by the last prefill position, under the prefill limit.
    candidate = copy.deepcopy(reference)
    candidate["logprobs"][0][1] += 0.25
    assert parity.logprob_errors(candidate, reference)[2] == [0.0, 0.0]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("bad_side", ["reference", "candidate"])
@pytest.mark.parametrize("position", [0, -1])  # max() and median ignore NaN only by position.
def test_nonfinite_scores_fail_on_either_side(tmp_path, reference, bad_value, bad_side, position):
    candidate = copy.deepcopy(reference)
    (reference if bad_side == "reference" else candidate)["logprobs"][0][position] = bad_value
    assert compare(tmp_path, reference, candidate) == 1


def test_greedy_divergence_cannot_hide_decode_errors(tmp_path, reference):
    candidate = copy.deepcopy(reference)
    candidate["greedy"] = [[9, 9, 9]]
    candidate["logprobs"] = [[-1.0, -10.0, -10.0, -10.0]]
    assert compare(tmp_path, reference, candidate) == 1


def test_greedy_divergence_alone_does_not_fail_identical_fixed_sequence_scores(tmp_path, reference):
    candidate = copy.deepcopy(reference)
    candidate["greedy"] = [[9, 9, 9]]
    assert compare(tmp_path, reference, candidate) == 0


def test_maximum_error_includes_mutually_certain_positions(tmp_path, reference):
    reference["logprobs"] = [[-0.001] * 4]
    candidate = copy.deepcopy(reference)
    candidate["logprobs"][0][-1] = -0.02
    assert compare(tmp_path, reference, candidate, max_error=0.01) == 1


def test_identical_all_certain_scores_have_a_defined_median(tmp_path, reference):
    reference["logprobs"] = [[0.0] * 4]
    assert compare(tmp_path, reference, copy.deepcopy(reference)) == 0


@pytest.mark.parametrize("downloaded", ["same-revision", "wrong-revision"])
def test_dump_pins_and_records_the_downloaded_snapshot_revision(
    monkeypatch, tmp_path, reference, downloaded
):
    reference_path, output = tmp_path / "reference.json", tmp_path / "output.json"
    reference_path.write_text(json.dumps(reference))
    calls = []

    def snapshot(model, **kwargs):
        calls.append((model, kwargs["revision"]))
        return f"/cache/snapshots/{downloaded}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(
        parity,
        "dump_hf",
        lambda *args: {
            key: copy.deepcopy(reference[key]) for key in ("greedy", "continuations", "logprobs")
        },
    )
    args = SimpleNamespace(
        reference=str(reference_path),
        model="same-model",
        dtype=None,
        engine="hf",
        options="{}",
        out=str(output),
    )
    if downloaded == "same-revision":
        parity.dump(args)
        assert json.loads(output.read_text())["model_commit"] == "same-revision"
    else:
        with pytest.raises(ValueError, match="revision"):
            parity.dump(args)
        assert not output.exists()
    assert calls == [("same-model", "same-revision")]


@pytest.mark.parametrize(
    "key,value",
    [
        ("continuations", [[3, 4, 6]]),
        ("prompts", [[1, 7]]),
        ("dtype", "float16"),
        ("model", "other-model"),
        ("model_commit", "other-revision"),
        ("logprobs", [[-1.0] * 3]),
        ("logprobs", [[-1.0] * 5]),
        ("logprobs", [[]]),
        ("prompts", []),
        ("format_version", 1),
    ],
)
def test_incomparable_or_incomplete_reports_fail(tmp_path, reference, key, value):
    candidate = copy.deepcopy(reference)
    candidate[key] = value
    assert compare(tmp_path, reference, candidate) == 1


def test_reports_truncated_identically_still_fail(tmp_path, reference):
    reference["logprobs"] = [[-1.0] * 3]
    assert compare(tmp_path, reference, copy.deepcopy(reference)) == 1


def test_worker_exit_does_not_hide_a_just_flushed_queue_result(monkeypatch):
    import multiprocessing as mp
    import queue
    from unittest.mock import Mock

    report = {"result": "last worker's flushed report"}
    results = Mock()
    results.get.side_effect = [queue.Empty, report]
    workers = [Mock(exitcode=0), Mock(exitcode=0)]
    for worker in workers:
        worker.is_alive.return_value = False
    context = SimpleNamespace(Queue=lambda: results, Process=Mock(side_effect=workers))
    monkeypatch.setattr(mp, "get_context", lambda method: context)
    assert parity.dump_minisgl_ranks("unused", [[1]], "float16", [[2]], "pp", 2) == report
    results.close.assert_called_once()
    results.join_thread.assert_called_once()


def test_sglang_adapter_scores_the_reference_continuation_and_uses_requested_dtype(monkeypatch):
    import sys

    calls = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == "float16"

        def generate(self, *, input_ids, **kwargs):
            if not kwargs.get("return_logprob"):
                return {"output_ids": [99] * parity.MAX_NEW_TOKENS}
            calls.append(input_ids)
            return {
                "meta_info": {
                    "input_token_logprobs": [(None, input_ids[0], None)]
                    + [(-1.0, token, None) for token in input_ids[1:]]
                }
            }

        def shutdown(self):
            pass

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    result = parity.dump_sglang("unused", [[1, 2]], "float16", [[3, 4]])
    assert calls == [[1, 2, 3, 4]]
    assert result["continuations"] == [[3, 4]]
    assert result["greedy"] == [[99] * parity.MAX_NEW_TOKENS]
    assert result["logprobs"] == [[-1.0] * 3]


def test_vllm_adapter_scores_the_reference_continuation_and_uses_requested_dtype(monkeypatch):
    import sys

    calls = []

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs["dtype"] == "float16"

        def generate(self, requests, params, **kwargs):
            if not getattr(params, "prompt_logprobs", None):
                return [
                    SimpleNamespace(
                        outputs=[SimpleNamespace(token_ids=[99] * parity.MAX_NEW_TOKENS)]
                    )
                ]
            sequences = [request["prompt_token_ids"] for request in requests]
            calls.extend(sequences)
            return [
                SimpleNamespace(
                    prompt_logprobs=[None]
                    + [{token: SimpleNamespace(logprob=-1.0)} for token in ids[1:]]
                )
                for ids in sequences
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: SimpleNamespace(**kw)),
    )
    result = parity.dump_vllm("unused", [[1, 2]], "float16", [[3, 4]])
    assert calls == [[1, 2, 3, 4]]
    assert result["continuations"] == [[3, 4]]
    assert result["greedy"] == [[99] * parity.MAX_NEW_TOKENS]
    assert result["logprobs"] == [[-1.0] * 3]
