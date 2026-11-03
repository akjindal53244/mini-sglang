r"""Score one fixed token sequence in every engine, independently of greedy output.

Create an HF reference, then reuse BOTH its prompt IDs and continuation IDs:

    python tests/parity.py dump --engine hf --model Qwen/Qwen3-0.6B --out hf.json
    python tests/parity.py dump --engine minisgl --model Qwen/Qwen3-0.6B \
        --reference hf.json --out mini.json
    python tests/parity.py compare hf.json mini.json

Reference engines need separate environments. The flags skip optional JIT steps
requiring a newer nvcc than this checkout's CUDA 12.8 toolkit:

    uv venv .parity/sglang --python 3.12
    uv pip install --python .parity/sglang/bin/python 'sglang==0.5.23' \
        --prerelease=allow --torch-backend cu130
    uv venv .parity/vllm --python 3.12
    uv pip install --python .parity/vllm/bin/python 'vllm==0.24.0' \
        --torch-backend cu130
    SGL_ENABLE_JIT_DEEPGEMM=0 .parity/sglang/bin/python tests/parity.py dump \
        --engine sglang --model Qwen/Qwen3-0.6B --reference hf.json --out sglang.json
    VLLM_USE_FLASHINFER_SAMPLER=0 .parity/vllm/bin/python tests/parity.py dump \
        --engine vllm --model Qwen/Qwen3-0.6B --reference hf.json --out vllm.json
    python tests/parity.py compare hf.json mini.json sglang.json vllm.json

For minisgl, add --options '{"tp": 2}' or '{"pp": 2}' to exercise distributed
execution. Graphs, overlap, page size, and prefill budget are also LLM options.
Prefix caching stays disabled so every prompt position is actually computed;
tests/smoke.py checks cache reuse separately.

Mini-SGLang scores prompt positions during prefill and forces the reference IDs
through real decode. The reference engines score the same full sequence. Greedy
IDs are reported separately: a rounding-induced argmax change cannot shorten the
numerical check. Reports from the older, truncated comparison must be regenerated.
Keep imports independent of minisgl so this script also runs in reference venvs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import queue
import signal
import statistics
import sys
import time
from pathlib import Path

CHAT = [
    "What is 2+2? Answer briefly.",
    "Explain KV caching in one sentence.",
    "Write a haiku about GPUs.",
    "List the planets of the solar system in order.",
]
RAW = ["The capital of France is", "def fibonacci(n):\n    if n < 2:\n        return n\n"]
MAX_NEW_TOKENS = 32
CERTAIN_LOGPROB = -0.05
FORMAT_VERSION = 2


def long_text() -> str:
    # Low RoPE frequencies need long-range evidence; short prompts hide base errors.
    facts = " ".join(f"Item {i} has code {i * 7919 % 10007}." for i in range(220))
    return f"{facts} Question: what code does item 5 have? Answer: Item 5 has code"


def build_prompts(tokenizer) -> list[list[int]]:
    prompts = [tokenizer.encode(text) for text in [*RAW, long_text()]]
    if tokenizer.chat_template:
        for text in CHAT:
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
                )
            )
    return prompts


def dump_hf(model_path, prompts, dtype, continuations=None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    # Load straight to the GPU: a 30B checkpoint does not fit in host RAM as well.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, dtype), device_map="cuda"
    ).eval()
    greedy, scores = [], []
    with torch.inference_mode():
        for prompt in prompts:
            generated = []
            for _ in range(MAX_NEW_TOKENS):
                ids = torch.tensor([prompt + generated], device="cuda")
                generated.append(int(model(ids).logits[0, -1].argmax()))
            greedy.append(generated)
        continuations = continuations if continuations is not None else greedy
        for prompt, forced in zip(prompts, continuations, strict=True):
            sequence = torch.tensor([prompt + forced], device="cuda")
            # Input excludes the last target, so every continuation ID is scored.
            logits = model(sequence[:, :-1]).logits[0].float().log_softmax(-1)
            picked = logits.gather(1, sequence[0, 1:, None])[:, 0]
            scores.append(picked.cpu().tolist())
    return {"greedy": greedy, "continuations": continuations, "logprobs": scores}


def dump_minisgl(model_path, prompts, dtype, continuations=None, **settings) -> dict:
    import minisgl.engine as engine
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    options = {
        "num_page_override": 32768,
        "max_running_req": 8,
        "max_seq_len_override": 8192,
        "cuda_graph_bs": [],
        "overlap": False,
        "cache_type": "naive",
    }
    options.update(settings)
    if options["cache_type"] != "naive":
        raise ValueError("Parity needs cache_type='naive' to score every prompt position")
    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)
    with LLM(model_path, dtype=getattr(torch, dtype), **options) as llm:
        greedy = [result["token_ids"] for result in llm.generate(prompts, params)]
        if any(len(ids) != MAX_NEW_TOKENS for ids in greedy):
            raise ValueError("Generation was clipped: increase the context limit or KV pool")
        continuations = continuations if continuations is not None else greedy
        model = llm.scheduler.engine.model
        prompt_scores, output_scores, reports = [], [], []
        current = None
        prompt, forced = [], []
        original_forward, original_sample = engine.Engine.forward_batch, engine.sample

        def forward_batch(self, batch, args):
            nonlocal current
            current = batch  # Also runs for decode graphs, outside the captured forward.
            return original_forward(self, batch, args)

        def score_prefill(_module, inputs, hidden):
            batch = inputs[0]
            if hidden is None or not batch.is_prefill:
                return
            request = batch.reqs[0]  # Scoring sends one prompt at a time.
            end = min(request.scheduled_len, len(prompt) - 1)
            count = end - request.cached_len
            if count > 0:
                # Gather on GPU; don't copy a [prompt_len, vocab] matrix to CPU.
                logits = model.lm_head(hidden[:count]).float().log_softmax(-1)
                targets = torch.tensor(
                    prompt[request.cached_len + 1 : end + 1], device=logits.device
                )
                prompt_scores.extend(logits.gather(1, targets[:, None])[:, 0].cpu().tolist())

        def teacher_forced_sample(logits, _args):
            if current.reqs[0].is_prefill_chunk:
                return logits.new_zeros((1,), dtype=torch.int32)  # Discarded, not a continuation.
            token = forced[len(output_scores)]
            output_scores.append(logits[0].float().log_softmax(-1)[token].item())
            return logits.new_full((1,), token, dtype=torch.int32)

        hook = model.model.register_forward_hook(score_prefill)
        engine.Engine.forward_batch, engine.sample = forward_batch, teacher_forced_sample
        try:
            for prompt, forced in zip(prompts, continuations, strict=True):
                prompt_scores.clear(), output_scores.clear()
                llm.generate([prompt], SamplingParams(max_tokens=len(forced), ignore_eos=True))
                if model.lm_head is None:  # Only the final pipeline stage has scores.
                    continue
                if len(prompt_scores) != len(prompt) - 1 or len(output_scores) != len(forced):
                    raise ValueError(
                        "Incomplete scoring: check the context limit and prefill budget"
                    )
                reports.append(prompt_scores + output_scores)
        finally:
            engine.Engine.forward_batch, engine.sample = original_forward, original_sample
            hook.remove()
    return {"greedy": greedy, "continuations": continuations, "logprobs": reports}


def _minisgl_rank(rank, size, kind, model_path, prompts, dtype, continuations, settings, results):
    from minisgl.distributed import DistributedInfo

    report = dump_minisgl(
        model_path,
        prompts,
        dtype,
        continuations,
        **{f"{kind}_info": DistributedInfo(rank, size)},
        **settings,
    )
    if rank == (size - 1 if kind == "pp" else 0):
        results.put(report)


def dump_minisgl_ranks(model_path, prompts, dtype, continuations, kind, size, **settings):
    import multiprocessing as mp

    context = mp.get_context("spawn")
    results, workers = context.Queue(), []
    try:
        for rank in range(size):
            worker = context.Process(
                target=_minisgl_rank,
                args=(
                    rank,
                    size,
                    kind,
                    model_path,
                    prompts,
                    dtype,
                    continuations,
                    settings,
                    results,
                ),
            )
            worker.start()
            workers.append(worker)
        deadline = time.monotonic() + 900
        while True:
            try:
                report = results.get(timeout=1)
                break
            except queue.Empty:
                if any(worker.exitcode not in (None, 0) for worker in workers):
                    raise RuntimeError("A parity worker failed; see its traceback above")
                if all(worker.exitcode is not None for worker in workers):
                    # Queue's feeder can finish just after the timed get. Exited
                    # producers have flushed it, so check once more before failing.
                    try:
                        report = results.get(timeout=1)
                        break
                    except queue.Empty:
                        raise RuntimeError("All parity workers exited without a report") from None
                if time.monotonic() >= deadline:
                    raise TimeoutError("Parity workers did not finish within 15 minutes")
        for worker in workers:
            worker.join(timeout=10)
            if worker.exitcode != 0:
                raise RuntimeError("A parity worker failed or did not shut down")
        return report
    finally:
        for worker in workers:
            if worker.is_alive():
                try:
                    os.kill(worker.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
        results.close()
        results.join_thread()


def dump_sglang(model_path, prompts, dtype, continuations=None) -> dict:
    import sglang

    engine = sglang.Engine(
        model_path=model_path,
        dtype=dtype,
        mem_fraction_static=0.85,  # Engines run one at a time; 30B MoE needs ~61 GB.
        disable_cuda_graph=True,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        random_seed=0,
        log_level="error",
    )
    greedy, scores = [], []
    try:
        params = {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS, "ignore_eos": True}
        for prompt in prompts:
            greedy.append(engine.generate(input_ids=prompt, sampling_params=params)["output_ids"])
        continuations = continuations if continuations is not None else greedy
        for prompt, forced in zip(prompts, continuations, strict=True):
            sequence = prompt + forced
            result = engine.generate(
                input_ids=sequence,
                sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                return_logprob=True,
                logprob_start_len=0,
            )
            entries = result["meta_info"]["input_token_logprobs"]
            scores.append([entry[0] for entry in entries[1:]])
    finally:
        engine.shutdown()
    return {"greedy": greedy, "continuations": continuations, "logprobs": scores}


def dump_vllm(model_path, prompts, dtype, continuations=None) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype=dtype,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        moe_backend="triton",  # Its FlashInfer CUTLASS MoE JITs for SM120 and needs nvcc >=12.9.
        seed=0,
    )
    generated = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompts],
        SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True),
        use_tqdm=False,
    )
    greedy = [list(output.outputs[0].token_ids) for output in generated]
    continuations = continuations if continuations is not None else greedy
    sequences = [prompt + forced for prompt, forced in zip(prompts, continuations, strict=True)]
    scored = llm.generate(
        [{"prompt_token_ids": ids} for ids in sequences],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=1),
        use_tqdm=False,
    )
    scores = []
    for sequence, output in zip(sequences, scored, strict=True):
        scores.append(
            [
                entry[token].logprob
                for entry, token in zip(output.prompt_logprobs[1:], sequence[1:], strict=True)
            ]
        )
    return {"greedy": greedy, "continuations": continuations, "logprobs": scores}


def validate_report(report: dict) -> None:
    if report.get("format_version") != FORMAT_VERSION:
        raise ValueError("Legacy or unsupported report: regenerate it using --reference")
    if report.get("dtype") not in ("bfloat16", "float16") or not report.get("model"):
        raise ValueError("Report must identify its model and dtype")
    prompts, forced = report["prompts"], report["continuations"]
    if not prompts or not all(
        len(report[key]) == len(prompts) for key in ("continuations", "greedy", "logprobs")
    ):
        raise ValueError(
            "Report must have one continuation, greedy output, and score row per prompt"
        )
    for prompt, continuation, greedy, scores in zip(
        prompts, forced, report["greedy"], report["logprobs"], strict=True
    ):
        for ids in (prompt, continuation, greedy):
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Token sequences must contain nonnegative integer IDs")
        if len(scores) != len(prompt) - 1 + len(continuation):
            raise ValueError(
                "Missing scores: every prompt target and continuation token must be scored"
            )
        if any(not math.isfinite(score) or score > 1e-5 for score in scores):
            raise ValueError("Logprobs must be finite and nonpositive")


def logprob_errors(report: dict, reference: dict) -> tuple[list[float], list[float], list[float]]:
    """Errors at all positions, at uncertain positions, and at decode positions."""
    validate_report(reference)
    validate_report(report)
    for key in ("model", "model_commit", "dtype", "prompts", "continuations"):
        if report.get(key) != reference.get(key):
            raise ValueError(
                f"Different {key}: reports must score the same checkpoint, dtype, and IDs"
            )
    errors, uncertain, decode = [], [], []
    rows = zip(report["logprobs"], reference["logprobs"], reference["continuations"], strict=True)
    for ours, theirs, forced in rows:
        # The final prefill position scores forced[0]; each later token needs a decode step.
        first_decode = len(ours) - len(forced) + 1
        for position, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            error = abs(a - b)
            errors.append(error)
            if position >= first_decode:
                decode.append(error)
            # Keep asymmetric confidence errors. NaNs have already failed validation.
            if min(a, b) < CERTAIN_LOGPROB:
                uncertain.append(error)
    return errors, uncertain, decode


def compare(
    paths: list[str], max_median: float, max_error: float, max_decode_error: float = 0.2
) -> int:
    if len(paths) < 2:
        raise ValueError("Provide a reference report and at least one candidate")
    limits = (max_median, max_error, max_decode_error)
    if any(not math.isfinite(limit) or limit < 0 for limit in limits):
        raise ValueError("Comparison limits must be finite and nonnegative")
    reference = json.loads(Path(paths[0]).read_text())
    try:
        validate_report(reference)
    except (KeyError, TypeError, ValueError) as error:
        print(f"{paths[0]}: FAIL {error}")
        return 1
    print(f"reference: {reference['engine']} ({paths[0]})")
    failed = False
    for path in paths[1:]:
        report = json.loads(Path(path).read_text())
        try:
            errors, uncertain, decode = logprob_errors(report, reference)
            median, worst = statistics.median(uncertain) if uncertain else 0.0, max(errors)
            worst_decode = max(decode, default=0.0)
            passed = (
                median <= max_median and worst <= max_error and worst_decode <= max_decode_error
            )
            greedy = sum(a == b for a, b in zip(report["greedy"], reference["greedy"], strict=True))
            print(
                f"{report['engine']:8s} {'PASS' if passed else 'FAIL'}: "
                f"median |delta logprob|={median:.4f} ({len(uncertain)} uncertain positions), "
                f"max={worst:.3f} ({len(errors)} total positions), "
                f"decode max={worst_decode:.3f} ({len(decode)} positions); "
                f"greedy identical {greedy}/{len(reference['greedy'])}"
            )
            failed |= not passed
        except (KeyError, TypeError, ValueError) as error:
            print(f"{path}: FAIL {error}")
            failed = True
    return int(failed)


def dump(args) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if reference:
        validate_report(reference)
        if args.model != reference["model"]:
            raise ValueError("--model must match the reference model")
    dtype = args.dtype or (reference["dtype"] if reference else "bfloat16")
    if reference and dtype != reference["dtype"]:
        raise ValueError("--dtype must match the reference dtype")
    revision = reference.get("model_commit") if reference else None
    model_path = args.model
    if not Path(model_path).is_dir():
        # Freeze weights, config, and tokenizer together, even if the Hub name moves.
        model_path = snapshot_download(
            args.model,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"],
        )
        # The Hub returns snapshots/<commit>. Transformers 5 no longer retains
        # _commit_hash on its config, so identify the actual frozen files here.
        resolved_revision = Path(model_path).name
        if revision is not None and revision != resolved_revision:
            raise ValueError("Downloaded checkpoint does not match the reference revision")
        revision = resolved_revision
    prompts = (
        reference["prompts"]
        if reference
        else build_prompts(AutoTokenizer.from_pretrained(model_path))
    )
    continuations = reference["continuations"] if reference else None
    if args.engine == "minisgl":
        options = json.loads(args.options)
        if "pp" in options and "tp" in options:
            raise ValueError("Choose TP or PP, not both")
        kind = "pp" if "pp" in options else "tp"
        size = options.pop(kind, 1)
        if type(size) is not int or size < 1:
            raise ValueError("Parallel size must be a positive integer")
        report = (
            dump_minisgl_ranks(model_path, prompts, dtype, continuations, kind, size, **options)
            if size > 1
            else dump_minisgl(model_path, prompts, dtype, continuations, **options)
        )
    else:
        report = {"hf": dump_hf, "sglang": dump_sglang, "vllm": dump_vllm}[args.engine](
            model_path, prompts, dtype, continuations
        )
    packages = {}
    for name in ("torch", "transformers", "flashinfer-python", "sglang", "vllm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report.update(
        format_version=FORMAT_VERSION,
        engine=args.engine,
        model=args.model,
        model_commit=revision,
        dtype=dtype,
        prompts=prompts,
        packages=packages,
        options=json.loads(args.options),
    )
    validate_report(report)
    Path(args.out).write_text(json.dumps(report, allow_nan=False) + "\n")
    print(f"{args.engine}: wrote {args.out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dumper = commands.add_parser("dump")
    dumper.add_argument("--engine", choices=["hf", "minisgl", "sglang", "vllm"], required=True)
    dumper.add_argument("--model", required=True)
    dumper.add_argument("--out", required=True)
    dumper.add_argument(
        "--reference", "--prompts-from", help="Reuse reference prompts AND continuations"
    )
    dumper.add_argument("--options", default="{}", help="minisgl LLM options as JSON")
    dumper.add_argument(
        "--dtype", choices=["bfloat16", "float16"], help="Default: reference dtype, or bfloat16"
    )
    comparer = commands.add_parser("compare")
    comparer.add_argument("reports", nargs="+")
    # Diagnostic limits for the measured corpus; see the guide for results.
    # The worst error includes all positions; only the median skips mutually certain ones.
    # Long-prompt prefill sets --max-error, so decode needs its own, tighter limit.
    comparer.add_argument("--max-median", type=float, default=0.06)
    comparer.add_argument("--max-error", type=float, default=0.75)
    comparer.add_argument("--max-decode-error", type=float, default=0.2)
    args = parser.parse_args()
    try:
        if args.command == "dump":
            dump(args)
            return 0
        return compare(args.reports, args.max_median, args.max_error, args.max_decode_error)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
