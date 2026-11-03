"""A repeatable single-GPU check: python tests/smoke.py [--page-size 16] [--serial].

Exercise cold/cache-hit prompts, chunked prefill, graph padding, and greedy
generation. Print checkpoint/software revisions and token IDs for comparison.

A cache hit must reproduce the cold distribution, but not always its argmax: only
the uncached suffix is recomputed, so bf16 rounding can differ and break an exact
tie between two logits. Compare logits, which separates that from corrupted KV.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

# Rounding moves a logit near 30 by a step of 0.125, so whole vectors stay well under
# this on average. Wrong KV moves them by whole units; see the guide's parity section.
MAX_MEAN_LOGIT_SHIFT = 0.25


def generate_recording_logits(llm, prompts, settings):
    """Run generate() and return (outputs, {(uid, output_step): logits[vocab]})."""
    import minisgl.engine as engine

    logits_by_step, current = {}, {}
    original_forward, original_sample = engine.Engine.forward_batch, engine.sample

    def forward_batch(self, batch, args):
        current["batch"] = batch
        return original_forward(self, batch, args)

    def sample(logits, args):
        for row, request in enumerate(current["batch"].reqs):
            if not request.is_prefill_chunk:
                step = request.scheduled_len - request.prompt_len - 1
                logits_by_step[request.uid, step] = logits[row].float().cpu()
        return original_sample(logits, args)

    engine.Engine.forward_batch, engine.sample = forward_batch, sample
    try:
        return llm.generate(prompts, settings), logits_by_step
    finally:
        engine.Engine.forward_batch, engine.sample = original_forward, original_sample


def assert_same_distribution(cold, cached):
    """Compare each step up to the first differing token; inputs differ after that."""
    (cold_outputs, cold_logits), (cached_outputs, cached_logits) = cold, cached
    worst = 0.0
    for uid, (a, b) in enumerate(zip(cold_outputs, cached_outputs, strict=True)):
        ids_a, ids_b = a["token_ids"], b["token_ids"]
        same = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), len(ids_a))
        for step in range(min(same + 1, len(ids_a))):
            shift = (cold_logits[uid, step] - cached_logits[uid, step]).abs().mean().item()
            worst = max(worst, shift)
            assert shift < MAX_MEAN_LOGIT_SHIFT, f"KV reuse changed logits: {uid=} {step=} {shift=}"
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--page-size", type=int, choices=[1, 16, 32, 64], default=1)
    parser.add_argument("--serial", action="store_true", help="Disable CPU/GPU overlap")
    parser.add_argument("--attn", default="fi", help="Attention backend to exercise")
    parser.add_argument("--no-graph", action="store_true", help="Execute decode eagerly")
    parser.add_argument("--report", type=Path, help="Save a machine-readable reference report")
    parser.add_argument("--compare", type=Path, help="Compare input/output IDs with a saved report")
    args = parser.parse_args()

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    # Keep a small KV pool so this check does not reserve most of a large GPU.
    # A 128-token prefill budget forces the long prompt into multiple chunks.
    llm = LLM(
        args.model,
        attention_backend=args.attn,
        overlap=not args.serial,
        page_size=args.page_size,
        num_page_override=16384 // args.page_size,
        max_running_req=4,
        max_seq_len_override=2048,
        max_extend_tokens=128,
        cuda_graph_bs=[] if args.no_graph else [1, 4],
        cache_type="radix",
    )
    try:
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in [
                "Reply with the single word HELLO.",
                "Explain KV caching in one sentence.",
                "Explain KV caching in one sentence.",  # Duplicate: both miss, one is freed.
                "Explain KV caching in one sentence. " + "Be precise. " * 100,
            ]
        ]
        settings = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        runs = [generate_recording_logits(llm, prompts, settings) for _ in range(3)]
        outputs = [run[0] for run in runs]
        llm.scheduler.cache_manager.check_integrity()
        worst_shift = assert_same_distribution(runs[0], runs[1])
        # The second and third runs recompute the same suffix from the same KV.
        assert outputs[1] == outputs[2], "Identical cached runs diverged"
        assert all(result["token_ids"] for result in outputs[0])
        assert outputs[0][1] == outputs[0][2], "Duplicate prompts in one batch diverged"
        root = Path(__file__).resolve().parents[1]
        source = hashlib.sha256()
        for path in sorted((root / "python/minisgl").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
        report = {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
            ),
            "source_sha256": source.hexdigest(),
            "model": args.model,
            "model_commit": getattr(llm.scheduler.config.hf_config, "_commit_hash", None),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["transformers", "flashinfer-python", "sgl-kernel", "apache-tvm-ffi"]
            },
            "page_size": llm.scheduler.config.page_size,
            "attention_backend": llm.scheduler.config.attention_backend,
            "cuda_graph_bs": llm.scheduler.config.cuda_graph_bs,
            "serial": args.serial,
            "cold_vs_cached_mean_logit_shift": worst_shift,
            "input_token_ids": prompts,
            "outputs": outputs[0],
        }
        if args.compare:
            reference = json.loads(args.compare.read_text())
            assert report["model_commit"] == reference["model_commit"], (
                "Checkpoint revisions differ"
            )
            assert prompts == reference["input_token_ids"], "Input token IDs differ"
            assert outputs[0] == reference["outputs"], "Generated token IDs/text differ"
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        print(f"PASS: largest cold/cached mean logit shift {worst_shift:.3f}; KV accounting intact")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
