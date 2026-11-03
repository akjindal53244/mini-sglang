"""Offline throughput on English/Chinese prompts from the first WildChat shard."""

import shutil
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from minisgl.core import SamplingParams
from minisgl.llm import LLM

SHARD = "train-00000-of-00086.parquet"
URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/" + SHARD


def prompt_ids(tokenizer):
    path = Path(__file__).parent / SHARD
    if not path.exists():
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL, timeout=300) as response, path.open("wb") as output:
            shutil.copyfileobj(response, output)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=256, columns=["conversation"]):
        for conversation in batch.to_pydict()["conversation"]:
            turn = next((turn for turn in conversation or [] if turn.get("role") == "user"), None)
            if not turn or turn.get("language") not in {"English", "Chinese"}:
                continue
            text = (turn.get("content") or "").strip()
            if not text or turn.get("redacted") or turn.get("toxic"):
                continue
            yield tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
            )


def length_statistics(label, lengths):
    lengths = sorted(lengths)
    percentiles = [lengths[min(int(len(lengths) * p), len(lengths) - 1)] for p in (0.5, 0.9, 0.99)]
    print(f"{label}: min={lengths[0]}, p50/p90/p99={percentiles}, max={lengths[-1]}")


def main():
    from itertools import islice

    num_requests, max_output_tokens = 256, 4096
    with LLM("Qwen/Qwen3-0.6B") as llm:
        prompts = list(islice(prompt_ids(llm.tokenizer), num_requests))
        if not prompts:
            raise ValueError("No matching prompts in the shard")
        settings = SamplingParams(temperature=0.6, max_tokens=max_output_tokens)
        warmup = llm.generate(prompts[:1], settings)[0]
        print("Warmup:", warmup["text"][:200])
        started = time.perf_counter()
        results = llm.generate(prompts, settings)
        elapsed = time.perf_counter() - started
    lengths = [len(result["token_ids"]) for result in results]
    length_statistics("Input tokens", [len(ids) for ids in prompts])
    length_statistics("Output tokens", lengths)
    print(
        f"{len(results)} requests, {sum(lengths)} output tokens, {elapsed:.2f} s, "
        f"{sum(lengths) / elapsed:.2f} tokens/s"
    )


if __name__ == "__main__":
    main()
