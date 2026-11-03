"""Synthetic offline throughput; edit the workload constants below.

Adapted from https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py
"""

import random
import time

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    random.seed(0)
    num_requests, max_input_length, max_output_length = 256, 1024, 1024
    prompts = [
        [random.randint(0, 10000) for _ in range(random.randint(100, max_input_length))]
        for _ in range(num_requests)
    ]
    settings = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=random.randint(100, max_output_length)
        )
        for _ in prompts
    ]
    with LLM(
        "Qwen/Qwen3-0.6B",
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        page_size=256,
    ) as llm:
        llm.generate([[1, 2, 3]], SamplingParams(max_tokens=2))  # Exclude JIT from timing.
        started = time.perf_counter()
        results = llm.generate(prompts, settings)
        elapsed = time.perf_counter() - started
    tokens = sum(len(result["token_ids"]) for result in results)
    print(f"Output: {tokens} tokens, {elapsed:.2f} s, {tokens / elapsed:.2f} tokens/s")


if __name__ == "__main__":
    main()
