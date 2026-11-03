"""Time a concurrent synthetic chat batch against a server on port 1919."""

import asyncio
import random

from client import (  # benchmark/online/client.py
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from openai import AsyncOpenAI
from transformers import AutoTokenizer


async def main():
    random.seed(42)
    batch_size, max_input_length, max_output_length = 64, 8192, 1024
    async with AsyncOpenAI(base_url="http://127.0.0.1:1919/v1", api_key="dummy") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        await benchmark_one(client, "hello", 2, model)  # Connection check and warmup.
        prompts = [
            generate_prompt(tokenizer, random.randint(1, max_input_length))
            for _ in range(batch_size)
        ]
        lengths = [random.randint(16, max_output_length) for _ in prompts]
        results = await benchmark_one_batch(client, prompts, lengths, model)
        process_benchmark_results(results)


if __name__ == "__main__":
    asyncio.run(main())
