"""Optional timing helpers. Inference does not import this module.

HTTP streams contain text chunks, not necessarily one token per event. Report
time to first text and request latency; do not count role/finish events as tokens.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from itertools import islice

from minisgl.utils import init_logger
from tqdm import tqdm

logger = init_logger(__name__)


def perf_cuda(function, *, repetitions=10, cuda_graph_repetitions=10, init_stream=True):
    import torch

    stream = torch.cuda.Stream() if init_stream else torch.cuda.current_stream()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        function()
        count = cuda_graph_repetitions or 1
        if cuda_graph_repetitions:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(count):
                    function()
            run = graph.replay
        else:
            run = function
        run()
        start.record()
        for _ in range(repetitions):
            run()
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / (count * repetitions)


def compare_memory_kernel_perf(
    *, baseline, our_impl, memory_footprint, description="", extra_kwargs=None
):
    durations = [perf_cuda(function, **(extra_kwargs or {})) for function in (baseline, our_impl)]
    bandwidths = [memory_footprint / (milliseconds * 1e6) for milliseconds in durations]
    logger.info(
        "%s baseline %.3f ms / %.1f GB/s; kernel %.3f ms / %.1f GB/s",
        description,
        durations[0],
        bandwidths[0],
        durations[1],
        bandwidths[1],
    )
    return tuple(bandwidths)


@dataclass(frozen=True)
class BenchmarkTrace:
    timestamp: float
    message: str
    output_length: int
    input_length: int | None = None


@dataclass(frozen=True)
class RequestTiming:
    started: float
    first_text: float | None
    finished: float
    requested_output_tokens: int


def generate_prompt(tokenizer, n: int) -> str:
    # Decoding random IDs and re-encoding can change token boundaries. Adjust
    # the IDs until the *text* round-trips to the requested synthetic length.
    ids = [random.randint(0, tokenizer.vocab_size // 2) for _ in range(n)]
    for _ in range(64):
        prompt = tokenizer.decode(ids)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) == n:
            return prompt
        ids = ids[:n] + [
            random.randint(0, tokenizer.vocab_size // 2) for _ in range(max(0, n - len(ids)))
        ]
    raise ValueError("Could not generate a prompt of the requested token length")


async def benchmark_one(client, prompt: str, output_length: int, model: str) -> RequestTiming:
    # Start BEFORE awaiting HTTP headers, which may already include prefill.
    started, first_text = time.perf_counter(), None
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=output_length,
        temperature=0,
        extra_body={"ignore_eos": True, "top_k": 1},
    )
    async for chunk in response:
        if first_text is None and chunk.choices and chunk.choices[0].delta.content:
            first_text = time.perf_counter()
    return RequestTiming(started, first_text, time.perf_counter(), output_length)


async def benchmark_trace(client, traces: list[BenchmarkTrace], model: str, *, pbar=True):
    if not traces:
        return []
    start, first = time.perf_counter(), min(trace.timestamp for trace in traces)
    with tqdm(total=len(traces), desc="Requests", disable=not pbar) as progress:

        async def run(trace):
            await asyncio.sleep(max(0, start + trace.timestamp - first - time.perf_counter()))
            result = await benchmark_one(client, trace.message, trace.output_length, model)
            progress.update()
            return result

        return await asyncio.gather(*(run(trace) for trace in traces))


async def benchmark_one_batch(client, prompts, output_lengths, model, *, pbar=True):
    if isinstance(output_lengths, int):
        output_lengths = [output_lengths] * len(prompts)
    traces = [
        BenchmarkTrace(0, prompt, length)
        for prompt, length in zip(prompts, output_lengths, strict=True)
    ]
    return await benchmark_trace(client, traces, model, pbar=pbar)


def process_benchmark_results(results: list[RequestTiming]):
    if not results:
        return []
    elapsed = max(result.finished for result in results) - min(result.started for result in results)

    def statistics(label, values):
        if not values:
            return
        values = sorted(values)
        logger.info(
            "%s: mean %.1f ms; p50 %.1f ms; p90 %.1f ms; p99 %.1f ms",
            label,
            sum(values) / len(values),
            *(values[min(int(len(values) * p), len(values) - 1)] for p in (0.5, 0.9, 0.99)),
        )

    statistics(
        "First text",
        [(r.first_text - r.started) * 1000 for r in results if r.first_text is not None],
    )
    statistics("Request latency", [(r.finished - r.started) * 1000 for r in results])
    logger.info(
        "%d requests in %.2f s: %.2f requests/s", len(results), elapsed, len(results) / elapsed
    )
    # This is an explicit workload assumption, not a count of streamed chunks.
    # Keep requested lengths inside context limits when comparing engines.
    requested = sum(result.requested_output_tokens for result in results)
    logger.info(
        "Requested output budget: %d tokens (%.1f requested tokens/s)",
        requested,
        requested / elapsed,
    )
    return [asdict(result) for result in results]


def read_qwen_trace(file_path, tokenizer, n=None, shared_prefix=False):
    with open(file_path) as file:
        records = [json.loads(line) for line in islice(file, n)]
    if not records:
        return []
    shared_ids = None
    if shared_prefix:
        prompt = generate_prompt(tokenizer, max(record["input_length"] for record in records))
        shared_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # This replays arrival times and lengths, not the private original text.
    # Shared synthetic prefixes must not masquerade as the trace's cache locality.
    return [
        BenchmarkTrace(
            record["timestamp"],
            (
                tokenizer.decode(shared_ids[: record["input_length"]])
                if shared_ids is not None
                else generate_prompt(tokenizer, record["input_length"])
            ),
            record["output_length"],
            record["input_length"],
        )
        for record in records
    ]


def scale_traces(traces, scale):
    first = min(trace.timestamp for trace in traces)
    return sorted(
        [
            BenchmarkTrace(
                (trace.timestamp - first) * scale,
                trace.message,
                trace.output_length,
                trace.input_length,
            )
            for trace in traces
        ],
        key=lambda trace: trace.timestamp,
    )


async def get_model_name(client) -> str:
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available")
