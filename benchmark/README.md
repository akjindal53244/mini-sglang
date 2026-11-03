# Benchmarks

These scripts are experiments, separate from the default CPU test suite. Install
optional clients/readers with `uv pip install -e '.[benchmark]' -c constraints.txt`.
Validate outputs before comparing speed; use `tests/smoke.py` to record a workload
and its environment.

## Offline workloads

```bash
python benchmark/offline/bench.py
python benchmark/offline/bench_wildchat.py
```

The first generates synthetic token IDs for Qwen3-0.6B. The second downloads the
first WildChat-4.8M Parquet shard and selects English/Chinese prompts. Review the
constants in each script to choose request counts and lengths. Throughput counts
returned output IDs; EOS that terminates generation is excluded. Warmup is outside
the timed region; the WildChat run can reuse its warmup prompt's prefix.

## Online workloads

Launch the server first, then run a client:

```bash
python -m minisgl --model Qwen/Qwen3-0.6B
python benchmark/online/bench_simple.py
```

Both clients import their HTTP timing helpers from `client.py` beside them.
`bench_simple.py` sends a concurrent synthetic batch. `bench_qwen.py` downloads
Qwen's anonymous usage trace and replays arrival times and requested lengths.
It creates synthetic text, so chat-template overhead and tokenization can change
actual input lengths. Its shared-prefix option creates artificial cache locality:
use `--cache naive` on the server for that workload, or disable `shared_prefix`
when studying radix caching. Do not interpret it as the original private prompts.

The client clock starts **before** awaiting response headers. Results report
time to first nonempty text, full request latency, requests/s, and requested
output-budget tokens/s. SSE events are text/control chunks, not a reliable count
of generated tokens. Budget throughput assumes requests complete their budgets;
context clipping or an error invalidates that assumption. These scripts do not
establish parity or reproduce upstream published benchmark charts.

## Custom kernel measurement

```bash
python benchmark/kernel/store.py
```

The script checks the Triton KV store against PyTorch indexing, then times both
inside CUDA graphs, as decode runs them. Kernel timings omit model scheduling,
HTTP overhead, and startup/JIT compilation.
