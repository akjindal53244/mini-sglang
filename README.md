# Mini-SGLang

A small LLM inference engine for learning how serving works, from a chat message
to GPU execution and back. This educational revamp builds on
[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang), with the
readability goals of [nanoGPT](https://github.com/karpathy/nanoGPT) and
[nanochat](https://github.com/karpathy/nanochat).

The package is **17 Python files and no native code**. The transformer is an
ordinary PyTorch `nn.Module`. One scheduler handles both offline and online
inference. The optimizations remain visible: continuous batching, chunked prefill,
paged KV storage, radix prefix reuse, CUDA graphs, CPU/GPU overlap, tensor and
pipeline parallelism, and routed experts.

## Run it

The reference environment is **Linux, Python 3.12, PyTorch 2.9.1 with CUDA 12.8**.
You need an NVIDIA GPU, a compatible driver, the CUDA toolkit (`nvcc`), and a C++
compiler: FlashInfer compiles its kernels on first use, and the first launch can
take about a minute.

```bash
git clone https://github.com/akjindal53244/mini-sglang
cd mini-sglang
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]' -c constraints.txt --torch-backend cu128
python -m minisgl --model Qwen/Qwen3-0.6B --shell
```

Try `What is 2+2? /no_think`. Qwen3 otherwise may produce a long thinking trace.
`/reset` clears the conversation; `/exit` closes the workers.

This checkout is tested on **RTX PRO 6000 Blackwell (SM120)**. `--attn auto` selects
FlashInfer for both phases there. Backend performance tuning remains in
[TODO.md](TODO.md). [constraints.txt](constraints.txt) records the tested direct
dependency versions; it is a reference environment, not a complete platform lock.

## Follow a single request

Start without graphs, overlap, or prefix reuse. Then enable each optimization
and compare the output IDs. The same scheduler executes both versions.

```python
from minisgl.core import SamplingParams
from minisgl.llm import LLM

with LLM(
    "Qwen/Qwen3-0.6B",
    num_page_override=16384,  # A small, fixed KV pool for this example.
    max_seq_len_override=2048,
    max_running_req=4,
    cuda_graph_bs=[],  # No CUDA graphs: faster startup for one prompt.
    overlap=False,
    cache_type="naive",
) as llm:
    ids = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = llm.generate([ids], SamplingParams(temperature=0, max_tokens=32))
    print(result[0]["text"])
```

String inputs to `LLM.generate` are raw prompts. Supply chat-template IDs, as
above, when matching the shell or HTTP behavior.

Read [models.py](python/minisgl/models.py) first if you want to begin with the
familiar transformer. To trace execution, follow
[llm.py](python/minisgl/llm.py) → [scheduler.py](python/minisgl/scheduler.py) →
[engine.py](python/minisgl/engine.py) → the model. The
[walkthrough guide](docs/learning-guide.md) maps all files, explains the state and
memory invariants, and suggests exercises from the original commit history.

## Serve chat over HTTP

```bash
python -m minisgl --model Qwen/Qwen3-0.6B \
    --max-running-requests 8 --cuda-graph-max-bs 8
```

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-0.6B", "messages":[{"role":"user","content":"What is 2+2? /no_think"}], "temperature":0, "max_tokens":64}'
```

Add `"stream":true` for SSE output. `/v1/models` lists the loaded model. This is a
small chat-completions API: it supports text messages, temperature, top-k, top-p,
output budgets, and cancellation. Send `"prompt"` instead of `"messages"` to skip
the chat template. `max_tokens` defaults to 1024, since admission reserves KV for
the whole budget. A worker failure rejects new requests with HTTP 503, fails
waiting requests, and shuts the server down with a nonzero exit code. Unsupported
settings such as tools, stop strings, multiple completions, and nonzero penalties
are rejected. Usage token counts are currently omitted.

## Check correctness

```bash
python -m pytest -q
python tests/smoke.py --report /tmp/minisgl-reference.json
python tests/smoke.py --serial --no-graph --page-size 16 \
    --compare /tmp/minisgl-reference.json
```

The GPU smoke workload repeats prompts, forces chunked prefill, pads decode batches,
checks that a cache hit preserves the logit distribution, and checks KV accounting.
Reports include checkpoint and software revisions, source hashes, input IDs, and
output IDs. Use a local Hub snapshot path with `--model` to freeze the checkpoint.
CPU tests need no model download.

On this SM120 setup, the hybrid backend can also be checked with:

```bash
python tests/smoke.py --attn fi,trtllm --page-size 64 \
    --compare /tmp/minisgl-reference.json
```

[tests/parity.py](tests/parity.py) compares this engine with Transformers, SGLang,
and vLLM on identical prompt **and continuation** IDs. Mini-SGLang forces the
reference continuation through prefill/decode; greedy text is a separate diagnostic.
The checker rejects nonfinite or incomplete scores, compares every position, and
applies a separate, tighter limit to decode.
Create an HF report, then pass it to each engine with `--reference hf.json`. Older
parity reports must be regenerated. Its docstring has the setup commands, and the
[validation workflow](docs/learning-guide.md#validation-workflow) has the results.

## Scope and switches

Run `python -m minisgl --help` for the CLI. Useful switches are `--attn`,
`--page-size`, `--cache naive`, `--disable-overlap`, `--cuda-graph-max-bs 0`,
`--max-prefill-length`, `--tp`, and `--pp`.

The implemented decoder families are Llama, Qwen2/2.5, Qwen3, Qwen3-MoE, and the
Mistral text decoder. The exact architecture table lives in `models.py`.
Hybrid/recurrent attention, sliding-window attention, vision input, quantization,
and speculative decoding are outside this implementation; a checkpoint with an
active sliding window is rejected at load. Qwen3.8-27B support is follow-up work.

`--tp N` splits each layer across N ranks; `--pp N` splits the decoder into N stages
of layers. They are alternatives, not a grid. With fewer GPUs than ranks, ranks
share a GPU and NCCL runs over loopback TCP: slow, but the real code path, so both
can be studied on one GPU. Only that single-GPU mode has been tested so far.

Optional tools have separate dependencies:

```bash
uv pip install -e '.[benchmark]' -c constraints.txt  # OpenAI client + Parquet reader
```

See [benchmark/README.md](benchmark/README.md) for workloads and timing definitions,
and [TODO.md](TODO.md) for backend tuning and reference-engine validation.

Mini-SGLang retains its upstream [MIT license](LICENSE).
