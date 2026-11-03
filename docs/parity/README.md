# Parity reports

Reports from [tests/parity.py](../../tests/parity.py), measured in bf16 on one RTX PRO 6000
Blackwell (SM120). Each report records its checkpoint revision, package versions,
options, prompt IDs, and the HF continuation every engine scored. Re-check any set:

```bash
cd docs/parity/qwen3-0.6b && python ../../../tests/parity.py compare hf.json minisgl*.json sglang.json vllm.json
```

The reference engine is always `hf.json`; the other files in a directory score its IDs.

| Directory | Contents |
|---|---|
| `qwen3-0.6b/` | This engine, SGLang 0.5.23, vLLM 0.24.0; TP=2 and PP=2 with 128-token chunks, graphs, and overlap; `padding-*` forces one request into a four-row decode graph |
| `qwen2.5-0.5b/`, `llama-3.2-1b/` | This engine, SGLang, vLLM |
| `qwen3-30b-a3b/` | A trained MoE: this engine (single, TP=2, PP=2), SGLang, vLLM. Compare with `--max-error 3 --max-decode-error 0.6`; the guide explains why |
| `random-weights/` | Tiny random Qwen3-MoE, Qwen3-MoE with `norm_topk_prob`, Mistral, and YaRN checkpoints. They check code paths only. The checkpoints are not in the repository; [make_tiny_checkpoints.py](../../tests/make_tiny_checkpoints.py) writes equivalent ones (new random weights, so dump a fresh HF reference) |
| `negative-controls/` | Deliberately broken runs: decode RoPE off by one (fails on Qwen3-0.6B, undetectable on the 30B MoE), YaRN without its amplitude, and MoE without top-k renormalization |

TP and PP ran as ranks sharing one GPU over loopback; real multi-GPU runs are not
covered. See the [validation workflow](../learning-guide.md#validation-workflow) for
how to read the numbers.
