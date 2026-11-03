# Follow-up work

- [ ] **Benchmark attention backends on RTX PRO 6000 Blackwell with larger models.**
  Keep `fi` as the working SM120 default. Compare prefill and decode separately,
  including `fi,trtllm` (FA2 prefill, XQA decode), longer contexts, and larger
  batches. Investigate SM120-compatible cuDNN/FMHAv2 prefill before changing the
  default. Validate numerical accuracy, paged KV access, prefix reuse, chunked
  prefill, and CUDA graphs for each candidate.
  Re-run the Qwen3-0.6B timing comparison after the batched TRTLLM query-layout
  fix: the earlier batched timings did not validate token parity. See the
  [upstream discussion](https://github.com/sgl-project/sglang/issues/24633#issuecomment-4579300993)
  for the distinction between unsupported TRT-LLM MLA prefill and alternatives.

- [ ] **Validate TP and PP on real multi-GPU hardware.** Both pass on one GPU with
  ranks sharing it over loopback TCP. Real interconnects, memory limits per GPU, and
  `benchmark/kernel/comm.py` (which needs four GPUs) are untested.

- [ ] **Design Qwen3.8-27B support.** Its hybrid linear/full-attention decoder
  needs recurrent state, new layer and weight mappings, output gates, partial
  RoPE, and compatible configuration/tokenizer handling. Treat multimodal input
  and prefix caching of recurrent state as explicit design work.
