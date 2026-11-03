"""Check the Triton KV store against PyTorch indexing, then time both in CUDA graphs."""

from __future__ import annotations

import torch
from minisgl.kvcache import store_kv
from triton.testing import do_bench_cudagraph


def main():
    head_size, num_slots = 128, 1 << 20
    k_cache = torch.randn((num_slots, head_size), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    for num_tokens in [2**n for n in range(16)]:
        # Slots must be unique: duplicates make the expected result ambiguous.
        slots = torch.randperm(num_slots, device="cuda")[:num_tokens].to(torch.int32)
        qkv = torch.randn((num_tokens, head_size * 4), device="cuda", dtype=torch.float16)
        k, v = qkv[:, :head_size], qkv[:, head_size : 2 * head_size]  # Strided, as in the model.
        store_kv(k_cache, v_cache, slots, k, v)
        assert torch.equal(k_cache[slots], k) and torch.equal(v_cache[slots], v), num_tokens

        def indexing(index=slots.long()):
            k_cache[index] = k
            v_cache[index] = v

        triton_ms = do_bench_cudagraph(lambda: store_kv(k_cache, v_cache, slots, k, v))
        torch_ms = do_bench_cudagraph(indexing)
        written = 2 * num_tokens * head_size * k_cache.element_size()  # K and V bytes
        print(
            f"T={num_tokens:6d}: Triton {triton_ms * 1e3:7.2f} us "
            f"({written / triton_ms / 1e6:6.1f} GB/s), indexing {torch_ms * 1e3:7.2f} us"
        )


if __name__ == "__main__":
    main()
