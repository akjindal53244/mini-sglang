"""Write the tiny random checkpoints behind docs/parity/random-weights.

Random weights suffice for parity: every engine must compute the same function of
the same numbers. These cover paths no small trained checkpoint exercises: routed
experts with and without top-k renormalization, the Mistral decoder, and YaRN in
float16. Qwen3's tokenizer makes tests/parity.py's default prompts apply.

    python tests/make_tiny_checkpoints.py /tmp/tiny
    python tests/parity.py dump --engine hf --model /tmp/tiny/moe --out moe-hf.json
    python tests/parity.py dump --engine minisgl --model /tmp/tiny/moe \
        --reference moe-hf.json --out moe-minisgl.json
    python tests/parity.py compare moe-hf.json moe-minisgl.json
"""

import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
    Qwen3MoeConfig,
)


def main(out: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    common = {
        "vocab_size": len(tokenizer),
        "hidden_size": 512,
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "intermediate_size": 1024,
        "max_position_embeddings": 8192,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tie_word_embeddings": False,
    }
    experts = {"num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 256}
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048}
    configs = {
        "moe": (Qwen3MoeConfig(**common, **experts, norm_topk_prob=False), torch.bfloat16),
        "moe_norm": (Qwen3MoeConfig(**common, **experts, norm_topk_prob=True), torch.bfloat16),
        "mistral": (MistralConfig(**common, sliding_window=None), torch.bfloat16),
        # Larger weights sharpen attention, so a wrong rotation shows in the logits.
        "yarn": (LlamaConfig(**common, rope_scaling=yarn, initializer_range=0.15), torch.float16),
    }
    for name, (config, dtype) in configs.items():
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        model.save_pretrained(out / name)
        tokenizer.save_pretrained(out / name)
        print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
