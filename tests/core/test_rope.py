"""Compare the complete RoPE tables, including YaRN amplitude, with Transformers."""

import copy

import pytest
import torch
from minisgl.layers import _get_rope
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

HEAD_DIM, BASE, MAX_POSITION = 64, 10000.0, 8192
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048}


def hf_tables(rope_scaling):
    config = LlamaConfig(
        hidden_size=HEAD_DIM,
        num_attention_heads=1,
        rope_theta=BASE,
        max_position_embeddings=MAX_POSITION,
        rope_scaling=copy.deepcopy(rope_scaling),
    )
    rotary = LlamaRotaryEmbedding(config, device="cpu")
    cos, sin = rotary(torch.empty(1, 1, HEAD_DIM), torch.arange(MAX_POSITION).unsqueeze(0))
    # HF repeats each frequency across the two halves. Our kernel stores one
    # copy of cos followed by one copy of sin, each with head_dim // 2 columns.
    return cos[0, :, : HEAD_DIM // 2], sin[0, :, : HEAD_DIM // 2]


@pytest.mark.parametrize(
    "rope_scaling",
    [
        None,
        {"rope_type": "default"},
        {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        YARN,
        {**YARN, "attention_factor": 0.7},
        {**YARN, "mscale": 0.5, "mscale_all_dim": 1.5},
        {**YARN, "truncate": False},
        {"rope_type": "yarn", "factor": 1.0},
        {"rope_type": "yarn", "factor": 4.0},
        {**YARN, "beta_slow": 1e-6},
    ],
    ids=[
        "none",
        "default",
        "llama3",
        "yarn",
        "explicit-amplitude",
        "mscale-ratio",
        "fractional-range",
        "no-extension",
        "implicit-original-context",
        "wide-correction-range",
    ],
)
def test_rope_tables_match_transformers(monkeypatch, rope_scaling):
    # Construction imports the CUDA function; this test only reads the CPU table.
    monkeypatch.setattr("flashinfer.apply_rope_with_cos_sin_cache_inplace", None, raising=False)
    rope = _get_rope(HEAD_DIM, HEAD_DIM, MAX_POSITION, BASE, rope_scaling)
    cos, sin = rope._cos_sin_cache.chunk(2, dim=-1)
    expected_cos, expected_sin = hf_tables(rope_scaling)
    torch.testing.assert_close(cos, expected_cos, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(sin, expected_sin, atol=1e-4, rtol=1e-5)


def test_rope_check_rejects_a_small_base_error(monkeypatch):
    monkeypatch.setattr("flashinfer.apply_rope_with_cos_sin_cache_inplace", None, raising=False)
    rope = _get_rope(HEAD_DIM, HEAD_DIM, MAX_POSITION, BASE * 1.05, None)
    expected_cos, _ = hf_tables(None)
    cos, _ = rope._cos_sin_cache.chunk(2, dim=-1)
    assert not torch.allclose(cos, expected_cos, atol=2e-3)


def test_yarn_check_rejects_missing_amplitude(monkeypatch):
    # The old test compared inverse frequencies alone and passed this exact bug.
    monkeypatch.setattr("flashinfer.apply_rope_with_cos_sin_cache_inplace", None, raising=False)
    unit_rope = _get_rope(HEAD_DIM, HEAD_DIM, MAX_POSITION, BASE, {**YARN, "attention_factor": 1.0})
    expected_cos, _ = hf_tables(YARN)
    assert expected_cos[0, 0].item() == pytest.approx(1.138629436)
    assert unit_rope._cos_sin_cache[0, 0].item() == 1.0
    assert not torch.allclose(unit_rope._cos_sin_cache[:, : HEAD_DIM // 2], expected_cos, atol=2e-3)
