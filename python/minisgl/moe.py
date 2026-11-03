"""Route tokens to experts, run their feed-forward projections, and combine outputs.

T is the number of tokens, E the experts, k the experts per token, I the intermediate
size, and H the hidden size. Equivalent to this loop, which is too slow to run:

    for each token t, for each of its k experts e, with routing weight w:
        output[t] += w * down[e] @ (silu(gate[e] @ x[t]) * up[e] @ x[t])

Instead, sort the T * k routes by expert, then run one matmul per expert block:

    x [T, H] -> gate_up [T, k, 2 * I] -> activated [T * k, I] -> output [T, k, H] -> sum [T, H]
"""

from __future__ import annotations

from typing import Any

import torch
from minisgl.utils import div_ceil


def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    import triton
    import triton.language as tl
    from minisgl.moe_kernel import fused_moe_kernel

    # C[route] = A[route] @ B[expert].T, for each route in sorted_token_ids.
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **config,
    )


def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    import triton
    from minisgl.moe_kernel import moe_sum_reduce_kernel

    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,  # type: ignore
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,  # type: ignore
    )


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    # Both [T, k]: each token's chosen experts and their routing weights.
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group flattened token/expert routes, padding each expert to whole blocks.

    For topk_ids=[[1, 0], [1, 2]] and block_size=2, the live sorted route IDs
    are [1, 4, 0, 2, 3, 4] and block expert IDs are [0, 1, 2]. Route 4 is
    padding: there are only four real routes. Kernels skip these sentinel IDs.
    """
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def _moe_config(num_tokens: int, num_experts: int) -> dict[str, int]:
    if num_tokens <= num_experts:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    from minisgl.layers import silu_and_mul

    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    num_experts, gate_up_size, _ = w1.shape
    config = _moe_config(num_tokens, num_experts)

    # Reuse storage after the activation consumes gate_up; down projection can
    # overwrite it with per-expert outputs before reducing into hidden_states.
    scratch = torch.empty(
        num_tokens * topk_ids.shape[1] * max(gate_up_size, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    gate_up = scratch[: num_tokens * topk_ids.shape[1] * gate_up_size].view(
        (num_tokens, topk_ids.shape[1], gate_up_size),
    )
    activated = torch.empty(
        (num_tokens * topk_ids.shape[1], gate_up_size // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    expert_output = scratch[: num_tokens * topk_ids.shape[1] * w2.shape[1]].view(
        (num_tokens, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    fused_moe_kernel_triton(
        hidden_states,
        w1,
        gate_up,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # Routing weights apply once, after the down projection.
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
    )
    silu_and_mul(gate_up.view(-1, gate_up_size), activated)
    fused_moe_kernel_triton(
        activated,
        w2,
        expert_output,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,  # activated already has one row per route, not per token.
        config,
        compute_type=compute_type,
    )

    moe_sum_reduce_triton(
        expert_output,
        hidden_states,
    )
    return hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
    )
    return fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)
