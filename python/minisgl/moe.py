"""Route tokens to experts, run their feed-forward projections, and combine outputs.

T is the number of tokens, E the experts, k the experts per token, I the intermediate
size, and H the hidden size. Equivalent to this loop, which is too slow to run:

    for each token t, for each of its k experts e, with routing weight w:
        output[t] += w * down[e] @ (silu(gate[e] @ x[t]) * up[e] @ x[t])

Instead, sort the T * k routes by expert, then run one matmul per expert block:

    x [T, H] -> gate_up [T, k, 2 * I] -> activated [T * k, I] -> output [T, k, H] -> sum [T, H]

sgl-kernel routes and sorts; the Triton grouped matmul at the end of this file
multiplies each block of routes by its expert's weights.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
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

    # Sum each token's k weighted expert outputs: [T, k, H] -> [T, H].
    return torch.sum(expert_output, dim=1, out=hidden_states)


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


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """Grouped GEMM: A[tokens, K] and B[experts, N, K] -> C[tokens, topk, N].

    Each program handles a block of token/expert routes for one expert.
    First GEMM: route_id // top_k selects the original token. After activation,
    inputs are expanded per route, so the second GEMM uses top_k=1. Padding
    routes are masked; output indices always refer to the original route order.
    """
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
