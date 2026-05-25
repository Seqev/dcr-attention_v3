"""
Pass 1 — Projection kernel for top-K Q-axis attention.

Per kernel_spec.md §4.4 Option B + §9.2 GQA layout:
- Grid: (B, H_kv); each program computes scores for n_per_kv query heads
- Streams K_cache from HBM in tiles of B_BLOCK keys
- §3 dtype boundaries: K bf16→fp32 in SRAM, accumulator fp32, output bf16

Note on tl.dot: Triton 3.x requires M,N ≥ 16 for WMMA/TC. For Llama-3.2-1B,
n_per_kv=4 < 16, so we use element-wise multiply + tl.sum per head instead of
tl.dot([N_PER_KV, D], [D, B_BLOCK]). Output is mathematically identical.
"""

from __future__ import annotations

import torch

try:
    import triton                          # type: ignore[import-not-found]
    import triton.language as tl           # type: ignore[import-not-found]
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _pass1_projection_kernel(
        # Pointers
        K_ptr,          # [BH_kv, N, D] bf16  (flattened batch-head)
        u_Q_ptr,        # [BH, D] fp32         (flattened batch-head, H = H_kv * N_PER_KV)
        scores_ptr,     # [BH, N] bf16         (flattened batch-head, H query heads)
        # Strides (flat layout: dim 0 = BH_kv or BH, dim 1 = N, dim 2 = D)
        K_stride_bh, K_stride_n, K_stride_d,
        uQ_stride_bh, uQ_stride_d,
        s_stride_bh, s_stride_n,
        # Shape
        N,                                  # sequence length (runtime)
        H_kv,                               # number of KV heads (runtime)
        D:          tl.constexpr,           # head dim = 64 for Llama-3.2-1B
        N_PER_KV:   tl.constexpr,           # query heads per KV head = 4
        B_BLOCK:    tl.constexpr,           # tile size = 64
    ):
        """
        Per-program work: scores for one (b, h_kv) → N_PER_KV score rows of length N.

        §3 dtype boundaries:
        - K loaded as bf16, promoted to fp32 in registers
        - u_Q already fp32 (computed host-side in _compute_u_Q)
        - scores stored as bf16 (demoted at write boundary, per §3)
        """
        pid_bh_kv = tl.program_id(0)       # flat (b * H_kv + h_kv) index

        # Decode b and h_kv; h_start = index of first query head in this KV group
        h_kv = pid_bh_kv % H_kv
        b    = pid_bh_kv // H_kv
        h_start = b * (H_kv * N_PER_KV) + h_kv * N_PER_KV   # index into flat u_Q / scores

        # Pre-load u_Q for all N_PER_KV heads in this group
        # Each u_Q_h is [D] fp32, loaded once per kernel (reused over all N-tiles)
        # tl.static_range: compile-time unrolled (N_PER_KV is constexpr)
        d_offs = tl.arange(0, D)

        # Load the N_PER_KV u_Q vectors into a [N_PER_KV, D] register block.
        # We load them row-by-row since N_PER_KV=4 < 16 prevents 2D tl.load tricks.
        uQ0 = tl.load(u_Q_ptr + (h_start + 0) * uQ_stride_bh + d_offs * uQ_stride_d)
        uQ1 = tl.load(u_Q_ptr + (h_start + 1) * uQ_stride_bh + d_offs * uQ_stride_d)
        uQ2 = tl.load(u_Q_ptr + (h_start + 2) * uQ_stride_bh + d_offs * uQ_stride_d)
        uQ3 = tl.load(u_Q_ptr + (h_start + 3) * uQ_stride_bh + d_offs * uQ_stride_d)
        # Each uQi: [D] fp32

        K_base = pid_bh_kv * K_stride_bh

        for tile_start in range(0, N, B_BLOCK):
            n_offs  = tile_start + tl.arange(0, B_BLOCK)
            n_mask  = n_offs < N

            # Load K tile [B_BLOCK, D] bf16 → fp32 (§3: promote at HBM→SRAM boundary)
            K_ptrs  = K_base + n_offs[:, None] * K_stride_n + d_offs[None, :] * K_stride_d
            K_tile  = tl.load(K_ptr + K_ptrs, mask=n_mask[:, None], other=0.0)
            K_f32   = K_tile.to(tl.float32)                    # §3: bf16 → fp32

            # score_h = K_f32 @ u_Qh  → [B_BLOCK] fp32
            # Element-wise broadcast + sum: avoids tl.dot (which needs M,N ≥ 16)
            # score[n] = sum_d K_f32[n, d] * uQ_h[d]
            s0 = tl.sum(K_f32 * uQ0[None, :], axis=1)         # [B_BLOCK] fp32
            s1 = tl.sum(K_f32 * uQ1[None, :], axis=1)
            s2 = tl.sum(K_f32 * uQ2[None, :], axis=1)
            s3 = tl.sum(K_f32 * uQ3[None, :], axis=1)

            # Demote to bf16 at write boundary (§3)
            s_base_offs = n_offs * s_stride_n
            tl.store(scores_ptr + (h_start + 0) * s_stride_bh + s_base_offs,
                     s0.to(tl.bfloat16), mask=n_mask)
            tl.store(scores_ptr + (h_start + 1) * s_stride_bh + s_base_offs,
                     s1.to(tl.bfloat16), mask=n_mask)
            tl.store(scores_ptr + (h_start + 2) * s_stride_bh + s_base_offs,
                     s2.to(tl.bfloat16), mask=n_mask)
            tl.store(scores_ptr + (h_start + 3) * s_stride_bh + s_base_offs,
                     s3.to(tl.bfloat16), mask=n_mask)


def topk_pass1_projection(
    K_cache: torch.Tensor,   # [B, H_kv, N, D] bf16
    u_Q: torch.Tensor,       # [B, H, D] fp32
) -> torch.Tensor:           # [B, H, N] bf16
    """
    Python wrapper for Pass 1 projection kernel.

    Validates inputs per spec §1.2, launches (B*H_kv) programs, returns scores.
    Caller (topk_qaxis_select) provides u_Q already in fp32 (_compute_u_Q).
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Pass 1 requires Triton. Install: pip install triton")
    if not K_cache.is_cuda:
        raise RuntimeError("Pass 1 requires CUDA tensors.")

    B, H_kv, N, D = K_cache.shape
    H = u_Q.shape[1]
    n_per_kv = H // H_kv

    assert K_cache.dtype == torch.bfloat16,  f"K_cache must be bf16, got {K_cache.dtype}"
    assert u_Q.dtype == torch.float32,        f"u_Q must be fp32, got {u_Q.dtype}"
    assert K_cache.is_contiguous(),           "K_cache must be contiguous"
    assert u_Q.is_contiguous(),               "u_Q must be contiguous"
    assert H % H_kv == 0,                     f"H={H} must be divisible by H_kv={H_kv}"
    assert n_per_kv == 4, (
        f"Pass 1 kernel hardcoded for n_per_kv=4 (Llama-3.2-1B GQA). Got n_per_kv={n_per_kv}."
    )

    # Flatten (B, H_kv) → BH_kv; (B, H) → BH — project convention
    BH_kv = B * H_kv
    BH    = B * H
    K_flat = K_cache.reshape(BH_kv, N, D)
    uQ_flat = u_Q.reshape(BH, D)

    scores = torch.empty(BH, N, dtype=torch.bfloat16, device=K_cache.device)

    grid = (BH_kv,)
    _pass1_projection_kernel[grid](
        K_flat, uQ_flat, scores,
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        uQ_flat.stride(0), uQ_flat.stride(1),
        scores.stride(0), scores.stride(1),
        N=N,
        H_kv=H_kv,
        D=D,
        N_PER_KV=n_per_kv,
        B_BLOCK=64,
    )

    return scores.reshape(B, H, N)
