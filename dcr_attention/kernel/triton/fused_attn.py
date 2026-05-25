"""
Pass 3 — Fused gather + online softmax-attention kernel.

Per kernel_spec.md §2 Step 4 + §4.4 Option B + §3 dtype boundaries:
- Grid: (B, H) — one program per (b, h) tuple
- Streams k_eff selected keys in tiles of B_BLOCK = 64
- Online softmax with fp32 running state (m, l, o) — Risk R3 mitigation
- Single bf16 demotion at output boundary
- Indirect gather via Pass 2 indices [B, H, k_eff] int32

Spec compliance:
- §2 Step 4: algorithm structure preserved (score → online softmax → output)
- §3 dtype boundaries: ALL fp32 in softmax loop, bf16 only at I/O
- §4.1 B_BLOCK = 64
- §6 edge cases handled (tile masking for k_eff % B_BLOCK ≠ 0)
- §9.2 grid layout (B, H)
- §9.4 NO split-K parallelism

tl.dot not used for score computation: Q is [D] (single vector per program)
so K_sel @ Q = sum over D — element-wise multiply + tl.sum is correct and
avoids the Triton M ≥ 16 WMMA constraint (same pattern as Pass 1, per INS-16
analogy).
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
    def _pass3_fused_attn_kernel(
        # Pointers
        Q_ptr,          # [B, H, D] bf16
        K_ptr,          # [B, H_kv, N, D] bf16
        V_ptr,          # [B, H_kv, N, D] bf16
        indices_ptr,    # [B, H, k_eff] int32
        O_ptr,          # [B, H, D] bf16 (output)
        # Strides
        Q_stride_b, Q_stride_h, Q_stride_d,
        K_stride_b, K_stride_hkv, K_stride_n, K_stride_d,
        V_stride_b, V_stride_hkv, V_stride_n, V_stride_d,
        idx_stride_b, idx_stride_h, idx_stride_k,
        O_stride_b, O_stride_h, O_stride_d,
        # Shape
        k_eff,                             # runtime int
        H_kv,                              # runtime int
        D:          tl.constexpr,          # 64 for Llama-3.2-1B
        N_PER_KV:   tl.constexpr,          # 4 for Llama-3.2-1B
        B_BLOCK:    tl.constexpr,          # 64 per spec §4.1
    ):
        """
        Per-program work: compute O[b, h, :] for one (b, h) tuple.

        §3 dtype boundaries (CRITICAL — do not demote inside loop):
        - Q: bf16 → fp32 once before loop
        - K_sel, V_sel: bf16 → fp32 on load
        - m, l, o (softmax state): fp32 throughout  ← Risk R3 mitigation
        - alpha, beta, scores: fp32
        - O: fp32 → bf16 at single write boundary
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        # GQA mapping: h → h_kv
        pid_hkv = pid_h // N_PER_KV

        # Load Q[b, h, :] bf16 → fp32 (§3: promoted once before any computation)
        d_range = tl.arange(0, D)
        Q_offsets = pid_b * Q_stride_b + pid_h * Q_stride_h + d_range * Q_stride_d
        Q_f32 = tl.load(Q_ptr + Q_offsets).to(tl.float32)   # [D] fp32

        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

        # Online softmax running state — fp32 throughout (spec §3 invariant, R3)
        m = tl.full([1], float("-inf"), tl.float32)          # scalar fp32
        l = tl.zeros([1], tl.float32)                        # scalar fp32
        o = tl.zeros([D], tl.float32)                        # [D] fp32

        for tile_start in range(0, k_eff, B_BLOCK):
            k_offsets = tile_start + tl.arange(0, B_BLOCK)  # [B_BLOCK]
            k_mask    = k_offsets < k_eff

            # Load indices for this tile
            idx_offsets = (
                pid_b * idx_stride_b
                + pid_h * idx_stride_h
                + k_offsets * idx_stride_k
            )
            idx_tile = tl.load(
                indices_ptr + idx_offsets, mask=k_mask, other=0
            )                                                # [B_BLOCK] int32

            # Gather K_sel_tile via indirect addressing
            # K_sel_tile[i, d] = K_cache[b, h_kv, idx_tile[i], d]
            K_offsets = (
                pid_b   * K_stride_b
                + pid_hkv * K_stride_hkv
                + idx_tile[:, None] * K_stride_n
                + d_range[None, :] * K_stride_d
            )
            K_sel_f32 = tl.load(
                K_ptr + K_offsets, mask=k_mask[:, None], other=0.0
            ).to(tl.float32)                                 # [B_BLOCK, D] fp32

            # Scores: [B_BLOCK] = (K_sel @ Q_f32) * scale  — fp32 accumulator
            # Element-wise + tl.sum: Q is [D], so K @ Q has effective M=1 < 16
            scores = tl.sum(K_sel_f32 * Q_f32[None, :], axis=1) * scale  # [B_BLOCK] fp32

            # Mask invalid tail positions (avoids -inf leaking into softmax max)
            scores = tl.where(k_mask, scores, float("-inf"))

            # Online softmax update — fp32 state, NO bf16 demotion (spec §3)
            tile_max = tl.max(scores, axis=0)                # scalar fp32
            m_new    = tl.maximum(m, tile_max)               # scalar fp32

            alpha = tl.exp(m - m_new)                        # scalar fp32
            beta  = tl.exp(scores - m_new)                   # [B_BLOCK] fp32

            # Zero out invalid positions so they don't contribute to l or o
            beta  = tl.where(k_mask, beta, 0.0)

            l = alpha * l + tl.sum(beta, axis=0)             # scalar fp32

            # Gather V_sel_tile and accumulate weighted sum
            V_offsets = (
                pid_b   * V_stride_b
                + pid_hkv * V_stride_hkv
                + idx_tile[:, None] * V_stride_n
                + d_range[None, :] * V_stride_d
            )
            V_sel_f32 = tl.load(
                V_ptr + V_offsets, mask=k_mask[:, None], other=0.0
            ).to(tl.float32)                                 # [B_BLOCK, D] fp32

            # o += beta @ V_sel_f32  (weighted sum, fp32)
            beta_v = tl.sum(beta[:, None] * V_sel_f32, axis=0)  # [D] fp32
            o = alpha * o + beta_v                           # [D] fp32

            m = m_new

        # Single bf16 demotion at output boundary (spec §3 — only here)
        O_f32    = o / l                                     # [D] fp32
        O_bf16   = O_f32.to(tl.bfloat16)                    # [D] bf16

        O_offsets = pid_b * O_stride_b + pid_h * O_stride_h + d_range * O_stride_d
        tl.store(O_ptr + O_offsets, O_bf16)


def fused_topk_attention(
    Q: torch.Tensor,            # [B, H, D] bf16
    K_cache: torch.Tensor,      # [B, H_kv, N, D] bf16
    V_cache: torch.Tensor,      # [B, H_kv, N, D] bf16
    indices: torch.Tensor,      # [B, H, k_eff] int32 (from Pass 2)
) -> torch.Tensor:              # [B, H, D] bf16
    """
    Pass 3 wrapper: fused gather + online softmax-attention.

    Validates inputs per spec §1, launches grid (B, H), returns O.
    No intermediate K_sel/V_sel tensors allocated — indirect gather inside kernel.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Pass 3 requires Triton. Install: pip install triton")
    if not Q.is_cuda:
        raise RuntimeError("Pass 3 requires CUDA tensors.")

    B, H, D       = Q.shape
    B_k, H_kv, N, D_k = K_cache.shape
    B_i, H_i, k_eff    = indices.shape

    assert Q.dtype     == torch.bfloat16,  f"Q must be bf16, got {Q.dtype}"
    assert K_cache.dtype == torch.bfloat16, f"K_cache must be bf16"
    assert V_cache.dtype == torch.bfloat16, f"V_cache must be bf16"
    assert indices.dtype == torch.int32,    f"indices must be int32, got {indices.dtype}"
    assert Q.is_contiguous()
    assert K_cache.is_contiguous()
    assert V_cache.is_contiguous()
    assert indices.is_contiguous()
    assert B == B_k == B_i and H == H_i and D == D_k
    assert H % H_kv == 0
    n_per_kv = H // H_kv
    assert n_per_kv == 4, (
        f"Pass 3 kernel hardcoded for n_per_kv=4 (Llama-3.2-1B). Got {n_per_kv}."
    )
    assert D == 64, f"spec locked at D=64; got D={D}"

    O = torch.empty(B, H, D, dtype=torch.bfloat16, device=Q.device)

    grid = (B, H)
    _pass3_fused_attn_kernel[grid](
        Q, K_cache, V_cache, indices, O,
        Q.stride(0),       Q.stride(1),       Q.stride(2),
        K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
        V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2),
        O.stride(0),       O.stride(1),       O.stride(2),
        k_eff=k_eff,
        H_kv=H_kv,
        D=D,
        N_PER_KV=n_per_kv,
        B_BLOCK=64,
    )
    return O
