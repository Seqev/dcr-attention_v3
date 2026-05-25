"""
Triton forward kernel for rank-local attention.

Structure:
  * one program per ``(batch·head, query-block)`` pair,
  * online softmax over sorted-K tiles inside each query-block,
  * per-element window mask ``|sorted_pos - r_center[i]| <= half_k``,
  * returns output and log-sum-exp (for future backward).

**CUDA only.**  On CPU or when triton is unavailable, callers should route to
``rank_local_fwd_torch`` — the public wrapper does this automatically.

Sources consulted while writing this kernel:
  * FlashAttention-2 reference Triton tutorial (online softmax pattern).
  * Triton 3.x API for ``tl.load``, ``tl.dot``, ``tl.max``, ``tl.where``.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch


try:
    import triton                                   # type: ignore[import-not-found]
    import triton.language as tl                    # type: ignore[import-not-found]
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    # Sentinel for the running softmax max.  Using -inf creates NaNs when
    # ``m_i = m_new = -inf`` (first iteration with a fully-masked window):
    # ``alpha = exp(-inf - -inf) = exp(NaN) = NaN``.  -1e30 is "infinitely
    # negative" for the softmax in fp32 / bf16 (exp(-1e30) underflows to 0)
    # but produces well-defined arithmetic at NaN-trap points.  Same trick
    # used in the FlashAttention-2 reference implementation.
    #
    # ``tl.constexpr`` annotation is required by Triton 3.1 — globals read
    # from inside ``@triton.jit`` must be either constexpr or a tl.constexpr
    # value, otherwise the compiler refuses with a NameError (see INS-15).
    _NEG_INF_SAFE = tl.constexpr(-1.0e30)

    @triton.jit
    def _rank_local_fwd_kernel(
        # Inputs
        Q_ptr, K_ptr, V_ptr,
        r_center_ptr,                   # [B, H, N] int32
        # Outputs
        O_ptr, LSE_ptr,
        # Strides (row-major, trailing D is contiguous)
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_rb, stride_rh, stride_rn,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_lb, stride_lh, stride_ln,
        # Scalar params
        scale,
        N_q,         # number of queries (== seq len of Q)
        N_kv,        # number of keys/values (== seq len of K/V); equals N_q in prefill
        half_k,
        # Compile-time constants
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """
        One program handles BLOCK_Q queries of one (b, h) head.

        Grid:  program_id(0) → query block index, program_id(1) → b*H + h.

        Phase 2-pre: ``N_q`` and ``N_kv`` are passed separately to support
        decode shape (``N_q=1``, ``N_kv=context_length``).  In prefill they
        coincide; in decode they don't.  The kernel logic was already
        symmetric — only the launcher was prefill-only.
        """
        # Program ids
        pid_qblock = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_q = pid_qblock * BLOCK_Q + tl.arange(0, BLOCK_Q)        # [BLOCK_Q]
        mask_q = offs_q < N_q                                        # [BLOCK_Q]

        Q_base = Q_ptr + pid_bh * stride_qb
        K_base = K_ptr + pid_bh * stride_kb
        V_base = V_ptr + pid_bh * stride_vb
        R_base = r_center_ptr + pid_bh * stride_rb
        O_base = O_ptr + pid_bh * stride_ob
        LSE_base = LSE_ptr + pid_bh * stride_lb

        # Load Q block in native dtype, then promote to fp32 for the dot.
        # ``q * scale`` in native dtype + an explicit ``.to(tl.float32)`` keeps
        # the multiplication cheap and the dot operands dtype-matched, which
        # Triton 3.1 ``tl.dot`` requires (see Bug 2b in INS-13).
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = (
            Q_base
            + offs_q[:, None] * stride_qn
            + offs_d[None, :] * stride_qd
        )
        q_native = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)   # [BQ, D]
        q = (q_native * scale).to(tl.float32)                         # [BQ, D] fp32

        # r_center per query.  After Phase 1.3 Q-sort, r_c is monotonically
        # non-decreasing within a BLOCK_Q tile, which is the property that
        # makes the bounded K-loop possible.
        r_center_ptrs = R_base + offs_q * stride_rn
        r_c = tl.load(r_center_ptrs, mask=mask_q, other=0)            # int [BQ]

        # Online-softmax accumulators (all fp32)
        m_i = tl.full([BLOCK_Q], _NEG_INF_SAFE, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

        # ---------------------------------------------------------------
        # Phase 1.3: bound the K-loop to the window union of this Q-block.
        #
        # All queries in this block see keys in [r_c[t] - half_k, r_c[t] + half_k].
        # Their union is [min(r_c) - half_k, max(r_c) + half_k].
        # Round outwards to BLOCK_K boundaries and iterate just that range.
        #
        # tl.min / tl.max with masked-out queries: substitute identity values
        # (max int32 for min-reduction, 0 for max-reduction) on padded slots.
        # ---------------------------------------------------------------
        _SENTINEL_HI = 2147483647                                     # int32 max
        r_c_for_min = tl.where(mask_q, r_c, _SENTINEL_HI)
        r_c_for_max = tl.where(mask_q, r_c, 0)
        r_min_in_block = tl.min(r_c_for_min, axis=0)
        r_max_in_block = tl.max(r_c_for_max, axis=0)

        k_lo = tl.maximum(0, r_min_in_block - half_k)
        k_hi = tl.minimum(N_kv, r_max_in_block + half_k + 1)

        # Round to BLOCK_K boundaries.
        k_block_lo = (k_lo // BLOCK_K) * BLOCK_K
        k_block_hi = ((k_hi + BLOCK_K - 1) // BLOCK_K) * BLOCK_K

        for k_start in range(k_block_lo, k_block_hi, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)                  # [BLOCK_K]
            mask_k = offs_k < N_kv

            # Window mask: |sorted_pos - r_center[i]| <= half_k
            # The mask is still required because not every (q, k) pair within
            # the union-bounded loop actually lies in the per-query window.
            dist = offs_k[None, :] - r_c[:, None]
            in_win = (dist >= -half_k) & (dist <= half_k)
            full_mask = in_win & mask_q[:, None] & mask_k[None, :]

            # Load K, V tiles  [BLOCK_K, HEAD_DIM] in native dtype, promote to fp32
            k_ptrs = (
                K_base
                + offs_k[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V_base
                + offs_k[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            k_tile = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)
            v_tile = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)

            # Scores = Q · K^T, shape [BLOCK_Q, BLOCK_K], both fp32.
            # ``input_precision="ieee"`` disables Tensor Core TF32 reduction
            # (10-bit mantissa) and forces IEEE fp32 multiply-accumulate.  Without
            # this, Triton's default TF32 path on Ampere/Ada introduces O(1e-3)
            # absolute error per dot — see INS-16.  Performance trade-off is
            # acknowledged: a Phase 4 autotune may switch back to TF32 once the
            # downstream task tolerates it (Llama inference probably does).
            s = tl.dot(q, tl.trans(k_tile), input_precision="ieee")

            # Apply window + sequence-bounds mask (use safe -inf sentinel)
            s = tl.where(full_mask, s, _NEG_INF_SAFE)

            # Online softmax update.  With m_i initialised to _NEG_INF_SAFE,
            # alpha and p stay finite even on the first iteration when the
            # window is empty: alpha = exp(-1e30 - -1e30) = 1, p ≈ 0.
            m_new = tl.maximum(m_i, tl.max(s, axis=1))                 # [BQ]
            alpha = tl.exp(m_i - m_new)                                # [BQ]
            p = tl.exp(s - m_new[:, None])                             # [BQ, BK]

            l_i = alpha * l_i + tl.sum(p, axis=1)
            # Same IEEE precision as the score dot above (INS-16).
            acc = acc * alpha[:, None] + tl.dot(p, v_tile, input_precision="ieee")
            m_i = m_new

        # Finalise: rows whose window was entirely empty have m_i = _NEG_INF_SAFE
        # and l_i ≈ 0; produce zero output and -inf LSE for those rows.
        empty = l_i <= 0.0
        safe_l = tl.where(empty, 1.0, l_i)
        out = acc / safe_l[:, None]
        out = tl.where(empty[:, None], 0.0, out)

        lse = m_i + tl.log(safe_l)
        lse = tl.where(empty, -float("inf"), lse)

        # Store — cast back to output dtype via tl.store auto-conversion
        o_ptrs = (
            O_base
            + offs_q[:, None] * stride_on
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, out, mask=mask_q[:, None])

        lse_ptrs = LSE_base + offs_q * stride_ln
        tl.store(lse_ptrs, lse, mask=mask_q)


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

def rank_local_fwd_triton(
    Q: torch.Tensor,
    K_sorted: torch.Tensor,
    V_sorted: torch.Tensor,
    rank_of_k: torch.Tensor,           # unused; kept for signature symmetry
    r_center: torch.Tensor,
    k_window: int,
    scale: Optional[float] = None,
    BLOCK_Q: Optional[int] = None,
    BLOCK_K: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Launch the Triton forward kernel.

    Shapes:
      Q                   : [B, H, N_q, D]
      K_sorted, V_sorted  : [B, H, N_kv, D]
      r_center            : [B, H, N_q]    — per-query insertion rank in K
      rank_of_k           : [B, H, N_kv]   — kept for backward (unused by fwd)

    In **prefill** ``N_q == N_kv``.  In **decode** ``N_q == 1`` and
    ``N_kv == context_length``.  See Phase 2-pre design.

    Block sizes and ``num_stages`` are auto-selected based on ``D`` to fit
    within the 100 KB shared-memory budget of Ada/Hopper SMs.  Callers may
    override for benchmarking.  See INS-19 for the SMEM math.

    Returns
    -------
    O   : [B, H, N_q, D]   same dtype as Q
    LSE : [B, H, N_q]      fp32
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton not available in this environment. "
            "Import dcr_attention.kernel.rank_local_fwd_torch.rank_local_fwd_torch instead."
        )
    if not Q.is_cuda:
        raise RuntimeError("rank_local_fwd_triton requires CUDA tensors.")

    B, H, N_q, D = Q.shape
    N_kv = K_sorted.shape[-2]
    if K_sorted.shape != V_sorted.shape:
        raise ValueError(
            f"K_sorted shape {tuple(K_sorted.shape)} must equal "
            f"V_sorted shape {tuple(V_sorted.shape)}"
        )
    if K_sorted.shape[:2] != Q.shape[:2] or K_sorted.shape[-1] != D:
        raise ValueError(
            f"K_sorted shape {tuple(K_sorted.shape)} incompatible with "
            f"Q shape {tuple(Q.shape)}: B, H, D must match"
        )
    if r_center.shape != (B, H, N_q):
        raise ValueError(
            f"r_center shape {tuple(r_center.shape)} must equal [B={B}, H={H}, N_q={N_q}]"
        )

    if scale is None:
        scale = 1.0 / (D ** 0.5)
    half_k = k_window // 2

    # Adaptive block / pipeline configuration to fit within ~100 KB SMEM.
    # SMEM budget per stage in fp32:
    #   Q-block:     BLOCK_Q · D · 4
    #   K, V tiles:  2 · BLOCK_K · D · 4
    #   acc:         BLOCK_Q · D · 4
    #   s, p:        2 · BLOCK_Q · BLOCK_K · 4
    # Triton multiplies tile loads by num_stages.  At D=128, BLOCK_Q=32,
    # BLOCK_K=64, num_stages=3 the kernel needs ~155 KB (observed).
    if D <= 64:
        _BLOCK_Q = 32 if BLOCK_Q is None else BLOCK_Q
        _BLOCK_K = 64 if BLOCK_K is None else BLOCK_K
        _num_stages = 3 if num_stages is None else num_stages
    else:
        # D=128 (Llama-class).  Halve BLOCK_K and reduce pipeline depth.
        _BLOCK_Q = 32 if BLOCK_Q is None else BLOCK_Q
        _BLOCK_K = 32 if BLOCK_K is None else BLOCK_K
        _num_stages = 2 if num_stages is None else num_stages

    # Collapse (B, H) into a single batch-head axis. This lets us pass a single
    # stride_bh into the kernel and use program_id(1) as the flat (b*H+h) index.
    BH = B * H
    Q_f = Q.reshape(BH, N_q, D).contiguous()
    K_f = K_sorted.reshape(BH, N_kv, D).contiguous()
    V_f = V_sorted.reshape(BH, N_kv, D).contiguous()
    r_center_f = r_center.reshape(BH, N_q).to(torch.int32).contiguous()

    O = torch.empty_like(Q_f)
    LSE = torch.empty((BH, N_q), device=Q.device, dtype=torch.float32)

    grid = (triton.cdiv(N_q, _BLOCK_Q), BH)

    _rank_local_fwd_kernel[grid](
        Q_f, K_f, V_f,
        r_center_f,
        O, LSE,
        # strides: batch-head axis is dim 0 (combined), seq is dim 1, d is dim 2
        Q_f.stride(0), 0, Q_f.stride(1), Q_f.stride(2),
        K_f.stride(0), 0, K_f.stride(1), K_f.stride(2),
        V_f.stride(0), 0, V_f.stride(1), V_f.stride(2),
        r_center_f.stride(0), 0, r_center_f.stride(1),
        O.stride(0), 0, O.stride(1), O.stride(2),
        LSE.stride(0), 0, LSE.stride(1),
        scale,
        N_q,
        N_kv,
        half_k,
        BLOCK_Q=_BLOCK_Q,
        BLOCK_K=_BLOCK_K,
        HEAD_DIM=D,
        num_stages=_num_stages,
    )

    return O.reshape(B, H, N_q, D), LSE.reshape(B, H, N_q)
