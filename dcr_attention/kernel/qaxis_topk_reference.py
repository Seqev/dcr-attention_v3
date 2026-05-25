"""
Top-K Q-Axis Fused Decode Attention — Pure PyTorch Reference Implementation

Implements kernel_spec.md §2 algorithm directly. Used as ground truth for
M5 validation. Correct but not optimized for speed.

Spec compliance:
- §1.2 input shapes/dtypes: enforced via assertions
- §2 algorithm: direct translation, steps 0–4 clearly delineated
- §3 dtype boundaries: every operation per §3 table
- §6 edge cases: all 9 cases handled
- §8 function signature: exact

torch.argsort(stable=True) is used for top-K selection instead of the
explicit heap loop described in §2 Step 2. This is equivalent because:
  1. stable=True preserves original index order for equal elements →
     smaller index wins on ties (§6 tie-breaking rule).
  2. Verified equivalent to explicit heap on synthetic data (test_heap_equivalence
     in test_qaxis_topk_reference.py).

DO NOT modify without architect review of the corresponding spec section.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# §3 dtype constants
# ---------------------------------------------------------------------------
_F32 = torch.float32
_BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Step helpers (exposed for white-box unit testing and M5 component tests)
# ---------------------------------------------------------------------------

def _compute_u_Q(
    Q: torch.Tensor,   # [B, H, D] bf16
    eps: float,
) -> torch.Tensor:     # [B, H, D] fp32
    """
    §2 Step 1: compute unit Q vector.

    §3: Q promoted to fp32 once; norm and division in fp32; output fp32.
    §6: zero-query handled via eps clamp.
    """
    q_f32 = Q.to(_F32)                                          # §3: fp32
    norms = torch.linalg.norm(q_f32, dim=-1, keepdim=True)     # [B, H, 1] fp32
    norms = norms.clamp(min=eps)                                # §6: zero-query guard
    return q_f32 / norms                                        # [B, H, D] fp32


def _compute_projection_scores(
    K_cache: torch.Tensor,   # [B, H_kv, N, D] bf16
    u_Q: torch.Tensor,       # [B, H, D] fp32
) -> torch.Tensor:           # [B, H, N] bf16
    """
    §2 Step 2 (score computation only): project all N keys onto u_Q.

    §3: K promoted to fp32 in projection einsum; accumulator fp32;
        output bf16 (for top-K comparison key only).

    Vectorized over (B, H_kv) groups to avoid O(B*H) Python loops.
    """
    B, H_kv, N, D = K_cache.shape
    H = u_Q.shape[1]
    n_per_kv = H // H_kv
    device = K_cache.device

    scores = torch.empty(B, H, N, dtype=_BF16, device=device)

    for h_kv in range(H_kv):
        K_kv_f32 = K_cache[:, h_kv, :, :].to(_F32)            # [B, N, D] fp32 (§3)
        h_start = h_kv * n_per_kv
        h_end = h_start + n_per_kv
        u_Q_group = u_Q[:, h_start:h_end, :]                   # [B, n_per_kv, D] fp32

        # score[b, h, n] = sum_d K_kv_f32[b, n, d] * u_Q_group[b, h-h_start, d]
        # einsum: 'bnd, bhd -> bnh'
        score_f32 = torch.einsum("bnd,bhd->bnh", K_kv_f32, u_Q_group)  # [B, N, n_per_kv]
        scores[:, h_start:h_end, :] = score_f32.transpose(-2, -1).to(_BF16)  # §3: →bf16

    return scores  # [B, H, N] bf16


def _select_topk_indices(
    scores: torch.Tensor,  # [B, H, N] bf16
    k_eff: int,
) -> torch.Tensor:         # [B, H, k_eff] int64, sorted ascending
    """
    §2 Step 2 (selection): top-K indices from projection scores.

    §6 tie-breaking: torch.argsort with stable=True preserves original order
    for equal elements → smaller index wins on bf16 ties, exactly matching §6.

    Returns sorted ascending (required by Step 4 tiling for sequential access).
    """
    scores_f32 = scores.float()                                 # promote for argsort
    # stable=True: equal elements keep original (ascending) index order → §6 tie-break
    sorted_idx = torch.argsort(scores_f32, dim=-1, descending=True, stable=True)
    topk_idx = sorted_idx[:, :, :k_eff]                        # [B, H, k_eff]
    topk_idx_sorted, _ = topk_idx.sort(dim=-1)                 # ascending for Step 4
    return topk_idx_sorted                                      # [B, H, k_eff] int64


def _gather_kv(
    K_cache: torch.Tensor,    # [B, H_kv, N, D] bf16
    V_cache: torch.Tensor,    # [B, H_kv, N, D] bf16
    indices: torch.Tensor,    # [B, H, k_eff] int64
) -> tuple[torch.Tensor, torch.Tensor]:  # ([B,H,k_eff,D], [B,H,k_eff,D]) bf16
    """
    §2 Step 3: gather K_sel, V_sel using top-K indices.

    §3: output is bf16 (storage format; promoted to fp32 in Step 4).
    GQA: K_cache has H_kv heads; each query head maps to h_kv = h // n_per_kv.
    """
    B, H_kv, N, D = K_cache.shape
    H = indices.shape[1]
    k_eff = indices.shape[2]
    n_per_kv = H // H_kv
    device = K_cache.device

    K_sel = torch.empty(B, H, k_eff, D, dtype=_BF16, device=device)
    V_sel = torch.empty(B, H, k_eff, D, dtype=_BF16, device=device)

    for h_kv in range(H_kv):
        h_start = h_kv * n_per_kv
        h_end = h_start + n_per_kv
        idx_group = indices[:, h_start:h_end, :]                # [B, n_per_kv, k_eff]

        # Expand index for gather: [B, n_per_kv, k_eff, D]
        idx_exp = idx_group.unsqueeze(-1).expand(-1, -1, -1, D) # [B, n_per_kv, k_eff, D]

        K_kv = K_cache[:, h_kv, :, :].unsqueeze(1).expand(-1, n_per_kv, -1, -1)  # [B, n_per_kv, N, D]
        V_kv = V_cache[:, h_kv, :, :].unsqueeze(1).expand(-1, n_per_kv, -1, -1)

        K_sel[:, h_start:h_end, :, :] = torch.gather(K_kv, dim=2, index=idx_exp)
        V_sel[:, h_start:h_end, :, :] = torch.gather(V_kv, dim=2, index=idx_exp)

    return K_sel, V_sel


def _fused_softmax_attention(
    K_sel: torch.Tensor,   # [B, H, k_eff, D] bf16
    V_sel: torch.Tensor,   # [B, H, k_eff, D] bf16
    Q: torch.Tensor,       # [B, H, D] bf16 — UNNORMALIZED original Q
    B_block: int = 64,
) -> torch.Tensor:         # [B, H, D] bf16
    """
    §2 Step 4: online FlashAttention softmax over k_eff selected keys.

    §3 CRITICAL: running state (m, l, o) is fp32 throughout. Never bf16.
    Final demotion to bf16 at single output write only.
    """
    B, H, k_eff, D = K_sel.shape
    scale = 1.0 / math.sqrt(D)
    device = K_sel.device

    Q_f32 = Q.to(_F32)                                          # §3: Q→fp32 once

    # §3: running online softmax state — fp32 throughout
    m = torch.full((B, H), float("-inf"), dtype=_F32, device=device)
    l = torch.zeros(B, H, dtype=_F32, device=device)
    o = torch.zeros(B, H, D, dtype=_F32, device=device)

    for tile_start in range(0, k_eff, B_block):
        tile_end = min(tile_start + B_block, k_eff)

        K_blk = K_sel[:, :, tile_start:tile_end, :].to(_F32)   # [B, H, tile_sz, D] §3
        V_blk = V_sel[:, :, tile_start:tile_end, :].to(_F32)   # [B, H, tile_sz, D] §3

        # Attention scores: §3 fp32 accumulator
        scores = torch.einsum("bhd,bhjd->bhj", Q_f32, K_blk) * scale  # [B, H, tile_sz] fp32

        # Online softmax update — §3: m, l, o never touch bf16
        tile_max = scores.amax(dim=-1)                          # [B, H] fp32
        m_new = torch.maximum(m, tile_max)                      # [B, H] fp32

        alpha = torch.exp(m - m_new)                            # [B, H] fp32
        beta = torch.exp(scores - m_new.unsqueeze(-1))          # [B, H, tile_sz] fp32

        l = alpha * l + beta.sum(dim=-1)                        # [B, H] fp32
        o = alpha.unsqueeze(-1) * o + torch.einsum("bhj,bhjd->bhd", beta, V_blk)  # [B, H, D]
        m = m_new

    # §3: single demotion to bf16 at output boundary only
    O = (o / l.unsqueeze(-1)).to(_BF16)                        # [B, H, D] bf16
    return O


# ---------------------------------------------------------------------------
# Main entry point — §8 signature
# ---------------------------------------------------------------------------

def topk_qaxis_attention_reference(
    Q: torch.Tensor,        # [B, H, D]        dtype=torch.bfloat16, contiguous
    K_cache: torch.Tensor,  # [B, H_kv, N, D]  dtype=torch.bfloat16, contiguous in N
    V_cache: torch.Tensor,  # [B, H_kv, N, D]  dtype=torch.bfloat16, contiguous in N
    k_eff: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:          # [B, H, D]  dtype=torch.bfloat16, contiguous
    """
    Top-K Q-axis fused decode attention — pure PyTorch reference implementation.

    Implements kernel_spec.md §2 with exact dtype boundaries from §3.
    Correct but not optimized for speed.

    Used as ground truth for M5 validation. If M5 disagrees with this function,
    the spec (§2/§3) is wrong — not the kernel.

    Must run on both CUDA and CPU (for unit tests without GPU).
    """
    # ── Input validation (§1.2, §6) ─────────────────────────────────────────
    assert Q.dtype == _BF16, f"Q must be bfloat16, got {Q.dtype}"
    assert K_cache.dtype == _BF16, f"K_cache must be bfloat16, got {K_cache.dtype}"
    assert V_cache.dtype == _BF16, f"V_cache must be bfloat16, got {V_cache.dtype}"
    assert Q.dim() == 3, f"Q must be [B, H, D], got shape {Q.shape}"
    assert K_cache.dim() == 4, f"K_cache must be [B, H_kv, N, D], got {K_cache.shape}"
    assert V_cache.dim() == 4, f"V_cache must be [B, H_kv, N, D], got {V_cache.shape}"

    B, H, D = Q.shape
    B_k, H_kv, N, D_k = K_cache.shape

    assert B_k == B, f"Batch mismatch: Q has B={B}, K_cache has B={B_k}"
    assert D_k == D, f"Head dim mismatch: Q D={D}, K_cache D={D_k}"
    assert H % H_kv == 0, f"H={H} must be divisible by H_kv={H_kv} (§6)"
    assert k_eff > 0, f"k_eff must be > 0, got {k_eff} (§6)"

    # ── Step 0: Degenerate case (§2, §6) ────────────────────────────────────
    if N <= k_eff:
        # Caller (M4 integration layer) handles fallback to standard SDPA;
        # this kernel is not invoked in this regime (§6, §9.3).
        raise ValueError(
            f"kernel invoked with N={N} <= k_eff={k_eff}; caller must dispatch to SDPA"
        )

    # ── Step 1: Compute unit Q vector (§2 Step 1, §3) ───────────────────────
    u_Q = _compute_u_Q(Q, eps)                                  # [B, H, D] fp32

    # ── Step 2: Project all N keys, select top-K (§2 Step 2, §3) ───────────
    scores = _compute_projection_scores(K_cache, u_Q)           # [B, H, N] bf16
    topk_indices = _select_topk_indices(scores, k_eff)          # [B, H, k_eff] int64

    # ── Step 3: Gather selected K, V (§2 Step 3, §3) ────────────────────────
    K_sel, V_sel = _gather_kv(K_cache, V_cache, topk_indices)  # [B, H, k_eff, D] bf16

    # ── Step 4: Fused online softmax-attention (§2 Step 4, §3) ──────────────
    O = _fused_softmax_attention(K_sel, V_sel, Q)               # [B, H, D] bf16

    # ── Final guard (§6: no NaN/inf) ─────────────────────────────────────────
    assert torch.isfinite(O).all(), "M1 reference produced non-finite output"

    return O                                                     # [B, H, D] bf16
