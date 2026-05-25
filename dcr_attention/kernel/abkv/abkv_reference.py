"""
ABKV reference implementation — CPU-correct, not optimised for speed.

Public API
----------
build_abkv_cache(K_cache, V_cache, num_bins, axis_vec=None) -> ABKVCache
    Pre-sort KV by axis score at prefill time.

abkv_attention_reference(Q, cache, k_eff, *, scan_budget=None) -> Tensor
    Decode-step attention using pre-sorted cache.
    scan_budget=None (default): take first k_eff keys → exact prefix mode.
    scan_budget=S > k_eff: scan S keys, pick true top-k within those S.

abkv_attention_with_signature(Q, K_cache, V_cache, k_eff, num_bins, axis_vec)
    One-shot convenience: build + attend.

Quality invariant (tested in test_abkv_reference.py):
    cos_sim(O_abkv, O_sdpa) >= 0.95 at c=0.8
    cos_sim(O_abkv, O_sdpa) >= 0.85 at c >= 0.10
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from dcr_attention.kernel.abkv.abkv_cache import ABKVCache

# ---------------------------------------------------------------------------
# dtype constants
# ---------------------------------------------------------------------------
_F32  = torch.float32
_BF16 = torch.bfloat16


# ===========================================================================
# Axis computation
# ===========================================================================

def _compute_axis_vec(K_cache: Tensor) -> Tensor:
    """
    Compute per-head axis vectors as the mean of unit K vectors.

    K_cache: [B, H_kv, N, D] bf16
    Returns: [H_kv, D] fp32

    The mean unit-K vector points toward the centroid direction of keys for
    each head and serves as a stable pre-sort axis that does not depend on
    the decode query.  Keys further along this direction are more likely to
    receive high attention weight from typical queries (locality assumption).
    """
    K_f32 = K_cache.float()                                    # [B, H_kv, N, D]
    norms = torch.linalg.norm(K_f32, dim=-1, keepdim=True)    # [B, H_kv, N, 1]
    norms = norms.clamp(min=1e-12)
    K_unit = K_f32 / norms                                     # [B, H_kv, N, D]
    axis = K_unit.mean(dim=(0, 2))                             # [H_kv, D]
    a_norms = torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-12)
    return axis / a_norms                                      # [H_kv, D] fp32 unit


# ===========================================================================
# Cache construction
# ===========================================================================

def build_abkv_cache(
    K_cache: Tensor,                    # [B, H_kv, N, D] bf16
    V_cache: Tensor,                    # [B, H_kv, N, D] bf16
    num_bins: int,
    axis_vec: Tensor | None = None,     # [H_kv, D] fp32; None → computed from K
) -> ABKVCache:
    """
    Pre-sort KV pairs by axis projection score at prefill time.

    Args:
        K_cache : [B, H_kv, N, D] bf16 — full key cache from prefill
        V_cache : [B, H_kv, N, D] bf16 — corresponding value cache
        num_bins: number of equally-sized bins to partition sorted sequence
        axis_vec: optional fixed axis [H_kv, D] fp32; computed from mean-K if None

    Returns:
        ABKVCache with K_sorted, V_sorted in descending axis-score order.
    """
    assert K_cache.dtype == _BF16, f"K_cache must be bf16, got {K_cache.dtype}"
    assert V_cache.dtype == _BF16, f"V_cache must be bf16, got {V_cache.dtype}"
    assert K_cache.shape == V_cache.shape, "K and V shapes must match"
    assert num_bins >= 1, f"num_bins must be >= 1, got {num_bins}"

    B, H_kv, N, D = K_cache.shape
    device = K_cache.device

    # Compute or validate axis vector
    if axis_vec is None:
        axis_vec = _compute_axis_vec(K_cache)                  # [H_kv, D] fp32
    else:
        axis_vec = axis_vec.to(device=device, dtype=_F32)
        assert axis_vec.shape == (H_kv, D), (
            f"axis_vec must be [H_kv={H_kv}, D={D}], got {axis_vec.shape}"
        )

    # Project all keys onto axis: score[b, h, n] = K[b,h,n,:] · axis[h,:]
    K_f32 = K_cache.float()                                    # [B, H_kv, N, D]
    # einsum: b h n d, h d -> b h n
    scores = torch.einsum("bhnd,hd->bhn", K_f32, axis_vec)    # [B, H_kv, N] fp32

    # Sort descending: highest-score keys first
    sort_perm = torch.argsort(scores, dim=-1, descending=True, stable=True)  # [B, H_kv, N]

    # Reorder K and V by sort_perm
    perm_exp = sort_perm.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, H_kv, N, D]
    K_sorted = torch.gather(K_cache, dim=2, index=perm_exp)   # [B, H_kv, N, D] bf16
    V_sorted = torch.gather(V_cache, dim=2, index=perm_exp)   # [B, H_kv, N, D] bf16
    axis_scores_sorted = torch.gather(scores, dim=-1, index=sort_perm)  # [B, H_kv, N]

    bin_size = math.ceil(N / num_bins)

    return ABKVCache(
        K_sorted=K_sorted,
        V_sorted=V_sorted,
        axis_scores=axis_scores_sorted,
        axis_vec=axis_vec,
        sort_perm=sort_perm,
        num_bins=num_bins,
        bin_size=bin_size,
        N=N,
        H_kv=H_kv,
        D=D,
    )


# ===========================================================================
# Decode attention
# ===========================================================================

def _online_softmax_attention(
    Q: Tensor,      # [B, H, D] fp32
    K: Tensor,      # [B, H, k, D] fp32
    V: Tensor,      # [B, H, k, D] fp32
    B_block: int = 64,
) -> Tensor:        # [B, H, D] fp32
    """Online FlashAttention over k selected keys. Running state stays fp32."""
    B, H, k, D = K.shape
    scale = 1.0 / math.sqrt(D)
    device = K.device

    m = torch.full((B, H), float("-inf"), dtype=_F32, device=device)
    l = torch.zeros(B, H, dtype=_F32, device=device)
    o = torch.zeros(B, H, D, dtype=_F32, device=device)

    for ts in range(0, k, B_block):
        te = min(ts + B_block, k)
        K_blk = K[:, :, ts:te, :]                              # [B, H, tile, D]
        V_blk = V[:, :, ts:te, :]                              # [B, H, tile, D]
        s = torch.einsum("bhd,bhjd->bhj", Q, K_blk) * scale   # [B, H, tile]
        m_tile = s.amax(dim=-1)                                # [B, H]
        m_new  = torch.maximum(m, m_tile)
        alpha  = torch.exp(m - m_new)
        beta   = torch.exp(s - m_new.unsqueeze(-1))
        l = alpha * l + beta.sum(dim=-1)
        o = alpha.unsqueeze(-1) * o + torch.einsum("bhj,bhjd->bhd", beta, V_blk)
        m = m_new

    return o / l.unsqueeze(-1)                                 # [B, H, D] fp32


def abkv_attention_reference(
    Q: Tensor,                     # [B, H, D] bf16
    cache: ABKVCache,
    k_eff: int,
    *,
    scan_budget: int | None = None,
) -> Tensor:                       # [B, H, D] bf16
    """
    ABKV decode-step attention using pre-sorted KV cache.

    The pre-sorted cache lets decode take the leading `k_eff` entries (or
    `scan_budget` entries when specified) directly from K_sorted instead of
    scanning all N keys.

    scan_budget=None (default):
        Take leading k_eff keys (pure prefix — fastest; quality depends on
        axis alignment with Q direction at decode time).

    scan_budget=S where S > k_eff:
        Scan the first S pre-sorted keys, rerank by true Q·K score within
        those S, then compute attention on the true top-k_eff.  Improves
        quality at the cost of scanning more entries.

    GQA: H query heads share H_kv key heads.  n_per_kv = H // H_kv.
    """
    assert Q.dtype == _BF16, f"Q must be bf16, got {Q.dtype}"
    B, H, D = Q.shape
    H_kv = cache.H_kv
    assert H % H_kv == 0, f"H={H} must be divisible by H_kv={H_kv}"
    assert k_eff > 0, f"k_eff must be > 0, got {k_eff}"
    assert k_eff <= cache.N, f"k_eff={k_eff} > N={cache.N}"

    n_per_kv = H // H_kv
    device = Q.device

    budget = scan_budget if (scan_budget is not None and scan_budget > k_eff) else k_eff
    budget = min(budget, cache.N)

    # Prefix-select from sorted cache
    K_cand = cache.K_sorted[:, :, :budget, :].to(device)      # [B, H_kv, budget, D] bf16
    V_cand = cache.V_sorted[:, :, :budget, :].to(device)      # [B, H_kv, budget, D] bf16

    Q_f32 = Q.float()                                          # [B, H, D] fp32

    # Expand K/V from H_kv to H for GQA
    K_exp = K_cand.repeat_interleave(n_per_kv, dim=1).float() # [B, H, budget, D]
    V_exp = V_cand.repeat_interleave(n_per_kv, dim=1).float() # [B, H, budget, D]

    if scan_budget is not None and scan_budget > k_eff:
        # Rerank within scanned candidates: compute Q·K and select true top-k_eff
        scale = 1.0 / math.sqrt(D)
        qk = torch.einsum("bhd,bhjd->bhj", Q_f32, K_exp) * scale  # [B, H, budget]
        topk_idx = qk.topk(k_eff, dim=-1, largest=True, sorted=False).indices  # [B, H, k_eff]
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)    # [B, H, k_eff, D]
        K_sel = torch.gather(K_exp, dim=2, index=idx_exp)          # [B, H, k_eff, D]
        V_sel = torch.gather(V_exp, dim=2, index=idx_exp)
    else:
        K_sel = K_exp     # [B, H, k_eff, D]
        V_sel = V_exp

    O_f32 = _online_softmax_attention(Q_f32, K_sel, V_sel)    # [B, H, D] fp32

    assert torch.isfinite(O_f32).all(), "abkv_attention_reference produced non-finite output"
    return O_f32.to(_BF16)


# ===========================================================================
# One-shot convenience wrapper
# ===========================================================================

def abkv_attention_with_signature(
    Q: Tensor,                      # [B, H, D] bf16
    K_cache: Tensor,                # [B, H_kv, N, D] bf16
    V_cache: Tensor,                # [B, H_kv, N, D] bf16
    k_eff: int,
    *,
    num_bins: int = 32,
    axis_vec: Tensor | None = None,
    scan_budget: int | None = None,
) -> Tensor:                        # [B, H, D] bf16
    """
    Build ABKV cache then run one decode step.  Convenience function for tests.
    """
    cache = build_abkv_cache(K_cache, V_cache, num_bins, axis_vec=axis_vec)
    return abkv_attention_reference(Q, cache, k_eff, scan_budget=scan_budget)


# ===========================================================================
# SDPA baseline (for quality comparisons)
# ===========================================================================

def sdpa_reference(
    Q: Tensor,       # [B, H, D] bf16
    K_cache: Tensor, # [B, H_kv, N, D] bf16
    V_cache: Tensor, # [B, H_kv, N, D] bf16
) -> Tensor:         # [B, H, D] bf16
    """Full softmax attention over all N keys (ground truth)."""
    B, H, D = Q.shape
    _, H_kv, N, _ = K_cache.shape
    n_per_kv = H // H_kv
    scale = 1.0 / math.sqrt(D)

    Q_f32 = Q.float()
    K_f32 = K_cache.repeat_interleave(n_per_kv, dim=1).float()  # [B, H, N, D]
    V_f32 = V_cache.repeat_interleave(n_per_kv, dim=1).float()  # [B, H, N, D]

    scores = torch.einsum("bhd,bhnd->bhn", Q_f32, K_f32) * scale  # [B, H, N]
    weights = scores.softmax(dim=-1)
    out_f32 = torch.einsum("bhn,bhnd->bhd", weights, V_f32)        # [B, H, D]
    return out_f32.to(_BF16)


def cosine_similarity_output(O_abkv: Tensor, O_sdpa: Tensor) -> float:
    """
    Mean cosine similarity between ABKV and SDPA outputs, averaged over (B, H).

    O_abkv, O_sdpa: [B, H, D] bf16
    Returns: scalar in [-1, 1].
    """
    a = O_abkv.float().flatten(0, 1)  # [B*H, D]
    b = O_sdpa.float().flatten(0, 1)
    return F.cosine_similarity(a, b, dim=-1).mean().item()
