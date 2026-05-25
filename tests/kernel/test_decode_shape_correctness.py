"""
Phase 2-pre decode correctness.

Verify that the kernel + torch fallback produce correct output when
Q.shape[-2] = 1 and K.shape[-2] = N_kv ≠ 1.

The reference for decode-shape is dense attention with rank-window mask
applied to K-side positions; our kernel/fallback is correct iff its output
matches dense attention computed over the in-window subset of K/V.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dcr_attention.kernel.sort_helpers import (
    gather_by_sort_idx,
    prepare_sort_indices,
)
from dcr_attention.kernel.rank_local_fwd_torch import rank_local_fwd_torch

try:
    from dcr_attention.kernel.rank_local_fwd_triton import (
        TRITON_AVAILABLE,
        rank_local_fwd_triton,
    )
except ImportError:
    TRITON_AVAILABLE = False
    rank_local_fwd_triton = None


def _decode_dense_reference(
    Q: torch.Tensor,         # [B, H, 1, D]
    K: torch.Tensor,         # [B, H, N_kv, D]  (sorted)
    V: torch.Tensor,         # [B, H, N_kv, D]  (sorted)
    r_center: torch.Tensor,  # [B, H, 1]
    half_k: int,
) -> torch.Tensor:
    """
    Brute-force dense reference for one-query attention with rank window.
    All math in fp32.
    """
    B, H, _, D = Q.shape
    N_kv = K.shape[-2]
    Q32, K32, V32 = Q.float(), K.float(), V.float()

    # [B, H, 1, N_kv]
    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhid,bhjd->bhij", Q32, K32) * scale

    # Window mask: |sorted_pos - r_c| <= half_k
    positions = torch.arange(N_kv, device=Q.device).view(1, 1, 1, N_kv)
    in_window = (positions - r_center.unsqueeze(-1)).abs() <= half_k
    scores = scores.masked_fill(~in_window, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.einsum("bhij,bhjd->bhid", attn, V32)
    return out.to(Q.dtype)


# ---------------------------------------------------------------------------
# CPU fallback decode correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N_kv", [64, 256, 1024])
@pytest.mark.parametrize("k_window", [16, 64])
def test_torch_fallback_decode_shape_correctness(N_kv, k_window):
    """Fallback path must give correct output for Q.N=1, K.N=N_kv."""
    torch.manual_seed(42)
    B, H, D = 1, 2, 32
    Q = torch.randn(B, H, 1, D)
    K = torch.randn(B, H, N_kv, D)
    V = torch.randn(B, H, N_kv, D)
    axis = torch.randn(D); axis = axis / axis.norm()

    idx = prepare_sort_indices(Q, K, axis)
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_s = gather_by_sort_idx(Q, idx.sort_idx_q)   # identity for N_q=1

    out, lse = rank_local_fwd_torch(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center,
        k_window=k_window,
    )

    ref = _decode_dense_reference(Q_s, K_s, V_s, idx.r_center, k_window // 2)
    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, (
        f"N_kv={N_kv}, k={k_window}: max|out - ref| = {diff:g}"
    )

    assert lse is not None
    assert lse.shape == (B, H, 1)


def test_torch_fallback_decode_full_window_equals_dense():
    """k_window=2*N_kv: bounded loop = full attention; should match dense SDPA."""
    torch.manual_seed(0)
    B, H, D = 1, 2, 32
    N_kv = 128
    Q = torch.randn(B, H, 1, D)
    K = torch.randn(B, H, N_kv, D)
    V = torch.randn(B, H, N_kv, D)
    axis = torch.randn(D); axis = axis / axis.norm()

    idx = prepare_sort_indices(Q, K, axis)
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_s = gather_by_sort_idx(Q, idx.sort_idx_q)

    out, _ = rank_local_fwd_torch(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center,
        k_window=2 * N_kv,
    )

    # Dense reference (against UNSORTED K, V — should match because softmax
    # is permutation-invariant).
    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) * scale
    dense = torch.einsum("bhij,bhjd->bhid", F.softmax(scores, dim=-1), V.float())
    assert torch.allclose(out, dense, atol=1e-5)


# ---------------------------------------------------------------------------
# GPU triton decode correctness
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="Triton decode tests require CUDA + triton",
)


@pytestmark_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N_kv", [256, 1024, 4096])
@pytest.mark.parametrize("k_window", [16, 64, 256])
@pytest.mark.parametrize("D", [64, 128])
def test_triton_decode_matches_torch_fallback(dtype, N_kv, k_window, D):
    """Phase 2-pre decode kernel must match torch fallback within dtype tolerance."""
    if k_window >= N_kv:
        pytest.skip("k >= N_kv degenerate")

    g = torch.Generator(device="cuda").manual_seed(0)
    B, H = 1, 4
    Q = torch.randn(B, H, 1, D, generator=g, device="cuda", dtype=dtype)
    K = torch.randn(B, H, N_kv, D, generator=g, device="cuda", dtype=dtype)
    V = torch.randn(B, H, N_kv, D, generator=g, device="cuda", dtype=dtype)
    axis = torch.randn(D, generator=g, device="cuda", dtype=torch.float32)
    axis = (axis / axis.norm()).to(dtype)

    idx = prepare_sort_indices(Q, K, axis)
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_s = gather_by_sort_idx(Q, idx.sort_idx_q)

    out_torch, _ = rank_local_fwd_torch(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center, k_window=k_window,
    )
    out_triton, _ = rank_local_fwd_triton(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center, k_window=k_window,
    )

    if dtype == torch.float32:
        atol = 1e-5
    else:
        from math import sqrt
        atol = max(1e-2, 1.5e-3 * sqrt(D))

    diff = (out_torch.float() - out_triton.float()).abs().max().item()
    assert diff < atol, (
        f"N_kv={N_kv}, k={k_window}, D={D}, dtype={dtype}: max|diff| = {diff:.3e}"
    )
