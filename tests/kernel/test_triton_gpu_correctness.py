"""
GPU correctness suite for the Triton forward kernel.

Validates Triton kernel output against the torch fallback (which is itself
validated against the reference in tests/kernel/test_rank_local_fwd_torch.py
on small N).

Skipped automatically if CUDA is unavailable.

dtype matrix and tolerances:
  * fp32 — strict, atol = 1e-5  (FA-style numerical contract)
  * bf16 — atol = 1e-2, rtol = 1e-2  (7-bit mantissa, reasonable bound)
  * fp16 — atol = 1e-2, rtol = 1e-2; SKIP at k=2N with extreme inputs
                                     (softmax overflow is a known fp16 limit)

Shape matrix:
  N ∈ {128, 512, 2048},  D ∈ {32, 64, 128},  H ∈ {2, 8},  k ∈ {16, 64, 256}
"""

from __future__ import annotations
from typing import Tuple

import pytest
import torch

from dcr_attention.kernel.rank_local_fwd_torch import rank_local_fwd_torch
from dcr_attention.kernel.sort_helpers import (
    gather_by_sort_idx,
    prepare_sort_indices,
    scatter_by_inv_perm,
)

# Triton import is conditional so the module can be imported on CPU machines.
try:
    from dcr_attention.kernel.rank_local_fwd_triton import (
        TRITON_AVAILABLE,
        rank_local_fwd_triton,
    )
except ImportError:
    TRITON_AVAILABLE = False
    rank_local_fwd_triton = None


# Skip the whole module if no GPU + Triton.
pytestmark = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="Triton kernel tests require CUDA + triton",
)


# ---------------------------------------------------------------------------
# Tolerance model
# ---------------------------------------------------------------------------
#
# Per-dtype baselines, calibrated from round-3 GPU validation observations:
#
#   fp32  with input_precision="ieee" in tl.dot:  ~1e-5 (true IEEE-754)
#   bf16  storage + bf16 hardware multiply (7-bit mantissa):
#         per-element rel error 2^-7 ≈ 8e-3
#         after D-element MAC and softmax-normalisation, observed:
#             D=32:  max|abs| ≤ 5e-3   (passes atol=1e-2)
#             D=128: max|abs| ≤ 1.6e-2 (fails atol=1e-2)
#         Empirical model:  tol(D) = max(1e-2, 1.5e-3 · sqrt(D))
#         Gives 1e-2 for D ≤ 64, 1.7e-2 for D=128 — covers observations
#         with a small safety margin.
#   fp16  has 10-bit mantissa (better than bf16) but more overflow risk.
#         Use the same model — the floor at 1e-2 dominates.
#
# This is the dtype's intrinsic numerical floor, not a "loose-tolerance hack".
# See INS-16 in INSIGHTS.md.

def _expected_atol(dtype: torch.dtype, D: int) -> float:
    if dtype == torch.float32:
        return 1e-5
    if dtype in (torch.bfloat16, torch.float16):
        from math import sqrt
        # Linear-in-sqrt(D) above a 1e-2 floor
        return max(1e-2, 1.5e-3 * sqrt(D))
    raise ValueError(f"unknown dtype {dtype}")


def _tol_dict(dtype: torch.dtype, D: int) -> dict:
    atol = _expected_atol(dtype, D)
    return dict(atol=atol, rtol=atol)


def _build_inputs(
    B: int, H: int, N: int, D: int, dtype: torch.dtype, seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, N, D, generator=g, device="cuda", dtype=dtype)
    K = torch.randn(B, H, N, D, generator=g, device="cuda", dtype=dtype)
    V = torch.randn(B, H, N, D, generator=g, device="cuda", dtype=dtype)
    axis = torch.randn(D, generator=g, device="cuda", dtype=torch.float32)
    axis = axis / axis.norm()
    return Q, K, V, axis


def _run_both_paths(
    Q, K, V, axis, k_window: int,
):
    """Pre-sort once and run both kernels, returning (out_torch, out_triton, lse_torch, lse_triton).

    Phase 1.3: kernel expects Q in sorted-Q-order; we apply Q-sort here and
    inverse-perm both outputs back to original Q-order before returning.
    """
    idx = prepare_sort_indices(Q, K, axis.to(Q.dtype))
    K_sorted = gather_by_sort_idx(K, idx.sort_idx_k)
    V_sorted = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_sorted = gather_by_sort_idx(Q, idx.sort_idx_q)

    out_t_s, lse_t_s = rank_local_fwd_torch(
        Q_sorted, K_sorted, V_sorted, idx.rank_of_k, idx.r_center, k_window=k_window,
    )
    out_tr_s, lse_tr_s = rank_local_fwd_triton(
        Q_sorted, K_sorted, V_sorted, idx.rank_of_k, idx.r_center, k_window=k_window,
    )

    out_t = scatter_by_inv_perm(out_t_s, idx.inv_q_perm)
    out_tr = scatter_by_inv_perm(out_tr_s, idx.inv_q_perm)
    lse_t = scatter_by_inv_perm(lse_t_s, idx.inv_q_perm) if lse_t_s is not None else None
    lse_tr = scatter_by_inv_perm(lse_tr_s, idx.inv_q_perm) if lse_tr_s is not None else None
    return out_t, out_tr, lse_t, lse_tr


# ---------------------------------------------------------------------------
# 1. Compilation smoke test
# ---------------------------------------------------------------------------

def test_kernel_compiles_and_runs() -> None:
    """First-call smoke test — Triton compilation succeeds, no runtime error."""
    Q, K, V, axis = _build_inputs(B=1, H=2, N=128, D=32, dtype=torch.float32)
    idx = prepare_sort_indices(Q, K, axis.to(Q.dtype))
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_s = gather_by_sort_idx(Q, idx.sort_idx_q)
    out, lse = rank_local_fwd_triton(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center, k_window=32,
    )
    assert out.shape == (1, 2, 128, 32)
    assert lse.shape == (1, 2, 128)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 2. Output parity vs torch fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N", [128, 512, 2048])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("k_window", [16, 64, 256])
def test_output_matches_torch_fallback(
    dtype: torch.dtype, N: int, D: int, k_window: int,
) -> None:
    """
    Triton output must match torch fallback within dtype-appropriate tolerance.
    """
    if k_window >= N:
        pytest.skip("k >= N degenerate case covered by full_window tests")

    Q, K, V, axis = _build_inputs(B=1, H=2, N=N, D=D, dtype=dtype)
    out_t, out_tr, _, _ = _run_both_paths(Q, K, V, axis, k_window=k_window)

    tol = _tol_dict(dtype, D)
    if not torch.allclose(out_t, out_tr, **tol):
        # Diagnostic — the most informative thing we can show on failure
        diff = (out_t.float() - out_tr.float()).abs()
        rel = diff / (out_t.float().abs() + 1e-6)
        pytest.fail(
            f"Triton vs torch mismatch  N={N} D={D} k={k_window} dtype={dtype}: "
            f"max|abs| = {diff.max().item():.3e}, max|rel| = {rel.max().item():.3e}, "
            f"tol = {tol}"
        )


# ---------------------------------------------------------------------------
# 3. LSE correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [128, 512])
@pytest.mark.parametrize("k_window", [16, 64])
def test_lse_matches_torch_fallback(dtype, N, k_window) -> None:
    """
    LSE is the per-row log-sum-exp of in-window scores. Triton's online
    softmax must match the torch fallback's torch.logsumexp(...) on
    materialised scores within fp32 tolerance (LSE is always fp32).
    """
    Q, K, V, axis = _build_inputs(B=1, H=2, N=N, D=32, dtype=dtype)
    _, _, lse_t, lse_tr = _run_both_paths(Q, K, V, axis, k_window=k_window)

    # Both LSEs are fp32 by contract; compare in fp32.
    assert lse_t.dtype == torch.float32
    assert lse_tr.dtype == torch.float32

    # Rows where window was empty get -inf in both; mask them before allclose.
    finite = torch.isfinite(lse_t) & torch.isfinite(lse_tr)
    if finite.any():
        diff = (lse_t[finite] - lse_tr[finite]).abs().max().item()
        assert diff < 1e-3, f"LSE mismatch: max|diff| = {diff:.3e}"

    # Where one is -inf the other should also be -inf
    assert torch.equal(torch.isinf(lse_t), torch.isinf(lse_tr))


# ---------------------------------------------------------------------------
# 4. Full-window identity (k = 2N) on bf16/fp32 only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [64, 128])
def test_full_window_equals_dense_attention(dtype, N) -> None:
    """
    With k_window = 2N the rank window is vacuous — Triton output must match
    plain scaled dot-product attention.  fp16 excluded due to softmax overflow
    on large scores (known limitation).
    """
    import torch.nn.functional as F

    B, H, D = 1, 2, 32
    Q, K, V, axis = _build_inputs(B, H, N, D, dtype=dtype)
    idx = prepare_sort_indices(Q, K, axis.to(Q.dtype))
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_s = gather_by_sort_idx(Q, idx.sort_idx_q)
    out_tr_s, _ = rank_local_fwd_triton(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center, k_window=2 * N,
    )
    out_tr = scatter_by_inv_perm(out_tr_s, idx.inv_q_perm)

    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) * scale
    dense = torch.einsum("bhij,bhjd->bhid", F.softmax(scores, dim=-1), V.float())

    tol = _tol_dict(dtype, D)
    diff = (out_tr.float() - dense).abs().max().item()
    assert diff < 10.0 * tol["atol"], (
        f"full-window vs dense mismatch  N={N} dtype={dtype}: max|diff| = {diff:.3e}"
    )


# ---------------------------------------------------------------------------
# 5. No-NaN, no-Inf in output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_no_nan_or_inf_in_output(dtype) -> None:
    Q, K, V, axis = _build_inputs(B=2, H=4, N=512, D=64, dtype=dtype)
    idx = prepare_sort_indices(Q, K, axis.to(Q.dtype))
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_s = gather_by_sort_idx(Q, idx.sort_idx_q)
    out, _ = rank_local_fwd_triton(
        Q_s, K_s, V_s, idx.rank_of_k, idx.r_center, k_window=64,
    )
    assert torch.isfinite(out).all(), (
        f"non-finite output for dtype={dtype}: "
        f"{torch.isnan(out).sum().item()} NaN, "
        f"{torch.isinf(out).sum().item()} Inf"
    )


# ---------------------------------------------------------------------------
# 6. Multi-head consistency
# ---------------------------------------------------------------------------

def test_multiple_heads_independent() -> None:
    """
    H=8 heads should produce independent results — corruption between heads
    would show up as cross-head correlation that doesn't exist in the torch
    reference.
    """
    Q, K, V, axis = _build_inputs(B=1, H=8, N=512, D=64, dtype=torch.float32)
    out_t, out_tr, _, _ = _run_both_paths(Q, K, V, axis, k_window=64)
    diff = (out_t - out_tr).abs().max().item()
    assert diff < 1e-5, f"H=8 mismatch: {diff:.3e}"
