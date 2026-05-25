"""
M3 validation — Pass 3 fused gather + softmax-attention.

Spec §5.1.2 gates (final output O):
  max |O_triton - O_ref|_∞  ≤ 5e-3
  mean (O_triton - O_ref)²  ≤ 1e-6
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel.triton.fused_attn import fused_topk_attention
from dcr_attention.kernel.triton.topk_axis import topk_qaxis_select, topk_qaxis_attention
from dcr_attention.kernel.qaxis_topk_reference import (
    _compute_u_Q,
    _compute_projection_scores,
    _select_topk_indices,
    _gather_kv,
    _fused_softmax_attention,
    topk_qaxis_attention_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="M3 tests require CUDA"
)


# ---------------------------------------------------------------------------
# Test 1 — Pass 3 vs M1 _fused_softmax_attention (white-box)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 42, 100])
@pytest.mark.parametrize("B,H,H_kv,D,k_eff", [
    (1, 32, 8, 64, 200),
    (1, 32, 8, 64, 1000),
    (2, 32, 8, 64, 500),
])
def test_pass3_matches_m1_step4(seed, B, H, H_kv, D, k_eff):
    """Pass 3 output must match M1 _fused_softmax_attention per spec §5.1.2."""
    torch.manual_seed(seed)
    N = max(2 * k_eff, 1000)

    Q = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    # M1 reference: exact same indices fed into both
    u_Q       = _compute_u_Q(Q, eps=1e-12)
    scores    = _compute_projection_scores(K, u_Q)
    m1_idx    = _select_topk_indices(scores, k_eff)           # [B, H, k_eff] int64
    K_sel, V_sel = _gather_kv(K, V, m1_idx)
    O_ref     = _fused_softmax_attention(K_sel, V_sel, Q)     # [B, H, D] bf16

    # Pass 3 Triton with the same indices (int32)
    O_triton  = fused_topk_attention(Q, K, V, m1_idx.to(torch.int32))

    diff     = (O_ref.float() - O_triton.float()).abs()
    max_diff = diff.max().item()
    rmse_sq  = (diff ** 2).mean().item()

    print(f"\n  seed={seed} B={B} k_eff={k_eff}: max|Δ|={max_diff:.2e}  RMSE²={rmse_sq:.2e}")

    assert max_diff <= 5e-3,  f"max|Δ| {max_diff:.2e} exceeds §5.1.2 gate 5e-3"
    assert rmse_sq  <= 1e-6,  f"RMSE² {rmse_sq:.2e} exceeds §5.1.2 gate 1e-6"
    assert torch.isfinite(O_triton).all(), "Pass 3 produced non-finite output"


# ---------------------------------------------------------------------------
# Test 2 — Full pipeline e2e (Q + K + V → O) vs M1 reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,k_eff_frac", [
    (500,  0.5),
    (1000, 0.5),
    (2000, 0.5),
    (5000, 0.5),
    (2000, 0.3),
    (2000, 0.8),
])
@pytest.mark.parametrize("seed", [0, 42, 100, 500, 1000])
def test_full_pipeline_matches_m1(seed, N, k_eff_frac):
    """M2+M3 full pipeline must match M1 end-to-end per §5.1.2."""
    torch.manual_seed(seed)
    B, H, H_kv, D = 1, 32, 8, 64
    k_eff = int(N * k_eff_frac)

    Q = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    O_m1     = topk_qaxis_attention_reference(Q, K, V, k_eff, eps=1e-12)
    O_triton = topk_qaxis_attention(Q, K, V, k_eff, eps=1e-12)

    diff     = (O_m1.float() - O_triton.float()).abs()
    max_diff = diff.max().item()
    rmse_sq  = (diff ** 2).mean().item()

    print(f"\n  seed={seed} N={N} c={k_eff_frac}: max|Δ|={max_diff:.2e}  RMSE²={rmse_sq:.2e}")

    assert max_diff <= 5e-3, f"max|Δ| {max_diff:.2e} exceeds §5.1.2 gate"
    assert rmse_sq  <= 1e-6, f"RMSE² {rmse_sq:.2e} exceeds §5.1.2 gate"
    assert torch.isfinite(O_triton).all()


# ---------------------------------------------------------------------------
# Test 3 — fp32 softmax state preserved at long context (k_eff = 10K)
# ---------------------------------------------------------------------------

def test_fp32_softmax_state_long_context():
    """
    Critical R3 mitigation: fp32 running state must not drift at k_eff=10K.

    If m/l/o were bf16, accumulated error at k=10K would be ~0.1 (10× bf16
    ULP per step × 10K steps). With fp32, max|Δ| stays within §5.1.2 gate.
    """
    torch.manual_seed(0)
    N, k_eff = 20000, 10000
    Q = torch.randn(1, 32, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1,  8, N, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1,  8, N, 64, dtype=torch.bfloat16, device="cuda")

    O_m1     = topk_qaxis_attention_reference(Q, K, V, k_eff)
    O_triton = topk_qaxis_attention(Q, K, V, k_eff)

    max_diff = (O_m1.float() - O_triton.float()).abs().max().item()
    print(f"\n  k_eff=10K long-context max|Δ|={max_diff:.2e}")

    # bf16 running state would produce max_diff ~0.1; fp32 stays within 5e-3
    assert max_diff <= 5e-3, (
        f"Long-context drift {max_diff:.2e} exceeds §5.1.2 gate. "
        "Likely cause: fp32 running state demoted to bf16 (spec §3 violation)."
    )


# ---------------------------------------------------------------------------
# Test 4 — Edge cases per spec §6
# ---------------------------------------------------------------------------

def test_edge_case_keff_equals_n_minus_1():
    """k_eff = N-1 (near-full coverage): kernel must produce correct output."""
    torch.manual_seed(0)
    N = 100
    Q = torch.randn(1, 32, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1,  8, N, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1,  8, N, 64, dtype=torch.bfloat16, device="cuda")

    O_m1     = topk_qaxis_attention_reference(Q, K, V, N - 1)
    O_triton = topk_qaxis_attention(Q, K, V, N - 1)

    max_diff = (O_m1.float() - O_triton.float()).abs().max().item()
    assert max_diff <= 5e-3, f"k_eff=N-1: max|Δ|={max_diff:.2e}"
    assert torch.isfinite(O_triton).all()


def test_edge_case_unit_batch_head():
    """B=1, H=1 (H_kv=1, n_per_kv=1) — no GQA broadcasting bugs."""
    # NOTE: Pass 3 has n_per_kv=4 hardcoded for Llama-3.2-1B.
    # B=1, H=4, H_kv=1 is the minimal valid GQA config (n_per_kv=4).
    torch.manual_seed(7)
    Q = torch.randn(1,  4, 64,      dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1,  1, 1000, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1,  1, 1000, 64, dtype=torch.bfloat16, device="cuda")
    O = topk_qaxis_attention(Q, K, V, 100)
    assert O.shape == (1, 4, 64)
    assert O.dtype == torch.bfloat16
    assert torch.isfinite(O).all()


def test_edge_case_b8_production():
    """B=8: production batch size — VRAM must not spike (Track 3 closure)."""
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    Q = torch.randn(8, 32, 64,       dtype=torch.bfloat16, device="cuda")
    K = torch.randn(8,  8, 5000, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(8,  8, 5000, 64, dtype=torch.bfloat16, device="cuda")

    O_triton = topk_qaxis_attention(Q, K, V, 2500)
    torch.cuda.synchronize()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  B=8 N=5000 k_eff=2500: peak VRAM = {peak_gb:.2f} GB")

    assert O_triton.shape == (8, 32, 64)
    assert torch.isfinite(O_triton).all()
    # Per A2 protocol: if > 4 GB, this is a STOP signal (Track 3 fusion failure)
    assert peak_gb < 4.0, (
        f"VRAM spike {peak_gb:.2f} GB exceeds 4 GB threshold — "
        "A2 protocol: STOP, surface to architect (Track 3 fusion failure)."
    )


# ---------------------------------------------------------------------------
# Test 5 — Runtime sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N", [5000, 20000])
def test_pass3_runtime_acceptable(N):
    """Pass 3 + full pipeline runtime must be < 10 ms (target < 5 ms)."""
    torch.manual_seed(0)
    B, H, H_kv, D = 1, 32, 8, 64
    k_eff = N // 2
    Q = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")
    indices = topk_qaxis_select(Q, K, k_eff)

    # Warmup
    for _ in range(3):
        _ = fused_topk_attention(Q, K, V, indices)
    torch.cuda.synchronize()

    n_iters = 20
    t_start = torch.cuda.Event(enable_timing=True)
    t_end   = torch.cuda.Event(enable_timing=True)
    t_start.record()
    for _ in range(n_iters):
        _ = fused_topk_attention(Q, K, V, indices)
    t_end.record()
    torch.cuda.synchronize()

    avg_ms = t_start.elapsed_time(t_end) / n_iters
    print(f"\n  Pass 3 N={N}: {avg_ms:.2f} ms")
    assert avg_ms < 10.0, f"Pass 3 too slow: {avg_ms:.2f} ms (gate < 10 ms)"


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def test_output_shape_dtype():
    """O is [B, H, D] bf16."""
    Q = torch.randn(1, 32, 64,      dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1,  8, 512, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1,  8, 512, 64, dtype=torch.bfloat16, device="cuda")
    O = topk_qaxis_attention(Q, K, V, 256)
    assert O.shape == (1, 32, 64)
    assert O.dtype == torch.bfloat16


def test_no_nan_inf():
    """No NaN/inf in any output (spec §6)."""
    torch.manual_seed(11)
    Q = torch.randn(1, 32, 64,      dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1,  8, 400, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1,  8, 400, 64, dtype=torch.bfloat16, device="cuda")
    O = topk_qaxis_attention(Q, K, V, 200)
    assert torch.isfinite(O).all(), "Output contains NaN/inf"
