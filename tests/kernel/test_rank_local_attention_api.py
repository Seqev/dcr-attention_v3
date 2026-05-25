"""
Tests for the public rank_local_attention API.

Cover:
  * forward correctness (via torch fallback; CUDA path exercised on Seqev's box).
  * self-adaptive ctx: nothing saved when no input requires grad.
  * backward placeholder raises NotImplementedError cleanly when grad required.
  * axis normalisation (caller may pass unnormalised).
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel import (
    RankLocalAttentionFn,
    rank_local_attention,
    TRITON_AVAILABLE,
)
from dcr_attention.reference import rank_local_attention_reference


# ---------------------------------------------------------------------------
# Forward correctness (fallback path on CPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k_window", [16, 64, 256])
def test_public_api_matches_reference(k_window):
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 128, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis = torch.randn(D); axis = axis / axis.norm()

    out = rank_local_attention(Q, K, V, axis, k_window=k_window)
    ref = rank_local_attention_reference(Q, K, V, axis, k_window=k_window)
    assert torch.allclose(out, ref, atol=1e-5)


def test_axis_is_normalised_internally():
    """Caller may pass an unnormalised axis; the function must normalise it."""
    torch.manual_seed(0)
    Q = torch.randn(1, 2, 64, 16)
    K = torch.randn(1, 2, 64, 16)
    V = torch.randn(1, 2, 64, 16)
    axis_unit = torch.randn(16); axis_unit = axis_unit / axis_unit.norm()
    axis_scaled = 7.3 * axis_unit                   # same direction, different norm

    out_unit = rank_local_attention(Q, K, V, axis_unit, k_window=32)
    out_scaled = rank_local_attention(Q, K, V, axis_scaled, k_window=32)
    assert torch.allclose(out_unit, out_scaled, atol=1e-6)


# ---------------------------------------------------------------------------
# Self-adaptive ctx (P1)
# ---------------------------------------------------------------------------

def test_no_grad_skips_save_for_backward():
    """
    Forward with no-grad tensors must not populate ctx.saved_tensors.
    We verify via a wrapping trick: run forward inside `torch.no_grad()` — no
    ctx is created at all.  Stronger check: run with leaves requires_grad=False
    and inspect ctx.needs_backward == False.
    """
    torch.manual_seed(0)
    Q = torch.randn(1, 1, 32, 8)
    K = torch.randn(1, 1, 32, 8)
    V = torch.randn(1, 1, 32, 8)
    axis = torch.zeros(8); axis[0] = 1.0

    # Run forward directly via the Function to capture ctx
    captured = {}

    class SpyFn(RankLocalAttentionFn):
        @staticmethod
        def forward(ctx, *args, **kw):
            out = RankLocalAttentionFn.forward(ctx, *args, **kw)
            captured["needs_backward"] = ctx.needs_backward
            captured["saved"] = getattr(ctx, "saved_tensors", None)
            return out

    _ = SpyFn.apply(Q, K, V, axis, 16, None)
    assert captured["needs_backward"] is False


def test_grad_path_populates_ctx_and_backward_raises():
    """
    When an input requires grad, ctx is populated and backward raises a
    clean NotImplementedError (Phase 1.2 contract).
    """
    torch.manual_seed(0)
    Q = torch.randn(1, 1, 32, 8, requires_grad=True)
    K = torch.randn(1, 1, 32, 8)
    V = torch.randn(1, 1, 32, 8)
    axis = torch.zeros(8); axis[0] = 1.0

    out = rank_local_attention(Q, K, V, axis, k_window=16)
    assert out.requires_grad
    with pytest.raises(NotImplementedError):
        out.sum().backward()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_triton_flag_reflects_import_state():
    """
    Regression guard: TRITON_AVAILABLE is a compile-time flag set once at
    import. On this machine it may be True or False depending on install;
    either is fine as long as the flag is a bool.
    """
    assert isinstance(TRITON_AVAILABLE, bool)


def test_invalid_axis_shape_rejected():
    Q = K = V = torch.randn(1, 1, 32, 8)
    axis_bad = torch.randn(5)
    with pytest.raises(ValueError):
        rank_local_attention(Q, K, V, axis_bad, k_window=16)
