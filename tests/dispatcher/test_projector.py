"""
Tests for the learnable axis projector.

Three layers:

  1. Plumbing — shapes, parameter counts, validation errors.
  2. Math — pca_axis_target on rank-1, cosine_alignment_loss properties.
  3. **Learning** — train projector on synthetic rank-1 / rank-4 data and
     assert it reaches the acceptance thresholds from
     docs/design/phase1_axis_projector.md §6.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dcr_attention.dispatcher.projector import (
    AxisProjector,
    axis_cosine_similarity,
    cosine_alignment_loss,
    pca_axis_target,
    synthetic_rank_k_batch,
)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_projector_forward_shape():
    proj = AxisProjector(head_dim=32, num_heads=4, hidden_mult=2)
    Q = torch.randn(2, 4, 64, 32)
    u = proj(Q)
    assert u.shape == (2, 4, 32)


def test_projector_output_is_unit_norm():
    proj = AxisProjector(head_dim=16, num_heads=2)
    proj.eval()
    Q = torch.randn(3, 2, 128, 16)
    u = proj(Q)
    norms = u.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_projector_rejects_wrong_head_count():
    proj = AxisProjector(head_dim=8, num_heads=2)
    Q = torch.randn(1, 3, 16, 8)        # 3 heads, projector wants 2
    with pytest.raises(ValueError):
        proj(Q)


def test_projector_rejects_wrong_head_dim():
    proj = AxisProjector(head_dim=8, num_heads=2)
    Q = torch.randn(1, 2, 16, 12)       # D=12, projector wants 8
    with pytest.raises(ValueError):
        proj(Q)


def test_projector_rejects_non_4d_input():
    proj = AxisProjector(head_dim=8, num_heads=2)
    with pytest.raises(ValueError):
        proj(torch.randn(2, 16, 8))     # missing head dim


def test_projector_parameter_count():
    """
    Per-head MLP with covariance pooling:
      input_dim = D(D+1)/2 + D     # upper-tri Σ + u_pca
      Linear(input_dim → D_hid):  input_dim·D_hid + D_hid
      Linear(D_hid → D):           D_hid·D + D
    Total per head, times H.
    """
    D, H, mult = 32, 4, 2
    proj = AxisProjector(head_dim=D, num_heads=H, hidden_mult=mult)
    n_total = sum(p.numel() for p in proj.parameters())
    D_hid = mult * D
    feat_dim = D * (D + 1) // 2 + D
    expected_per_head = feat_dim * D_hid + D_hid + D_hid * D + D
    assert n_total == H * expected_per_head, (
        f"got {n_total}, expected {H * expected_per_head}"
    )


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def test_pca_target_recovers_rank_one_axis():
    """Strong rank-1 signal: PCA axis matches the planted direction."""
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 256, 16
    u_planted = torch.randn(B, H, D)
    u_planted = u_planted / u_planted.norm(dim=-1, keepdim=True)
    scalars = torch.randn(B, H, N, 1)
    Q = 5.0 * scalars * u_planted.unsqueeze(2) + 0.1 * torch.randn(B, H, N, D)
    u_pca = pca_axis_target(Q)

    cos_abs = axis_cosine_similarity(u_pca, u_planted, sign_agnostic=True)
    assert (cos_abs > 0.99).all(), f"cos_abs = {cos_abs.tolist()}"


def test_pca_target_is_unit_norm():
    torch.manual_seed(0)
    Q = torch.randn(2, 3, 64, 16)
    u = pca_axis_target(Q)
    assert torch.allclose(u.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_cosine_alignment_loss_zero_at_match():
    torch.manual_seed(0)
    u = torch.randn(2, 3, 16)
    u = u / u.norm(dim=-1, keepdim=True)
    loss = cosine_alignment_loss(u, u)
    # 1 - cos² has fp32 epsilon ~ 1e-7 even at exact match
    assert loss.item() < 1e-6


def test_cosine_alignment_loss_zero_at_flipped_axis():
    """Sign-agnostic: u and -u must give the same loss as u and u."""
    torch.manual_seed(0)
    u = torch.randn(2, 3, 16)
    u = u / u.norm(dim=-1, keepdim=True)
    loss = cosine_alignment_loss(u, -u)
    assert loss.item() < 1e-6


def test_cosine_alignment_loss_one_at_orthogonal():
    """Orthogonal vectors → loss = 1."""
    u = torch.zeros(1, 1, 4); u[..., 0] = 1.0
    v = torch.zeros(1, 1, 4); v[..., 1] = 1.0
    loss = cosine_alignment_loss(u, v)
    assert abs(loss.item() - 1.0) < 1e-6


def test_loss_shape_validation():
    with pytest.raises(ValueError):
        cosine_alignment_loss(torch.randn(2, 3, 8), torch.randn(2, 3, 16))


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def test_synthetic_batch_shapes():
    Q, u = synthetic_rank_k_batch(batch_size=2, num_heads=3, seq_len=64,
                                  head_dim=16, rank=2)
    assert Q.shape == (2, 3, 64, 16)
    assert u.shape == (2, 3, 16)
    assert torch.allclose(u.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_synthetic_batch_reproducible_with_generator():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    Q1, _ = synthetic_rank_k_batch(2, 2, 32, 8, rank=1, generator=g1)
    Q2, _ = synthetic_rank_k_batch(2, 2, 32, 8, rank=1, generator=g2)
    assert torch.equal(Q1, Q2)


def test_synthetic_rejects_invalid_rank():
    with pytest.raises(ValueError):
        synthetic_rank_k_batch(1, 1, 16, 8, rank=0)
    with pytest.raises(ValueError):
        synthetic_rank_k_batch(1, 1, 16, 8, rank=10)         # > head_dim


# ---------------------------------------------------------------------------
# **Learning** — acceptance criteria §6
# ---------------------------------------------------------------------------

def _train_projector(
    rank: int,
    snr: float,
    n_steps: int = 200,
    lr: float = 3e-3,
    head_dim: int = 16,
    num_heads: int = 2,
    batch_size: int = 16,
    seq_len: int = 128,
    seed: int = 0,
) -> tuple[AxisProjector, list[float]]:
    """
    Train an AxisProjector on synthetic rank-`rank` batches.  Returns the
    trained projector and the list of per-step training losses.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)

    proj = AxisProjector(head_dim=head_dim, num_heads=num_heads, hidden_mult=4)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(n_steps):
        Q, u_target = synthetic_rank_k_batch(
            batch_size, num_heads, seq_len, head_dim,
            rank=rank, snr=snr, generator=g,
        )
        u_hat = proj(Q)
        loss = cosine_alignment_loss(u_hat, u_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return proj, losses


def _eval_axis_quality(
    proj: AxisProjector,
    rank: int,
    snr: float,
    n_eval: int = 64,
    head_dim: int = 16,
    num_heads: int = 2,
    seed: int = 1234,
) -> float:
    """Mean held-out cosine similarity (sign-agnostic)."""
    g = torch.Generator().manual_seed(seed)
    proj.eval()
    with torch.no_grad():
        Q, u_target = synthetic_rank_k_batch(
            n_eval, num_heads, 128, head_dim,
            rank=rank, snr=snr, generator=g,
        )
        u_hat = proj(Q)
        cos = axis_cosine_similarity(u_hat, u_target, sign_agnostic=True)
    return cos.mean().item()


def test_projector_learns_rank_one_axis():
    """
    Acceptance §6: rank-1 synthetic, cos-sim ≥ 0.95 on held-out batches.
    """
    proj, losses = _train_projector(rank=1, snr=5.0, n_steps=200)
    cos_eval = _eval_axis_quality(proj, rank=1, snr=5.0)
    assert cos_eval >= 0.95, (
        f"rank-1: held-out cos-sim {cos_eval:.4f} below 0.95.  "
        f"Final 5 train losses: {losses[-5:]}"
    )


def test_projector_learns_rank_four_axis():
    """
    Acceptance §6: rank-4 synthetic, cos-sim ≥ 0.85 on held-out batches.
    """
    proj, losses = _train_projector(rank=4, snr=5.0, n_steps=400)
    cos_eval = _eval_axis_quality(proj, rank=4, snr=5.0)
    assert cos_eval >= 0.85, (
        f"rank-4: held-out cos-sim {cos_eval:.4f} below 0.85.  "
        f"Final 5 train losses: {losses[-5:]}"
    )


def test_projector_init_matches_pca_baseline():
    """
    Critical architectural check: with zero-initialized residual head, the
    projector's output equals the sign-fixed PCA axis on input data.
    This means training starts at the right baseline.
    """
    torch.manual_seed(0)
    proj = AxisProjector(head_dim=16, num_heads=2, hidden_mult=2)
    proj.eval()

    g = torch.Generator().manual_seed(1)
    Q, u_target = synthetic_rank_k_batch(8, 2, 128, 16, rank=1, snr=5.0, generator=g)

    with torch.no_grad():
        u_hat = proj(Q)
    cos = axis_cosine_similarity(u_hat, u_target, sign_agnostic=True).mean().item()
    assert cos > 0.999, (
        f"at init the residual head is zero, projector should equal PCA baseline; "
        f"got cos = {cos:.6f}"
    )


def test_projector_training_does_not_diverge():
    """
    Because the projector starts at the PCA optimum (loss ≈ 0), training is
    unlikely to *decrease* the loss noticeably — but it must not *increase*
    it catastrophically.  We assert the late-window loss is within 5× of the
    early-window loss.  This is a regression guard against bad init / bad lr,
    not a learning-progress claim.
    """
    _, losses = _train_projector(rank=1, snr=5.0, n_steps=200)
    early = sum(losses[:50]) / 50
    late = sum(losses[-50:]) / 50
    assert late < 5.0 * (early + 1e-6), (
        f"training diverged: early avg {early:.6f} → late avg {late:.6f}"
    )
