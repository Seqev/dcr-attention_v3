"""
PATCH-08 S2 — Permutation equivariance.

For any permutation π over the sequence dimension N:

    rank_local_attention(π·Q, π·K, π·V, axis)
        ≡ π · rank_local_attention(Q, K, V, axis).

This is a structural property of the algorithm.  A failure here points to
a bug in ``prepare_sort_indices`` or ``gather_by_sort_idx`` that other
tests can miss because they do not exercise permutation symmetry.

We test over multiple random permutations to ensure π is genuinely a
nontrivial reordering.  Note that the dispatcher is *not* in scope here —
we are testing the kernel's algebraic property, not the routing decision.
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel import rank_local_attention


def _random_permutation(N: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(N, generator=g)


def _apply_perm(X: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """X[..., perm, :] along the N axis (assumed dim=-2)."""
    return X.index_select(dim=-2, index=perm)


@pytest.mark.parametrize("k_window", [16, 64, 256])
@pytest.mark.parametrize("perm_seed", [0, 1, 2, 7, 42])
def test_permutation_equivariance(k_window: int, perm_seed: int) -> None:
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 128, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis = torch.randn(D); axis = axis / axis.norm()

    perm = _random_permutation(N, perm_seed)
    inv_perm = torch.argsort(perm)

    # π · Q means apply π to the sequence dim of Q
    Qp = _apply_perm(Q, perm)
    Kp = _apply_perm(K, perm)
    Vp = _apply_perm(V, perm)

    out_original = rank_local_attention(Q, K, V, axis, k_window=k_window)
    out_permuted = rank_local_attention(Qp, Kp, Vp, axis, k_window=k_window)

    # Equivariance: out_permuted should equal π applied to out_original
    out_original_then_permuted = _apply_perm(out_original, perm)

    max_abs = (out_permuted - out_original_then_permuted).abs().max().item()
    assert max_abs < 1e-5, (
        f"k={k_window}, perm_seed={perm_seed}: "
        f"max|π·attn(x) - attn(π·x)| = {max_abs:g} > 1e-5"
    )


def test_identity_permutation_is_noop() -> None:
    """Sanity edge case: π = identity must give exactly the same output."""
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 64, 16
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis = torch.zeros(D); axis[0] = 1.0
    perm = torch.arange(N)

    out1 = rank_local_attention(Q, K, V, axis, k_window=32)
    Qp, Kp, Vp = _apply_perm(Q, perm), _apply_perm(K, perm), _apply_perm(V, perm)
    out2 = rank_local_attention(Qp, Kp, Vp, axis, k_window=32)
    assert torch.equal(out1, out2)


def test_reverse_permutation_equivariance() -> None:
    """
    Reverse permutation π(i) = N-1-i is the most adversarial test for
    sort-and-gather correctness.
    """
    torch.manual_seed(7)
    B, H, N, D = 1, 1, 64, 16
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis = torch.randn(D); axis = axis / axis.norm()
    perm = torch.arange(N - 1, -1, -1)

    out_orig = rank_local_attention(Q, K, V, axis, k_window=16)
    out_perm = rank_local_attention(
        _apply_perm(Q, perm), _apply_perm(K, perm), _apply_perm(V, perm),
        axis, k_window=16,
    )
    expected = _apply_perm(out_orig, perm)
    assert torch.allclose(out_perm, expected, atol=1e-5)
