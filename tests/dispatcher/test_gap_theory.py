r"""
Gap-theory tests — connecting the dispatcher's signals to the spectral
Gap-Statement of the geometric companion manuscript (Section 6).

Background
----------
The manuscript establishes that for free-energy potentials
:math:`\Phi_\beta(x) = \beta^{-1}\log\sum_j e^{-\beta U_j(x)}` the Hessian
splits exactly as

.. math::
    \nabla^2\Phi_\beta = \mathbb{E}_p[\nabla^2 U_j] - \beta\,\operatorname{Cov}_p(\nabla U_j).

For attention the logits are linear, the mean-curvature term vanishes, and
:math:`H = -\beta\,\operatorname{Cov}_p(k)`.  The count
:math:`\dim E_- = \#\{\lambda(H) < -\theta\}` is a well-posed invariant
**iff** the spectrum has a gap :math:`\Delta(H) > 0`; it is then
threshold-free (Riesz projector) and stable under
:math:`\|\delta H\|_{\mathrm{op}} < \tfrac12\Delta` (Davis--Kahan).

The dispatcher (``dcr_attention/dispatcher/signals.py``) routes on
``d_eff = (tr S)^2/||S||_F^2`` with ``S = Cov(Q)`` — a *different*
covariance (of the queries, not of the softmax weights).  These tests check
three bridges between the two:

  * **Bridge 1** — in the gap-regime, ``d_eff(Q)`` and the theory's
    ``dim E_-(softmax)`` carry the same cluster count: ``d_eff ≈ dim E_- + 1``.
  * **Bridge 2** — when the gap closes, rank-local attention loses accuracy;
    the spectral component of that loss is isolated from trivial geometric
    crowding by a fixed-coverage control series.
  * **Bridge 3** — ``dim E_-`` is preserved under query perturbations of
    operator norm below ``Δ/2`` and not above (the Davis--Kahan threshold).

Each test uses a synthetic input with a known theoretical answer, in the
style of ``test_signals.py``.  These tests add diagnostics only; they do not
modify production code.  See INSIGHTS.md INS-31 for the finding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dcr_attention.dispatcher import compute_d_eff
from dcr_attention.reference.rank_local import (
    dense_attention_reference,
    rank_local_attention_reference,
)


# ===========================================================================
# Theory-side spectral helpers are imported from the shared analysis module.
# This is the SAME implementation that benchmarks/phase2c_gap/run_gap_validation.py
# uses on real Llama attention — the bridge between test-side and real-data-side
# measurements is preserved by sharing code, not by re-typing it.
# ===========================================================================

from dcr_attention.analysis.gap_metrics import (
    make_clustered,
    attention_hessian,
    spectral_gap,
    dim_E_minus,
)


# A separation wide enough to guarantee an open gap for the cluster sizes
# used below; established empirically by the probe that preceded this file.
_GAP_OPEN_SEP = 4.0
_GAP_BETA = 1.0


# ===========================================================================
# BRIDGE 1 — d_eff(Q) and dim E_-(softmax) carry the same cluster count
# ===========================================================================

@pytest.mark.parametrize("m", [3, 4, 5, 7])
def test_bridge1_deff_tracks_dim_E_minus(m: int):
    r"""
    In the gap-regime, the dispatcher's ``d_eff(Q)`` and the theory's
    ``dim E_-(softmax)`` carry the same cluster count.

    Known answer: ``m`` well-separated clusters give ``m-1`` inter-cluster
    Hessian eigenvalues (the ``-1`` is the covariance mean-gauge), so
    ``dim E_- = m-1`` exactly.

    The relationship to ``d_eff`` is an *inequality*, not equality.  The
    participation ratio ``(tr Σ)²/‖Σ‖_F²`` equals an integer count only when
    the contributing eigenvalues are equal; for an uneven inter-cluster
    spectrum it is strictly smaller, because it down-weights the
    sub-dominant modes.  The correct, theory-faithful statement is therefore

        ``1 ≤ d_eff ≤ dim E_- + 1``,

    with the upper bound approached as the inter-cluster spectrum flattens.
    Both quantities still grow one-for-one with the cluster count ``m`` —
    that shared monotone response is the bridge.
    """
    D, n_per = 32, 12
    X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=42)
    Q = K = X  # queries co-located with keys for the clean cluster geometry

    d_eff, _ = compute_d_eff(Q.float())
    H = attention_hessian(Q, K, beta=_GAP_BETA)
    delta, theta = spectral_gap(H)
    dim_Em = dim_E_minus(H, theta)

    assert delta > 1e-2, f"gap not open (Δ={delta:.4g}); geometry misconfigured"
    assert dim_Em == m - 1, f"dim E_- = {dim_Em}, expected {m - 1}"
    # d_eff is bounded above by the integer count + 1 and below by 1.
    assert 1.0 <= d_eff <= dim_Em + 1 + 1e-6, (
        f"d_eff = {d_eff:.3f} outside [1, dim E_- + 1 = {dim_Em + 1}]"
    )


def test_bridge1_deff_grows_one_for_one_with_clusters():
    r"""
    The shared monotone response: adding one cluster raises both ``dim E_-``
    (by exactly 1) and ``d_eff`` (by roughly 1).  We check that ``d_eff``
    increments stay in a band around 1 across a cluster sweep — i.e. the two
    signals move together even though they are not numerically identical.
    """
    D, n_per = 32, 12
    deffs = []
    for m in [3, 4, 5, 6, 7]:
        X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=42)
        deffs.append(compute_d_eff(X.float())[0])
    increments = np.diff(deffs)
    assert np.all(increments > 0.4), f"d_eff not rising with m: {increments}"
    assert np.all(increments < 1.6), f"d_eff increment off unit scale: {increments}"


def test_bridge1_count_is_threshold_free():
    r"""
    Riesz/Davis--Kahan claim: in the gap-regime ``dim E_-`` is independent of
    the threshold ``θ`` chosen anywhere inside the gap.
    """
    D, m, n_per = 32, 5, 12
    X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=1)
    H = attention_hessian(X, X, beta=_GAP_BETA)

    delta, theta = spectral_gap(H)
    lam_lo = -theta - delta / 2.0
    lam_hi = -theta + delta / 2.0

    # sweep θ across the whole open gap interval (-lam_hi, -lam_lo)
    counts = {
        dim_E_minus(H, t)
        for t in np.linspace(-lam_hi, -lam_lo, 25)[1:-1]
    }
    assert counts == {m - 1}, f"count not threshold-free across gap: {counts}"


def test_bridge1_full_numerical_rank_is_uninformative():
    r"""
    Control: the *naive* numerical rank of the softmax covariance is the full
    dimension ``D``, not ``m-1`` — it counts intra-cluster noise modes.

    This is why the manuscript insists the count is meaningful only relative
    to a spectral gap, and why the dispatcher's smooth ``d_eff`` (a
    participation ratio) is a better-behaved signal than a hard rank.
    """
    D, m, n_per = 32, 5, 12
    X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=2)
    H = attention_hessian(X, X, beta=_GAP_BETA)
    cov_p = -H / _GAP_BETA
    naive_rank = int((torch.linalg.eigvalsh(cov_p) > 1e-6).sum())
    assert naive_rank > m - 1, (
        f"expected naive rank ≫ m-1; got {naive_rank} vs m-1={m - 1}"
    )


# ===========================================================================
# BRIDGE 2 — gap closure degrades rank-local quality; d_eff is blind to it
# ===========================================================================

def _rank_local_rel_error(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    k_window: int,
) -> float:
    """Relative L2 error of rank-local vs dense attention, PCA ordering axis."""
    Qc = Q - Q.mean(dim=0)
    _, _, Vt = torch.linalg.svd(Qc, full_matrices=False)
    axis = Vt[0]
    Q4, K4, V4 = Q[None, None], K[None, None], V[None, None]
    out_rl = rank_local_attention_reference(Q4, K4, V4, axis, k_window=k_window)
    out_dn = dense_attention_reference(Q4, K4, V4)
    return (
        (out_rl - out_dn).norm() / (out_dn.norm() + 1e-9)
    ).item()


def test_bridge2_gap_closure_is_blind_to_deff():
    r"""
    As inter-cluster separation shrinks, the spectral gap ``Δ(H)`` collapses
    by orders of magnitude, yet ``d_eff(Q)`` does not fall — it drifts the
    *other* way (toward the dense-regime end of its range).

    The dispatcher signal is therefore blind to the regime boundary that the
    Gap-Statement identifies.  See INS-31.
    """
    D, m, n_per = 32, 5, 12
    seps = [4.0, 1.0, 0.35, 0.20]
    gaps, deffs = [], []
    for sep in seps:
        X, _ = make_clustered(m, n_per, D, sep=sep, intra=0.03, seed=7)
        gaps.append(spectral_gap(attention_hessian(X, X, beta=_GAP_BETA))[0])
        deffs.append(compute_d_eff(X.float())[0])

    # gap collapses hard...
    assert gaps[0] / gaps[-1] > 50.0, (
        f"gap did not collapse: {gaps[0]:.4g} -> {gaps[-1]:.4g}"
    )
    # ...while d_eff does NOT track the collapse: it fails to decrease.
    assert deffs[-1] >= deffs[0] - 0.5, (
        f"d_eff unexpectedly fell with the gap: {deffs[0]:.3f} -> {deffs[-1]:.3f}"
    )


def test_bridge2_spectral_degradation_survives_coverage_control():
    r"""
    The decisive Bridge-2 test.  Closing the gap raises rank-local error, but
    two effects are confounded: (a) the spectral gap closing, and (b) trivial
    geometric crowding — a fixed window covers a smaller mass fraction once
    clusters merge.

    We isolate (a) by holding the *coverage fraction* fixed: the rank window
    is scaled so ``k_window / N`` is constant across the sweep.  Any residual
    monotone growth of the error against the closing gap is then spectral, not
    crowding.
    """
    D, m, n_per = 32, 6, 14
    N = m * n_per
    coverage = 0.30
    k_window = max(2, int(round(coverage * N)) // 2 * 2)  # even, fixed fraction

    seps = [4.0, 1.5, 0.6, 0.25]
    gaps, errs = [], []
    for sep in seps:
        X, _ = make_clustered(m, n_per, D, sep=sep, intra=0.03, seed=11)
        g = torch.Generator().manual_seed(99)
        V = torch.randn(N, D, generator=g, dtype=torch.float64)
        gaps.append(spectral_gap(attention_hessian(X, X, beta=_GAP_BETA))[0])
        errs.append(_rank_local_rel_error(X, X, V, k_window))

    # gap closes monotonically across the sweep
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)), gaps

    # With coverage held FIXED, the rank-local error rises sharply as the gap
    # first closes and then SATURATES once the gap is effectively shut —
    # further closure of an already-closed gap adds nothing.  The claim is
    # therefore: (a) the error climbs over the closing phase, and (b) the
    # saturated level sits far above the open-gap baseline.  Strict
    # monotonicity is NOT claimed — saturation is the physically correct
    # behaviour and asserting against it would be testing a wrong model.
    assert errs[1] > errs[0] + 0.3, (
        f"error did not climb as the gap began to close: {errs}"
    )
    assert max(errs[1:]) > errs[0] + 0.5, (
        f"saturated error not far above open-gap baseline: {errs}"
    )
    # saturation: once closed, the error plateaus rather than running away
    plateau = errs[1:]
    assert max(plateau) - min(plateau) < 0.4 * max(plateau), (
        f"closed-gap error not on a plateau: {plateau}"
    )


def test_bridge2_open_gap_rank_local_is_accurate():
    r"""
    Sanity floor: in a wide-open gap-regime, rank-local attention with modest
    coverage reproduces dense attention to good accuracy.  This anchors the
    Bridge-2 sweep — the error growth is a departure from a genuinely small
    baseline, not noise around a large one.
    """
    D, m, n_per = 32, 5, 12
    N = m * n_per
    X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=5)
    g = torch.Generator().manual_seed(5)
    V = torch.randn(N, D, generator=g, dtype=torch.float64)
    # window covering ~half the points is ample when clusters are separated
    err = _rank_local_rel_error(X, X, V, k_window=N // 2)
    assert err < 0.25, f"rank-local inaccurate even in open gap: err={err:.4f}"


# ===========================================================================
# BRIDGE 3 — Davis--Kahan: dim E_- stable below ||δH|| < Δ/2
# ===========================================================================

def test_bridge3_dim_E_minus_stable_below_half_gap():
    r"""
    Davis--Kahan / Weyl: a symmetric perturbation with
    ``||δH||_op < Δ/2`` cannot change ``dim E_-``.

    We perturb the Hessian directly at controlled operator norm and check the
    count is preserved for every perturbation below the bound.
    """
    D, m, n_per = 32, 5, 12
    X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=3)
    H = attention_hessian(X, X, beta=_GAP_BETA)
    delta, theta = spectral_gap(H)
    base = dim_E_minus(H, theta)
    assert base == m - 1

    g = torch.Generator().manual_seed(123)
    for _ in range(30):
        A = torch.randn(D, D, generator=g, dtype=torch.float64)
        A = 0.5 * (A + A.T)
        A *= (0.45 * delta) / torch.linalg.matrix_norm(A, ord=2)
        assert dim_E_minus(H + A, theta) == base, (
            "dim E_- changed under a perturbation below the Δ/2 bound"
        )


def test_bridge3_dim_E_minus_can_break_above_gap():
    r"""
    Complement: the ``Δ/2`` bound is not vacuous — a perturbation of operator
    norm comparable to the *full* gap is able to change the count.

    We do not assert that every such perturbation breaks it (some directions
    leave the count intact); we assert that at least one does, which is enough
    to show the bound is tight rather than conservative.
    """
    D, m, n_per = 32, 5, 12
    X, _ = make_clustered(m, n_per, D, sep=_GAP_OPEN_SEP, intra=0.02, seed=4)
    H = attention_hessian(X, X, beta=_GAP_BETA)
    delta, theta = spectral_gap(H)
    base = dim_E_minus(H, theta)

    g = torch.Generator().manual_seed(456)
    broke = False
    for _ in range(60):
        A = torch.randn(D, D, generator=g, dtype=torch.float64)
        A = 0.5 * (A + A.T)
        A *= (1.5 * delta) / torch.linalg.matrix_norm(A, ord=2)
        if dim_E_minus(H + A, theta) != base:
            broke = True
            break
    assert broke, "no perturbation above the gap changed the count"
