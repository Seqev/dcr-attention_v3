"""
Learnable axis projector for Stage B.

Architecture (per docs/design/phase1_axis_projector.md):
    h_summary = mean(Q, dim=N)                # [B, H, D]   pooled query stats
    raw       = MLP(h_summary)                # [B, H, D]
    axis_hat  = raw / ||raw||                 # unit norm

Training: surrogate rank-prediction loss against PCA axis on calibration data.
No backward through attention kernel required.

INS-6 follow-up: replaces per-forward SVD with O(D²) projector, preserves
per-input axis adaptivity.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


_EPS = 1e-12


def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unit-norm along ``dim``.  Numerically safe at zero vectors."""
    return x / (x.norm(dim=dim, keepdim=True) + _EPS)


class AxisProjector(nn.Module):
    r"""
    Maps query covariance :math:`\Sigma_{bh} = \frac{1}{N} Q_{bh}^\top Q_{bh}`
    to a unit-norm ordering axis :math:`\hat u_{bh} \in \mathbb{R}^D`.

    .. math::
        \Sigma_{bh} &= \tfrac{1}{N}\, Q_{bh}^\top Q_{bh}\;\in\mathbb{R}^{D\times D}, \\
        u^{\text{PCA}}_{bh} &= \mathrm{sign\_fix}(\mathrm{top\_eigvec}(\Sigma_{bh})), \\
        \mathrm{feats}_{bh} &= [\mathrm{upper\_tri}(\Sigma_{bh}) \,\|\, u^{\text{PCA}}_{bh}], \\
        \delta_{bh} &= \mathrm{MLP}(\mathrm{feats}_{bh}), \\
        \hat u_{bh} &= \frac{u^{\text{PCA}}_{bh} + \delta_{bh}}
                            {\|u^{\text{PCA}}_{bh} + \delta_{bh}\|_2}.

    The MLP predicts a **residual** from PCA, not the axis from scratch.
    At init (small random weights) it produces ``δ ≈ 0`` so the projector
    starts near the PCA baseline.

    See ``docs/design/phase1_axis_projector.md`` §2 for the rationale and
    INS-10 for the design history (mean-pooling fails because PCA signal
    lives in the covariance, not the mean).

    Per-head weights — different heads specialise.

    Parameters
    ----------
    head_dim
        :math:`D`.
    num_heads
        :math:`H`.
    hidden_mult
        Width of the MLP hidden layer as a multiple of ``head_dim``.
        Default 2 — the input is already PCA-loaded so we don't need wide.
    dropout
        Hidden-layer dropout, off by default.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        hidden_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads

        D = head_dim
        D_hid = hidden_mult * D
        feat_dim = D * (D + 1) // 2 + D                # upper-tri Σ + u_pca

        # Per-head MLPs as 3D parameter tensors.
        self.W1 = nn.Parameter(torch.empty(num_heads, feat_dim, D_hid))
        self.b1 = nn.Parameter(torch.zeros(num_heads, D_hid))
        self.W2 = nn.Parameter(torch.empty(num_heads, D_hid, D))
        self.b2 = nn.Parameter(torch.zeros(num_heads, D))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

        # Cache the upper-triangle indices once (constant, shape [feat - D, 2])
        ut = torch.triu_indices(D, D)                  # [2, D(D+1)/2]
        self.register_buffer("_ut_rows", ut[0], persistent=False)
        self.register_buffer("_ut_cols", ut[1], persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard Kaiming for hidden layer.
        nn.init.kaiming_uniform_(self.W1, a=5 ** 0.5)
        # Zero-init the residual head: δ ≈ 0 at start → projector ≡ PCA.
        nn.init.zeros_(self.W2)
        nn.init.zeros_(self.b2)

    @staticmethod
    def _covariance(Q: torch.Tensor) -> torch.Tensor:
        r"""
        Per-(b, h) second-moment matrix :math:`\Sigma = Q^\top Q / N`.

        We use the second moment (un-centred) rather than the centred
        covariance because the rank-local algorithm projects onto an axis
        and orders by inner product — the mean of :math:`Q` is irrelevant
        to the ordering, so removing or keeping it does not change the
        target axis. Skipping the mean-subtraction is one less reduction.

        Parameters
        ----------
        Q
            ``[B, H, N, D]``.

        Returns
        -------
        ``[B, H, D, D]``.
        """
        N = Q.shape[2]
        return torch.einsum("bhnd,bhne->bhde", Q, Q) / N

    @staticmethod
    def _sign_fix(u: torch.Tensor) -> torch.Tensor:
        r"""
        Resolve eigenvector sign ambiguity by forcing the largest-magnitude
        entry to be positive.

        ``u`` shape: ``[..., D]``.  Returns the same tensor or its negation
        per row.
        """
        idx = u.abs().argmax(dim=-1, keepdim=True)         # [..., 1]
        sign = torch.sign(u.gather(-1, idx))               # [..., 1]
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return u * sign

    def _features(self, Sigma: torch.Tensor, u_pca: torch.Tensor) -> torch.Tensor:
        r"""
        Concatenate flattened upper triangle of Σ with the PCA axis.

        Parameters
        ----------
        Sigma
            ``[B, H, D, D]``.
        u_pca
            ``[B, H, D]``.

        Returns
        -------
        ``[B, H, D(D+1)/2 + D]``.
        """
        # Upper triangle: shape [B, H, D(D+1)/2]
        ut = Sigma[..., self._ut_rows, self._ut_cols]
        return torch.cat([ut, u_pca], dim=-1)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the projected axis.

        Parameters
        ----------
        Q
            ``[B, H, N, D]``.

        Returns
        -------
        ``[B, H, D]``, unit-norm.
        """
        if Q.dim() != 4:
            raise ValueError(f"expected [B,H,N,D], got {tuple(Q.shape)}")
        if Q.shape[1] != self.num_heads:
            raise ValueError(
                f"Q has {Q.shape[1]} heads but projector built for {self.num_heads}"
            )
        if Q.shape[-1] != self.head_dim:
            raise ValueError(
                f"Q has D={Q.shape[-1]} but projector built for D={self.head_dim}"
            )

        Sigma = self._covariance(Q)                            # [B, H, D, D]
        # eigh returns eigenvalues ascending; top eigenvector is at index -1
        eigvals, eigvecs = torch.linalg.eigh(Sigma)
        u_pca_raw = eigvecs[..., -1]                           # [B, H, D]
        u_pca = self._sign_fix(u_pca_raw)

        feats = self._features(Sigma, u_pca)                   # [B, H, F]
        x = torch.einsum("bhf,hfe->bhe", feats, self.W1) + self.b1
        x = self.act(x)
        x = self.dropout(x)
        delta = torch.einsum("bhe,hed->bhd", x, self.W2) + self.b2

        return _l2_normalize(u_pca + delta, dim=-1)


# ---------------------------------------------------------------------------
# Targets and losses
# ---------------------------------------------------------------------------

def pca_axis_target(Q: torch.Tensor) -> torch.Tensor:
    r"""
    Compute the per-(b, h) PCA axis (top right-singular vector of centred Q)
    to use as a training target.

    .. math::
        u^{\text{target}}_{bh} \;=\; V^{(1)}_{bh}, \quad
            \text{where } \bar Q_{bh} = U_{bh}\, \Sigma_{bh}\, V_{bh}^{\!\top}.

    Returns
    -------
    ``[B, H, D]``, unit-norm.
    """
    if Q.dim() != 4:
        raise ValueError(f"expected [B,H,N,D], got {tuple(Q.shape)}")
    Qc = Q - Q.mean(dim=2, keepdim=True)
    # SVD per (b, h) head — torch batches over leading dims.
    _, _, Vt = torch.linalg.svd(Qc, full_matrices=False)        # Vt: [B, H, D, D]
    u = Vt[..., 0, :]                                           # [B, H, D]
    return _l2_normalize(u, dim=-1)


def cosine_alignment_loss(
    u_hat: torch.Tensor,
    u_target: torch.Tensor,
) -> torch.Tensor:
    r"""
    Sign-agnostic cosine alignment loss.

    .. math::
        \mathcal{L} \;=\;
            \mathbb{E}_{b,h}\!\left[1 - \langle \hat u_{bh}, u^{\text{target}}_{bh}\rangle^2\right].

    The square removes the sign ambiguity (``rank-local is invariant under
    axis flip``).  The gradient near alignment is smoother than the
    ``1 - |⟨·,·⟩|`` variant and easier on Adam at small lr.

    Both inputs assumed unit-norm along the last dim.
    """
    if u_hat.shape != u_target.shape:
        raise ValueError(
            f"shape mismatch {tuple(u_hat.shape)} vs {tuple(u_target.shape)}"
        )
    cos = (u_hat * u_target).sum(dim=-1)                        # [B, H]
    return (1.0 - cos.pow(2)).mean()


def axis_cosine_similarity(
    u_hat: torch.Tensor,
    u_target: torch.Tensor,
    sign_agnostic: bool = True,
) -> torch.Tensor:
    r"""
    Diagnostic: per-(b, h) cosine similarity, optionally absolute-valued.

    Returns
    -------
    ``[B, H]``.
    """
    cos = (u_hat * u_target).sum(dim=-1)
    return cos.abs() if sign_agnostic else cos


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def synthetic_rank_k_batch(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    rank: int = 1,
    snr: float = 5.0,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Generate ``Q`` with a rank-``k`` structure plus isotropic noise, and the
    corresponding PCA axis target.

    .. math::
        Q_{bhi} \;=\; \mathrm{snr} \cdot \sum_{r=1}^{\text{rank}} s_{bhi r}\, u^{(r)}_{bh}
                    \;+\; \xi_{bhi},
        \qquad u^{(r)}_{bh}, s_{bhir}, \xi_{bhi} \sim \mathcal{N}(0, I).

    The :math:`u^{(1)}` direction is the dominant PCA axis when ``snr`` is
    large enough; the routine returns ``pca_axis_target(Q)`` rather than
    ``u^{(1)}`` directly so the supervision is consistent with the eventual
    real-model regime where ground truth = PCA of activations.

    Returns
    -------
    Q
        ``[B, H, N, D]``.
    u_target
        ``[B, H, D]``, unit-norm.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank > head_dim:
        raise ValueError(f"rank {rank} > head_dim {head_dim}")

    g = generator
    B, H, N, D = batch_size, num_heads, seq_len, head_dim
    sample = lambda *shape: torch.randn(*shape, generator=g, device=device)

    # rank principal directions per head, orthogonalised to be well-defined
    raw = sample(B, H, rank, D)                                 # [B, H, k, D]
    # QR per (b, h) for orthogonality of the rank components
    q_orth, _ = torch.linalg.qr(raw.transpose(-1, -2))          # [B, H, D, k]
    U = q_orth.transpose(-1, -2)                                # [B, H, k, D]
    scalars = sample(B, H, N, rank)                             # [B, H, N, k]
    signal = torch.einsum("bhnk,bhkd->bhnd", scalars, U)
    noise = sample(B, H, N, D)
    Q = snr * signal + noise

    u_target = pca_axis_target(Q)
    return Q, u_target
