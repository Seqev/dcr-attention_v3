r"""
Stage B — ordering axis selection.

Given a query matrix :math:`Q \in \mathbb{R}^{N\times D}` and optional positional
embedding :math:`P \in \mathbb{R}^{N\times D}`, produce a unit vector
:math:`\hat u \in S^{D-1}` along which keys will be ranked.

Two strategies:

* **Positional** (preferred when PE available).  Find the axis along which the
  projection of :math:`P_{\text{centred}}` best tracks the integer position
  :math:`j - (N-1)/2`.  Solves

  .. math:: u^\star = \arg\min_u \; \|P_c\,u - j_c\|_2^2,

  then :math:`\hat u = u^\star / \|u^\star\|`.

  **Llama compatibility caveat.**  Llama-family models use RoPE, which bakes
  positional information into Q/K *projection matrices* rather than storing an
  additive embedding tensor.  Passing in a RoPE model's learned table will NOT
  give a meaningful axis.  For RoPE architectures use the PCA fallback
  (see INSIGHTS.md INS-6).

* **PCA** (fallback).  The principal direction of the centred query covariance,
  :math:`\hat u = V_{:,1}` from the SVD :math:`\bar Q = U\Sigma V^\top`.

Both paths return a unit-norm vector.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch

from dcr_attention.dispatcher.decision import AxisSource


_EPS = 1e-12


def _axis_from_positional(
    positional_embedding: torch.Tensor,
) -> torch.Tensor:
    r"""
    Solve :math:`\min_u \|P_c u - j_c\|_2^2` by least squares, where
    :math:`P_c = P - \bar P` and :math:`j_c = j - (N-1)/2`.
    """
    N, D = positional_embedding.shape
    pe_centred = positional_embedding - positional_embedding.mean(dim=0)
    j_target = (
        torch.arange(N, dtype=positional_embedding.dtype, device=positional_embedding.device)
        - (N - 1) / 2.0
    )
    # torch.linalg.lstsq returns a namedtuple with .solution of shape [D, 1]
    result = torch.linalg.lstsq(pe_centred, j_target.unsqueeze(1))
    u = result.solution.squeeze(-1)                             # [D]
    return u / (u.norm() + _EPS)


def _axis_from_pca(Q: torch.Tensor) -> torch.Tensor:
    r"""
    First right-singular vector of centred :math:`Q`.  Equivalently, the top
    eigenvector of :math:`\Sigma = \bar Q^\top \bar Q / N`.
    """
    Qc = Q - Q.mean(dim=0, keepdim=True)
    # full_matrices=False → economy SVD, much cheaper for N >> D
    _, _, Vt = torch.linalg.svd(Qc, full_matrices=False)
    u = Vt[0]
    return u / (u.norm() + _EPS)


def compute_axis(
    Q: torch.Tensor,
    positional_embedding: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, AxisSource]:
    r"""
    Resolve an ordering axis via positional-LS if a PE is supplied, otherwise
    via PCA of Q.

    Parameters
    ----------
    Q
        ``[N, D]``.  Caller reshapes multi-head tensors before passing in.
    positional_embedding
        Optional ``[N, D]`` additive embedding (non-RoPE models only).

    Returns
    -------
    axis, source : torch.Tensor (shape [D], unit-norm), AxisSource
    """
    if Q.dim() != 2:
        raise ValueError(f"expected Q shape [N, D], got {tuple(Q.shape)}")
    if positional_embedding is not None:
        if positional_embedding.shape != Q.shape:
            raise ValueError(
                f"positional_embedding shape {tuple(positional_embedding.shape)} "
                f"must match Q shape {tuple(Q.shape)}"
            )
        return _axis_from_positional(positional_embedding), "positional"
    return _axis_from_pca(Q), "pca"
