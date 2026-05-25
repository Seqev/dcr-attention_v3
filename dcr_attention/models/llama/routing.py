"""
Routing decision for ``DCRLlamaAttention``.

A single function ``_route`` decides whether a given forward call goes
through SDPA or the DCR rank-local kernel.  Pure logic, no tensors —
everything is a plain ``int`` / ``bool``.

This module deliberately avoids the full signal-based ``dispatcher.dispatch()``
function.  In decode (``N_q == 1``) the signals like ``s_local`` and
``s_spectral`` are degenerate; v9 measurements show DCR/SDPA ratio depends
almost entirely on ``N_kv``.  A hard threshold captures this with zero
per-call cost.  Signal-based routing is a Phase 2c expansion.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional

from dcr_attention.models.llama.config import DCRLlamaConfig


class Branch(Enum):
    """Which attention path a single forward call takes."""
    SDPA = "sdpa"
    DCR = "dcr"


def route(
    N_q: int,
    N_kv: int,
    layer_idx: int,
    cfg: DCRLlamaConfig,
) -> Branch:
    r"""
    Decide whether to use SDPA or DCR for this forward call.

    Decision tree:

      1. ``cfg.enable_dcr is False``               → SDPA  (master switch off)
      2. ``cfg.layer_filter`` set and ``layer_idx`` not in it → SDPA
      3. ``N_q > 1``                               → SDPA  (prefill, causal)
      4. ``N_kv < cfg.T_dispatch``                 → SDPA  (short context)
      5. otherwise                                 → DCR

    Parameters
    ----------
    N_q
        Number of queries in this forward call (= seq dim of post-RoPE Q).
    N_kv
        Number of keys/values in the cached KV state (= seq dim of K, V
        after ``past_key_value.update``).
    layer_idx
        Zero-indexed transformer layer.  Used only with ``cfg.layer_filter``.
    cfg
        :class:`DCRLlamaConfig`.

    Returns
    -------
    Branch
    """
    if not cfg.enable_dcr:
        return Branch.SDPA
    if cfg.layer_filter is not None and layer_idx not in cfg.layer_filter:
        return Branch.SDPA
    if N_q > 1:
        return Branch.SDPA          # prefill — causal correctness via SDPA mask
    if N_kv < cfg.T_dispatch:
        return Branch.SDPA          # decode but cache too short for DCR to win
    return Branch.DCR


def explain_route(
    N_q: int,
    N_kv: int,
    layer_idx: int,
    cfg: DCRLlamaConfig,
) -> str:
    r"""
    Human-readable reason for the routing decision.  Used by debug logs and
    by tests that want to assert *why* a particular branch was taken, not
    only *which* branch.
    """
    if not cfg.enable_dcr:
        return "SDPA: enable_dcr=False"
    if cfg.layer_filter is not None and layer_idx not in cfg.layer_filter:
        return f"SDPA: layer_idx={layer_idx} not in layer_filter"
    if N_q > 1:
        return f"SDPA: prefill (N_q={N_q})"
    if N_kv < cfg.T_dispatch:
        return f"SDPA: short context (N_kv={N_kv} < T_dispatch={cfg.T_dispatch})"
    return f"DCR: decode + long context (N_kv={N_kv} ≥ T_dispatch={cfg.T_dispatch})"
