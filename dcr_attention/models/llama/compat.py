"""
Compatibility shim between DCR wrapper and HuggingFace ``transformers``.

This module isolates *all* knowledge of HF API surface.  When transformers
moves an internal symbol (it does), only this file changes; ``attention.py``
remains stable.

Currently shimmed:
  * ``apply_rotary_pos_emb``  — RoPE rotation, signature varies by version.
  * ``repeat_kv``             — GQA expansion (Llama-3 8B: 8 KV heads → 32 Q heads).

**Version compatibility.**  Tested against transformers ≥ 4.40 (where Llama-3
was introduced) and against transformers 5.x (where ``apply_rotary_pos_emb``
dropped the ``position_ids`` parameter).  Each HF call probes the actual
function signature at runtime and dispatches accordingly — the alternative
(version pinning) would force users into a single transformers version.

The ``inspect.signature`` calls happen on first invocation per process and
the result is cached.  Per-call cost: zero after warm-up.
"""

from __future__ import annotations
import inspect
from functools import lru_cache
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Internal — resolve & cache HF symbols once per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _resolve_hf_apply_rotary():
    """Returns (callable, accepts_position_ids: bool) or (None, None)."""
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb as _hf_apply,
        )
    except ImportError:
        return None, None
    params = inspect.signature(_hf_apply).parameters
    accepts_position_ids = "position_ids" in params
    return _hf_apply, accepts_position_ids


@lru_cache(maxsize=None)
def _resolve_hf_repeat_kv():
    try:
        from transformers.models.llama.modeling_llama import (
            repeat_kv as _hf_repeat,
        )
    except ImportError:
        return None
    return _hf_repeat


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Apply RoPE rotation to Q and K.

    Defers to the version installed in ``transformers``; if not available
    (e.g. running CPU unit tests without transformers installed), uses an
    inline reference implementation matching the modern HF API.

    Parameters
    ----------
    Q, K
        Shape ``[B, H, N, D]``.
    cos, sin
        Shape ``[B, N, D]`` (modern API) or ``[max_pos, D]`` (legacy with
        ``position_ids``).  We unsqueeze on ``unsqueeze_dim`` to broadcast
        over heads.
    position_ids
        Used only by transformers ≥ 4.40 and < 5.0; transformers 5.x
        removed this parameter (positions are baked into cos/sin upstream).
        We probe the installed signature and pass / omit accordingly.
    unsqueeze_dim
        Where to insert the head dim for broadcasting.  Llama uses 1.

    Returns
    -------
    Q_rot, K_rot : same shape as input.
    """
    _hf_apply, accepts_position_ids = _resolve_hf_apply_rotary()
    if _hf_apply is not None:
        if accepts_position_ids:
            return _hf_apply(Q, K, cos, sin, position_ids, unsqueeze_dim)
        else:
            # transformers ≥ 5.0 — drops position_ids from the signature.
            return _hf_apply(Q, K, cos, sin, unsqueeze_dim)
    # No transformers installed — inline fallback for CPU unit tests.
    return _apply_rotary_pos_emb_reference(
        Q, K, cos, sin, position_ids, unsqueeze_dim
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    r"""
    Rotate the second half of the last dim by 90°.  This is the
    ``[x_2, -x_1, x_4, -x_3, ...]`` block-pair rotation used by RoPE.
    Equivalent to: split into two halves, swap, negate the new first half.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    unsqueeze_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Reference RoPE matching transformers ≥ 4.40 ``apply_rotary_pos_emb``.

    .. math::
        \mathrm{rotated}(x, \theta) = x \cdot \cos\theta + \mathrm{rotate\_half}(x) \cdot \sin\theta

    For modern transformers ``cos, sin`` are already shaped ``[B, N, D]`` —
    just unsqueeze on the head dim and broadcast.  ``position_ids`` is
    ignored in this case (positions already baked into cos/sin).

    For the legacy API, ``cos, sin`` are ``[max_pos, D]`` and ``position_ids``
    is ``[B, N]``: gather the relevant rows first.
    """
    if position_ids is not None and cos.dim() == 2:
        # Legacy API: gather rows by position_ids.
        cos = cos[position_ids]                 # [B, N, D]
        sin = sin[position_ids]                 # [B, N, D]

    # cos, sin are now [B, N, D]; unsqueeze on the head axis (dim=1 for Llama).
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    Q_rot = (Q * cos) + (_rotate_half(Q) * sin)
    K_rot = (K * cos) + (_rotate_half(K) * sin)
    return Q_rot, K_rot


# ---------------------------------------------------------------------------
# GQA (Grouped-Query Attention) expansion
# ---------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""
    Expand a ``[B, H_kv, N, D]`` tensor to ``[B, H_kv * n_rep, N, D]`` by
    repeating each head ``n_rep`` times along the head dim.

    Llama-3 8B has 32 query heads and 8 key/value heads (n_rep=4).  After
    RoPE+cache update we need the K/V tensors expanded to 32 heads to match
    the Q tensor.

    Parameters
    ----------
    x
        ``[B, H_kv, N, D]``.
    n_rep
        Repetition count.  ``n_rep=1`` is a no-op.

    Returns
    -------
    ``[B, H_kv * n_rep, N, D]``.
    """
    if n_rep == 1:
        return x
    _hf_repeat = _resolve_hf_repeat_kv()
    if _hf_repeat is not None:
        return _hf_repeat(x, n_rep)
    return _repeat_kv_reference(x, n_rep)


def _repeat_kv_reference(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""
    Reference GQA expansion matching transformers ``repeat_kv``.

    Implementation note: torch's ``repeat_interleave`` along the head dim is
    cleaner than the HF original (``expand`` + ``reshape``), but the HF
    version is preferred for fidelity when the real package is installed.
    """
    B, H_kv, N, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H_kv, n_rep, N, D)
        .reshape(B, H_kv * n_rep, N, D)
    )


# ---------------------------------------------------------------------------
# Cache update — defer to whatever the user passed in
# ---------------------------------------------------------------------------

def cache_update(
    past_key_value,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Append new K/V to the cache, return the full cached K/V.

    HF's ``Cache.update`` API has stable signature since 4.40.  We defer
    fully — no version branching needed.  This wrapper exists only so the
    attention forward doesn't import HF symbols directly.

    Parameters
    ----------
    past_key_value
        HF ``Cache`` object (``DynamicCache``, ``StaticCache``, etc.)
        or ``None`` for prefill before any cache exists.
    K_new, V_new
        ``[B, H_kv, N_q, D]`` newly-projected K/V.
    layer_idx
        Layer index for cache slot.
    cache_kwargs
        Forwarded to ``cache.update`` as ``**kwargs``.  Per v12 diagnostic
        Section I, transformers 5.x ``DynamicCache.update`` signature is
        ``(self, key_states, value_states, layer_idx, *args, **kwargs)`` —
        we must spread the dict, not pass it as a positional argument
        (which would land in ``*args[0]`` as a single dict, not be unpacked).

    Returns
    -------
    K_cached, V_cached : full cached tensors after the update.  In prefill
    these equal ``K_new, V_new``.
    """
    if past_key_value is None:
        return K_new, V_new
    # transformers 4.47.x: DynamicCache.update(key, value, layer_idx, cache_kwargs=None)
    # transformers 5.x+:   DynamicCache.update(key, value, layer_idx, *args, **kwargs)
    params = inspect.signature(past_key_value.update).parameters
    if "cache_kwargs" in params:
        return past_key_value.update(
            K_new, V_new, layer_idx, cache_kwargs=(cache_kwargs or None)
        )
    return past_key_value.update(
        K_new, V_new, layer_idx, **(cache_kwargs or {})
    )
