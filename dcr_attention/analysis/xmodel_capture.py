r"""
Cross-architecture Q/K capture for spectral measurement on trained
transformers.

Provides a uniform interface to extract post-projection (and post-RoPE
where applicable) ``Q`` and ``K`` tensors from each attention layer of:

  * GPT-2 (``GPT2Attention``)        — no RoPE
  * BERT  (``BertSelfAttention``)    — no RoPE, bidirectional
  * GPT-NeoX / Pythia (``GPTNeoXAttention``) — with RoPE
  * Llama (``LlamaAttention``)       — with RoPE, GQA

Mechanism
---------
A pre-forward hook is registered on each attention submodule.  When the
hook fires, the inputs (``hidden_states`` and, for RoPE models,
``position_embeddings``) are inspected and **the projection + RoPE
computation is replayed externally** to recover ``Q`` and ``K``.  The
model's own forward then proceeds normally — the hook is read-only with
respect to the network's state.

Cost: one extra projection (and RoPE) per attention layer per forward
call.  This roughly doubles the projection cost but is otherwise
negligible relative to the softmax and value-mix that dominate.  Hooks
are removed on :func:`detach_capture`; with no hooks attached, the
production forward path is byte-for-byte unaffected (zero residual
overhead).

Captured shape: ``Q, K`` always come out as ``[B, H, T, D]`` with the
model's natural head dimension ``D``.  For Llama (GQA) the K is the
**pre-repeat** key tensor of shape ``[B, H_kv, T, D]`` — head index 0..H-1
addresses query heads; we *do not* expand it here, the analysis loops
over the H_kv distinct key streams.

INS-28 numerical contract: both ``Q`` and ``K`` are promoted to fp32
before being handed to downstream analysis (eigensolves on bf16 are
unreliable).  The promotion is on the **captured** copy only; the
running model continues in its native dtype.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch


# ---------------------------------------------------------------------------
# Capture state — class-level (only one capture session is active at a time
# in this benchmark).  Mirrors the pattern from DCRLlamaAttention (INS-30).
# ---------------------------------------------------------------------------

@dataclass
class _CaptureState:
    enabled: bool = False
    entries: List[dict] = field(default_factory=list)
    handles: List[Any] = field(default_factory=list)
    arch: Optional[str] = None
    # Layer-index mapping populated by attach_capture.
    module_to_layer: dict = field(default_factory=dict)


_state = _CaptureState()


def enable() -> None:
    """Turn capture on globally."""
    _state.enabled = True


def disable() -> None:
    """Turn capture off globally (does NOT remove hooks)."""
    _state.enabled = False


def reset() -> None:
    """Clear accumulated captures (does NOT remove hooks)."""
    _state.entries = []


def get_captures() -> List[dict]:
    """Snapshot copy of the captures."""
    return list(_state.entries)


# ---------------------------------------------------------------------------
# Architecture-specific replays
# ---------------------------------------------------------------------------

def _replay_gpt2(module, hidden_states):
    """
    GPT-2 attention: ``c_attn`` produces concatenated [q | k | v]; split
    along the last dim by ``split_size``; per-head reshape via
    ``_split_heads``.  No RoPE.
    """
    qkv = module.c_attn(hidden_states)
    query, key, _ = qkv.split(module.split_size, dim=2)
    Q = module._split_heads(query, module.num_heads, module.head_dim)
    K = module._split_heads(key,   module.num_heads, module.head_dim)
    return Q, K   # [B, H, T, D], [B, H, T, D]


def _replay_bert(module, hidden_states):
    """BERT self-attention: separate Q/K/V linears; transpose_for_scores."""
    Q = module.transpose_for_scores(module.query(hidden_states))
    K = module.transpose_for_scores(module.key(hidden_states))
    return Q, K


def _replay_gpt_neox(module, hidden_states, position_embeddings):
    """
    GPT-NeoX (Pythia): a single ``query_key_value`` linear produces
    interleaved Q|K|V per-head; reshape, then apply **partial** RoPE
    to the first ``rotary_ndims`` channels of each head only (Pythia
    rotates 25% of head_dim by default; ``rotary_ndims`` lives on the
    module).

    transformers 4.47 layout:
        qkv  :  [B, T, num_heads, 3 * head_size]
        Q,K,V split along the LAST axis in thirds
        Then for RoPE: split each head into (rotary [:rotary_ndims],
        pass [rotary_ndims:]) and rotate only the first part.
    """
    qkv = module.query_key_value(hidden_states)            # [B, T, 3*H*D]
    B, T, _ = qkv.shape
    H = module.num_attention_heads
    D = module.head_size
    qkv = qkv.view(B, T, H, 3 * D)
    Q = qkv[..., :D].permute(0, 2, 1, 3)                   # [B, H, T, D]
    K = qkv[..., D:2 * D].permute(0, 2, 1, 3)

    if position_embeddings is not None:
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
        cos, sin = position_embeddings
        rotary_ndims = getattr(module, "rotary_ndims", D)
        if rotary_ndims < D:
            Q_rot, Q_pass = Q[..., :rotary_ndims], Q[..., rotary_ndims:]
            K_rot, K_pass = K[..., :rotary_ndims], K[..., rotary_ndims:]
            Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos, sin)
            Q = torch.cat((Q_rot, Q_pass), dim=-1)
            K = torch.cat((K_rot, K_pass), dim=-1)
        else:
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    return Q, K


def _replay_llama(module, hidden_states, position_embeddings):
    """
    Llama: separate q_proj / k_proj / v_proj.  GQA — k_proj has
    ``num_key_value_heads`` heads, not ``num_attention_heads``.  RoPE via
    ``position_embeddings``.

    Returns ``(Q, K_pre_gqa)``: ``Q`` has shape ``[B, H, T, D]`` (full
    query heads); ``K`` has shape ``[B, H_kv, T, D]`` (pre-repeat).  The
    caller (analysis script) expands K per query head index via
    ``h_kv = h // n_rep`` rather than materialising the repeat.
    """
    cfg = module.config
    H_q = getattr(cfg, "num_attention_heads")
    H_kv = getattr(cfg, "num_key_value_heads", H_q)
    D = getattr(cfg, "head_dim", None)
    if D is None:
        D = cfg.hidden_size // H_q

    B, T, _ = hidden_states.shape
    Q = module.q_proj(hidden_states).view(B, T, H_q,  D).transpose(1, 2)
    K = module.k_proj(hidden_states).view(B, T, H_kv, D).transpose(1, 2)

    if position_embeddings is not None:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
    return Q, K


# ---------------------------------------------------------------------------
# Attach / detach helpers (model_class -> replay function)
# ---------------------------------------------------------------------------

def _make_pre_hook(module, replay_fn, expects_position_embeddings: bool,
                   gqa: bool):
    """Return a hook closure for one attention module."""
    layer_idx = id(module)  # patched to real layer_idx in attach_capture

    def hook(mod, args, kwargs):
        if not _state.enabled:
            return
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        with torch.no_grad():
            if expects_position_embeddings:
                pe = kwargs.get("position_embeddings")
                Q, K = replay_fn(mod, hidden_states, pe)
            else:
                Q, K = replay_fn(mod, hidden_states)

            Q_cpu = Q.detach().to(torch.float32)
            K_cpu = K.detach().to(torch.float32)
        _state.entries.append({
            "layer_idx": _state.module_to_layer.get(id(mod), -1),
            "Q": Q_cpu,     # [B, H_q, T, D]
            "K": K_cpu,     # [B, H_kv, T, D]  (H_kv = H_q except Llama GQA)
            "gqa": bool(gqa and Q_cpu.shape[1] != K_cpu.shape[1]),
        })
    return hook


_ARCH_TABLE = {
    "gpt2": {
        "module_path": "transformers.models.gpt2.modeling_gpt2.GPT2Attention",
        "replay": _replay_gpt2,
        "expects_position_embeddings": False,
        "gqa": False,
        # Try multiple block paths: AutoModelForCausalLM gives ("transformer","h"),
        # AutoModel gives ("h",) directly.
        "model_blocks_alternatives": [("transformer", "h"), ("h",)],
        "attn_attr": "attn",
    },
    "bert": {
        "module_path": "transformers.models.bert.modeling_bert.BertSelfAttention",
        "replay": _replay_bert,
        "expects_position_embeddings": False,
        "gqa": False,
        "model_blocks_alternatives": [
            ("bert", "encoder", "layer"),
            ("encoder", "layer"),
        ],
        "attn_attr": ("attention", "self"),
    },
    "gpt_neox": {
        "module_path": "transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention",
        "replay": _replay_gpt_neox,
        "expects_position_embeddings": True,
        "gqa": False,
        "model_blocks_alternatives": [("gpt_neox", "layers"), ("layers",)],
        "attn_attr": "attention",
    },
    "llama": {
        "module_path": "transformers.models.llama.modeling_llama.LlamaAttention",
        "replay": _replay_llama,
        "expects_position_embeddings": True,
        "gqa": True,
        "model_blocks_alternatives": [("model", "layers"), ("layers",)],
        "attn_attr": "self_attn",
    },
}


def _enumerate_attention_modules(model, arch: str):
    """Yield (layer_idx, attention_module) for each layer of the model.

    Probes the architecture's known block-path alternatives in order;
    the first one that exists wins.  Raises an informative error if
    none match (likely a transformers version drift; INS-26).
    """
    spec = _ARCH_TABLE[arch]
    blocks = None
    last_err = None
    for path in spec["model_blocks_alternatives"]:
        try:
            obj = model
            for attr in path:
                obj = getattr(obj, attr)
            blocks = obj
            break
        except AttributeError as e:
            last_err = e
            continue
    if blocks is None:
        raise AttributeError(
            f"could not locate attention blocks on {type(model).__name__} "
            f"for arch={arch!r} via paths "
            f"{spec['model_blocks_alternatives']}: last error: {last_err}"
        )

    for li, block in enumerate(blocks):
        attn_attr = spec["attn_attr"]
        attn = block
        if isinstance(attn_attr, str):
            attn = getattr(attn, attn_attr)
        else:
            for a in attn_attr:
                attn = getattr(attn, a)
        yield li, attn


def attach_capture(model, arch: str) -> int:
    """
    Install pre-hooks on every attention layer of ``model``.  Returns the
    number of layers hooked.  Captures are inactive until :func:`enable`.

    Idempotency: calling twice will detach existing hooks first.
    """
    if _state.handles:
        detach_capture()

    if arch not in _ARCH_TABLE:
        raise ValueError(f"unknown arch {arch!r}; supported: {sorted(_ARCH_TABLE)}")

    spec = _ARCH_TABLE[arch]
    n = 0
    for li, attn in _enumerate_attention_modules(model, arch):
        _state.module_to_layer[id(attn)] = li
        hook = _make_pre_hook(
            attn, spec["replay"],
            expects_position_embeddings=spec["expects_position_embeddings"],
            gqa=spec["gqa"],
        )
        h = attn.register_forward_pre_hook(hook, with_kwargs=True)
        _state.handles.append(h)
        n += 1

    _state.arch = arch
    return n


def detach_capture() -> None:
    """Remove all installed hooks; restore the production forward path."""
    for h in _state.handles:
        h.remove()
    _state.handles = []
    _state.module_to_layer = {}
    _state.arch = None


# ---------------------------------------------------------------------------
# Convenience: compute per-(layer, batch, head, query position) metrics
# from a captures list, AT a single chosen query position.
# ---------------------------------------------------------------------------

def compute_metrics_at_position(
    captures: List[dict],
    query_position: int = -1,
    beta: Optional[float] = None,
) -> List[dict]:
    r"""
    For each capture entry, compute per-head spectral metrics at the
    SINGLE query position ``query_position`` (default -1 = the last token
    in the sequence).

    Returns a flat list of dicts:
        {layer_idx, head, lambdas_neg, chi, S_lambda, R, delta, theta,
         dim_E_minus, relgap}

    ``lambdas_neg`` is the full list of negative eigenvalues of ``H``
    for that (layer, head), sorted most-negative-first.  Caller may
    truncate before serialising if memory is tight, but at head_dim=64
    storing all ~64 floats per (layer, head, position) is cheap.

    ``beta = 1/sqrt(D)`` by default, with ``D`` the head dimension
    inferred from the captured ``K`` tensor's last axis.  This matches
    the scaled-dot-product convention every model in this study uses.
    """
    import math
    from dcr_attention.analysis.gap_metrics import (
        attention_hessian_for_query,
        spectral_gap, dim_E_minus,
        chi_dominant_mass, spectral_entropy, dominance_ratio,
        relative_gap, negative_eigenvalues,
    )

    out = []
    for entry in captures:
        Q = entry["Q"]                       # [B, H_q, T, D]
        K = entry["K"]                       # [B, H_kv, T, D]
        B, H_q, T, D = Q.shape
        H_kv = K.shape[1]
        n_rep = H_q // H_kv
        if beta is None:
            _beta = 1.0 / math.sqrt(D)
        else:
            _beta = beta

        for b in range(B):
            for h in range(H_q):
                h_kv = h // n_rep
                q  = Q[b, h,    query_position, :]      # [D]
                Kh = K[b, h_kv, :, :]                   # [T, D]
                H  = attention_hessian_for_query(q, Kh, beta=_beta)
                delta, theta = spectral_gap(H)
                ev_neg = negative_eigenvalues(H)
                out.append({
                    "layer_idx": entry["layer_idx"],
                    "batch_idx": b,
                    "head": h,
                    "lambdas_neg": ev_neg.tolist(),
                    "chi":       chi_dominant_mass(H),
                    "S_lambda":  spectral_entropy(H),
                    "R":         dominance_ratio(H),
                    "delta":     float(delta),
                    "theta":     float(theta),
                    "dim_E_minus": dim_E_minus(H, theta),
                    "relgap":    relative_gap(H),
                })
    return out


__all__ = [
    "enable", "disable", "reset", "get_captures",
    "attach_capture", "detach_capture",
    "compute_metrics_at_position",
]
