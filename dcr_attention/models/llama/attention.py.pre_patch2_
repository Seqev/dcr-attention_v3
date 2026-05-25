"""
``DCRLlamaAttention`` — drop-in HF ``LlamaAttention`` replacement that routes
between dense SDPA and the DCR rank-local kernel based on sequence shape.

Architecture: composition over ``LlamaAttention`` (not subclass).  We hold a
reference to a base layer and steal its q/k/v/o projection modules.  No
weight copying: parameters are shared via ``nn.Module`` reference, so
monkey-patching a loaded 8B model adds zero memory overhead.

Forward routing (see :func:`routing.route` for full decision tree):

  * ``N_q > 1``  (prefill)               → SDPA (causal correctness).
  * ``N_q == 1, N_kv < T_dispatch``      → SDPA (short context).
  * ``N_q == 1, N_kv ≥ T_dispatch``      → DCR rank-local kernel.

The two branches are bit-identical for SDPA cases (modulo the dispatcher
overhead, which is one int comparison).  Correctness for the DCR branch is
established by the kernel's own tests + Phase 2-pre decode correctness.
"""

from __future__ import annotations
from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcr_attention.dispatcher.stage_b import compute_axis
from dcr_attention.kernel.rank_local_attention import rank_local_attention
from dcr_attention.models.llama.compat import (
    apply_rotary_pos_emb,
    cache_update,
    repeat_kv,
)
from dcr_attention.models.llama.config import DCRLlamaConfig
from dcr_attention.models.llama.routing import Branch, route


class DCRLlamaAttention(nn.Module):
    r"""
    Wrapper around ``LlamaAttention`` that routes between SDPA and DCR.

    Construction is by **composition**: pass an existing ``LlamaAttention``
    instance and we re-use its projections and configuration.  This is what
    ``monkey_patch.patch_llama_with_dcr`` does for every layer of a loaded
    model.

    Parameters
    ----------
    base_layer
        A ``transformers.models.llama.modeling_llama.LlamaAttention``
        instance.  Its ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``,
        ``head_dim``, ``num_heads``, ``num_key_value_heads``, ``layer_idx``
        attributes are read.  Original module is retained via reference.
    cfg
        :class:`DCRLlamaConfig`.  Default: :data:`DEFAULT_DCR_LLAMA_CONFIG`.

    Notes on attributes copied from the base layer
    ----------------------------------------------
    We do NOT re-create ``q_proj``, ``k_proj``, etc. — we hold them by
    reference so that the loaded weights are shared without copying GBs
    of parameters.  This means modifying a projection in-place affects
    both the wrapper and the original; this is intentional and required
    for monkey-patching to be memory-neutral.

    Phase 2b instrumentation
    ------------------------
    Class-level counters track how many times each branch was invoked.
    Used by Phase 2b smoke tests to verify that DCR actually triggers
    when routing predicts it (vs. silently falling back to SDPA).

    Reset between test runs via :meth:`reset_counters`.  Single-process
    inference is single-threaded per CUDA stream, so plain int counters
    are safe — no atomicity needed.
    """

    # Phase 2b: class-level routing counters.
    _dcr_invocations: int = 0
    _sdpa_invocations: int = 0

    @classmethod
    def reset_counters(cls) -> None:
        """Reset DCR/SDPA invocation counters.  Call at start of test."""
        cls._dcr_invocations = 0
        cls._sdpa_invocations = 0

    @classmethod
    def get_counters(cls) -> dict:
        """Return current counter snapshot as ``{"dcr": n, "sdpa": m}``."""
        return {
            "dcr": cls._dcr_invocations,
            "sdpa": cls._sdpa_invocations,
        }

    def __init__(
        self,
        base_layer: nn.Module,
        cfg: Optional[DCRLlamaConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else DCRLlamaConfig()
        self._base_layer = base_layer

        # --- Read projections by reference (no copy) ---
        self.q_proj = base_layer.q_proj
        self.k_proj = base_layer.k_proj
        self.v_proj = base_layer.v_proj
        self.o_proj = base_layer.o_proj

        # --- Read shape config (cheap ints / from config) ---
        # HF API drift: transformers 4.40 exposed num_heads / num_key_value_heads
        # as instance attributes; transformers 5.x moved them to the model's
        # config and only kept head_dim on the layer.  We probe both, in order
        # of preference: instance attribute → base_layer.config → top-level
        # attribute aliases (num_attention_heads).  If everything fails we
        # raise a diagnostic error naming exactly what we checked.
        self.num_heads = self._resolve_shape_attr(
            base_layer,
            ["num_heads", "num_attention_heads"],
            config_keys=["num_attention_heads", "num_heads"],
        )
        self.num_key_value_heads = self._resolve_shape_attr(
            base_layer,
            ["num_key_value_heads"],
            config_keys=["num_key_value_heads"],
            default=self.num_heads,         # MHA fallback if no GQA distinction
        )
        self.head_dim = self._resolve_shape_attr(
            base_layer,
            ["head_dim"],
            config_keys=["head_dim"],
        )
        self.layer_idx = getattr(base_layer, "layer_idx", 0)

        if self.num_heads is None or self.head_dim is None:
            cfg_obj = getattr(base_layer, "config", None)
            cfg_keys = list(vars(cfg_obj).keys()) if cfg_obj is not None else []
            raise ValueError(
                f"DCRLlamaAttention: cannot resolve required shape parameters "
                f"from base_layer.\n"
                f"  num_heads (got: {self.num_heads}) — checked instance "
                f"attrs ['num_heads', 'num_attention_heads'] and "
                f"config keys ['num_attention_heads', 'num_heads'].\n"
                f"  head_dim  (got: {self.head_dim}) — checked instance attr "
                f"'head_dim' and config key 'head_dim'.\n"
                f"  base_layer type: {type(base_layer).__name__}\n"
                f"  base_layer.config type: {type(cfg_obj).__name__ if cfg_obj else 'None'}\n"
                f"  base_layer.config keys: {cfg_keys[:20]}"
                f"{'...' if len(cfg_keys) > 20 else ''}"
            )

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        self.n_rep = self.num_heads // self.num_key_value_heads

    @staticmethod
    def _resolve_shape_attr(
        base_layer,
        instance_keys,
        config_keys,
        default=None,
    ):
        r"""
        Try to read a shape int from ``base_layer`` resilient to HF API drift.

        Resolution order:
          1. Direct instance attribute (any of ``instance_keys``).
          2. ``base_layer.config.<key>`` (any of ``config_keys``).
          3. ``default``.

        This handles:
          * transformers 4.40 (instance attrs present)
          * transformers 5.x  (only head_dim on instance, rest on config)
          * Custom subclasses storing things in non-standard places.
        """
        for key in instance_keys:
            v = getattr(base_layer, key, None)
            if v is not None:
                return v
        cfg = getattr(base_layer, "config", None)
        if cfg is not None:
            for key in config_keys:
                v = getattr(cfg, key, None)
                if v is not None:
                    return v
        return default

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(                                                   # noqa: C901
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["object"] = None,                # HF Cache; 5.x kwarg name (plural)
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Llama-style attention forward.

        Signature matches HF ``LlamaAttention.forward``.  Accepts kwargs from
        both transformers 4.40+ and 5.x; per v12 diagnostic dump (Section J),
        HF 5.x calls us with kwarg ``past_key_values`` (plural form).  Earlier
        HF versions used ``past_key_value`` (singular); for backward-compat we
        check ``kwargs`` for the singular form too.

        Parameters
        ----------
        hidden_states
            ``[B, N_q, hidden_size]``.
        attention_mask
            Optional additive mask, ``[B, 1, N_q, N_kv]`` for prefill.
        position_ids
            ``[B, N_q]``.  Forwarded to legacy rotary path only; modern API
            (5.x) provides ``position_embeddings`` directly.
        past_key_values
            HF ``Cache`` instance, or ``None`` for prefill-only forward.
            **Plural name** is mandatory for transformers 5.x compatibility.
        output_attentions
            If ``True``, return attention weights.  **Not supported in DCR
            branch** — we raise.  SDPA branch returns None unless explicitly
            requested and supported.
        use_cache
            Whether to update ``past_key_values``.  Forwarded as-is.
        cache_position
            ``[N_q]`` int tensor giving absolute positions of new tokens
            within the cache.  Used by ``Cache.update`` for some types.
        position_embeddings
            Pre-computed ``(cos, sin)`` from the model's rotary emb module.
            Modern API path.

        Returns
        -------
        attn_output : ``[B, N_q, hidden_size]``
        attn_weights : ``[B, H, N_q, N_kv]`` or ``None``
        """
        if output_attentions:
            raise NotImplementedError(
                "output_attentions=True is not supported by DCRLlamaAttention. "
                "Disable it or use the unwrapped LlamaAttention for layers "
                "where attention weights are needed."
            )

        # Backward-compat with HF 4.x callers that passed `past_key_value`
        # (singular).  HF 5.x uses `past_key_values` (plural).  If the
        # singular form arrives via kwargs and the plural slot is empty,
        # honour it.  Per v12 diagnostic, plural is the 5.x contract.
        if past_key_values is None:
            past_key_values = kwargs.pop("past_key_value", None)

        B, N_q, _ = hidden_states.shape

        # 1. Q/K/V projections + reshape to [B, H, N_q, D].
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_q, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_q, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE.
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        elif position_ids is not None:
            # Legacy path — model holds rotary emb module on the base layer
            # (transformers 4.x layout).  HF 5.x always provides
            # ``position_embeddings`` upstream so this branch is dead in 5.x;
            # retained for older HF compatibility.
            rotary_emb = getattr(self._base_layer, "rotary_emb", None)
            if rotary_emb is None:
                raise RuntimeError(
                    "No position_embeddings provided and base_layer has no "
                    "rotary_emb attribute; cannot apply RoPE."
                )
            cos, sin = rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        # else: no positional info — caller's responsibility (e.g., ROPE-less
        # debug models).  We do not raise; we skip RoPE.

        # 3. KV cache update.  K, V here are pre-GQA-repeat (num_key_value_heads).
        cache_kwargs = {}
        if cache_position is not None:
            cache_kwargs["cache_position"] = cache_position
        if position_embeddings is not None:
            cache_kwargs["sin"] = position_embeddings[1]
            cache_kwargs["cos"] = position_embeddings[0]
        K, V = cache_update(past_key_values, K, V, self.layer_idx, cache_kwargs)

        # 4. GQA expansion to match Q's head count.
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        # 5. Routing decision.
        N_kv = K.shape[-2]
        branch = route(N_q=N_q, N_kv=N_kv, layer_idx=self.layer_idx, cfg=self.cfg)

        # 6. Dispatch.  Counters incremented for Phase 2b verification.
        if branch is Branch.SDPA:
            type(self)._sdpa_invocations += 1
            attn_output = self._sdpa_forward(Q, K, V, attention_mask, N_q, N_kv)
        else:
            type(self)._dcr_invocations += 1
            attn_output = self._dcr_forward(Q, K, V)

        # 7. o_proj.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, N_q, -1)
        attn_output = self.o_proj(attn_output)

        # We intentionally return None for attn_weights; output_attentions=True
        # was rejected at the top.  HF 5.x LlamaDecoderLayer unpacks exactly
        # two values: `hidden_states, _ = self.self_attn(...)`. Cache is
        # mutated in-place by `cache_update` above; not returned separately.
        # See v13 GPU report — 4-test failure on tuple arity confirmed this contract.
        return attn_output, None

    # ------------------------------------------------------------------
    # Branches
    # ------------------------------------------------------------------

    def _sdpa_forward(
        self,
        Q: torch.Tensor,                  # [B, H, N_q, D]
        K: torch.Tensor,                  # [B, H, N_kv, D]
        V: torch.Tensor,                  # [B, H, N_kv, D]
        attention_mask: Optional[torch.Tensor],
        N_q: int,
        N_kv: int,
    ) -> torch.Tensor:
        r"""
        Dense SDPA path.  Used for prefill (causal correctness) and short-
        context decode (where DCR loses to SDPA — see Phase 2-pre v9 §3).

        v14b: replaces ``F.scaled_dot_product_attention`` (which uses bf16
        softmax via FLASH backend on Ampere/Ada) with manual attention that
        does softmax in fp32 and casts back, exactly matching HF eager
        ``LlamaAttention``.

        Without this, bf16 softmax accumulates over 16 layers to ~0.30 in
        logits — see v14 GPU integration report (Outcome B):

            test_enable_dcr_false_matches_unpatched     max|diff| = 0.302734
            test_T_dispatch_infinity_matches_unpatched  max|diff| = 0.140625

        HF eager reference (transformers 5.6.2 ``modeling_llama``):

            attn_weights = torch.matmul(Q, K.transpose(2, 3)) / sqrt(d_head)
            attn_weights = attn_weights + attention_mask        # already causal
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(Q.dtype)
            attn_output  = torch.matmul(attn_weights, V)

        Performance trade-off accepted: this path runs only in
        ``enable_dcr=False`` (ablation) and short-context decode where DCR
        loses to dense anyway.  Core DCR Triton kernel untouched.
        """
        head_dim = Q.shape[-1]

        # 1) Q @ K^T / sqrt(d_head) — both bf16, accumulator inside matmul is fp32
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        # 2) Apply attention mask (HF eager always provides 4D mask containing
        #    causal + padding; fall back to manual causal for direct unit tests
        #    that don't pass a mask).
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        elif N_q > 1:
            # Manual causal mask (only used in unit tests / mask-less prefill).
            causal_mask = torch.triu(
                torch.full(
                    (N_q, N_kv),
                    torch.finfo(attn_weights.dtype).min,
                    dtype=attn_weights.dtype,
                    device=attn_weights.device,
                ),
                diagonal=N_kv - N_q + 1,
            )
            attn_weights = attn_weights + causal_mask

        # 3) Softmax in fp32 for numerical stability — THE FIX.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

        # 4) attn_weights @ V
        return torch.matmul(attn_weights, V)

    def _dcr_forward(
        self,
        Q: torch.Tensor,                  # [B, H, N_q=1, D]   (decode only)
        K: torch.Tensor,                  # [B, H, N_kv, D]
        V: torch.Tensor,                  # [B, H, N_kv, D]
    ) -> torch.Tensor:
        r"""
        Rank-local DCR path.  Only invoked in decode (N_q == 1) with
        N_kv ≥ T_dispatch.

        Axis: per-(b, h), computed from K via PCA.  See INS-6 (RoPE
        compatibility) for why post-RoPE PCA is the chosen axis source.

        Cost note: PCA is O(N_kv · D²) per head, called once per forward.
        For Llama-3 8B (B=1, H=32, D=128, N_kv=16384) this is ~1·10⁹ flops
        per layer ≈ 0.25× DCR-attention cost.  Phase 2c will add caching
        across decode steps to amortise this.
        """
        B, H, _, D = Q.shape
        # Compute per-head axis from K (post-RoPE).  One GPU→CPU transfer per
        # layer instead of H separate transfers — avoids H×16 WSL2 CUDA syncs/step.
        # Vectorised batched SVD is the Phase 2c speedup; this is the interim fix.
        K_cpu = K.float().cpu()                              # [B, H, N_kv, D] — ONE sync
        axes_cpu = torch.zeros(B, H, D)
        for b in range(B):
            for h in range(H):
                axis_bh, _ = compute_axis(K_cpu[b, h], positional_embedding=None)
                axes_cpu[b, h] = axis_bh
        axes = axes_cpu.to(device=K.device, dtype=K.dtype)  # ONE transfer back

        return rank_local_attention(
            Q, K, V, axes, k_window=self.cfg.k_window,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"layer_idx={self.layer_idx}, num_heads={self.num_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, "
            f"k_window={self.cfg.k_window}, T_dispatch={self.cfg.T_dispatch}, "
            f"axis_source={self.cfg.axis_source}"
        )
