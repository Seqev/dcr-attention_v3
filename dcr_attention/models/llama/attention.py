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
import transformers as _transformers

# transformers 4.x LlamaDecoderLayer.forward unpacks 3 values from self_attn:
#   hidden_states, attn_weights, present_key_value
# transformers 5.x unpacks 2:
#   hidden_states, _
# Detect once at import time so forward() returns the right arity.
_HF_RETURNS_PRESENT_KV = int(_transformers.__version__.split(".")[0]) < 5

from dcr_attention.dispatcher.stage_b import compute_axis
from dcr_attention.kernel.rank_local_attention import rank_local_attention
from dcr_attention.models.llama.adaptive import (
    adaptive_k_window,
    detect_leaky_boundary,
    widen_factor,
)
from dcr_attention.models.llama.compat import (
    apply_rotary_pos_emb,
    cache_update,
    repeat_kv,
)
from dcr_attention.kernel.qaxis_topk_reference import topk_qaxis_attention_reference
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

    # Phase 2c-gap: opt-in Q/K capture for INS-31 gap-theory validation on
    # real attention (benchmarks/phase2c_gap/run_gap_validation.py).
    # OFF BY DEFAULT — when False, the forward path is byte-for-byte
    # unaffected modulo a single attribute read per layer per call.
    # When True, the hook computes reduced spectral statistics inline
    # (no raw Q/K tensors stored).  See _record_capture for the
    # numerical contract.
    _capture_qk: bool = False
    _qk_captures: list = []   # populated only when _capture_qk is True

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

    @classmethod
    def enable_capture(cls) -> None:
        """Turn ON per-layer Q/K spectral capture (off by default).

        After enabling, every forward call records reduced statistics
        (per-head spectral gap, ``dim E_-``, top eigenvalues; per-layer
        ``d_eff``) into :attr:`_qk_captures`.  Disable with
        :meth:`disable_capture` when measurement is done — the hook
        otherwise runs on every layer of every forward call.
        """
        cls._capture_qk = True

    @classmethod
    def disable_capture(cls) -> None:
        """Turn OFF Q/K capture.  Pending captures remain available via
        :meth:`get_captures` until cleared by :meth:`reset_captures`.
        """
        cls._capture_qk = False

    @classmethod
    def reset_captures(cls) -> None:
        """Discard all accumulated capture entries.  Does not affect the
        enable / disable flag.
        """
        cls._qk_captures = []

    @classmethod
    def get_captures(cls) -> list:
        """Return a *copy* of the current capture list.  Each entry is a
        dict with keys ``layer_idx``, ``batch_idx``, ``N_q``, ``N_kv``,
        ``d_eff_layer`` (float) and ``heads`` (list of per-head dicts).
        """
        return list(cls._qk_captures)

    # ------------------------------------------------------------------
    # Causal spectral ablation (off by default — see SPECTRAL_ABLATION).
    # Class-level pattern mirrors the capture hook above; with the flag
    # off the forward path is byte-for-byte unaffected (one attribute
    # read per layer per call).
    # ------------------------------------------------------------------

    _ablation_enabled: bool = False
    _ablation_spec: Optional[dict] = None     # {layer_idx, head, mode, seed}
    _ablation_log: list = []

    @classmethod
    def enable_ablation(cls, layer_idx: int, head: int,
                        mode: str = "treatment", seed: int = 0) -> None:
        """Turn ON the per-head spectral-mode ablation.

        Parameters
        ----------
        layer_idx, head
            Identify which attention head is ablated.  Only that one head
            in that one layer is touched; everything else runs unchanged.
        mode
            ``"treatment"`` — suppress the dominant eigenmode of
            ``Cov_p(k)`` at the last query position.
            ``"control"`` — suppress a RANDOM non-dominant (bulk)
            eigenmode of the same construction, with the same
            ``||K||_F``-preservation rescale.  This is the matched
            comparison the experiment is built around.
        seed
            For ``"control"``, the random bulk index ``j`` is drawn from
            ``RandomState(seed + layer_idx*1000 + head)`` once per
            (layer, head) — deterministic across forward calls so that
            repeated measurements use the same ``j``.
        """
        if mode not in ("treatment", "control"):
            raise ValueError(f"mode must be 'treatment' or 'control', got {mode!r}")
        cls._ablation_enabled = True
        cls._ablation_spec = {
            "layer_idx": int(layer_idx),
            "head": int(head),
            "mode": mode,
            "seed": int(seed),
        }

    @classmethod
    def disable_ablation(cls) -> None:
        """Turn OFF ablation.  Pending log entries remain available."""
        cls._ablation_enabled = False
        cls._ablation_spec = None

    @classmethod
    def reset_ablation_log(cls) -> None:
        """Clear accumulated per-call ablation records."""
        cls._ablation_log = []

    @classmethod
    def get_ablation_log(cls) -> list:
        """Return a copy of the ablation log.  Each entry records the
        eigenvalue magnitudes and the perturbation Frobenius norm — the
        diagnostic data that lets the analysis verify the ``||K||_F``
        constraint and the matched-magnitude requirement.
        """
        return list(cls._ablation_log)

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

        # 4b. OPT-IN diagnostic capture (off by default → single attribute
        # read, no behaviour change).  See enable_capture / _record_capture.
        if type(self)._capture_qk:
            self._record_capture(Q, K)

        # 4c. OPT-IN causal spectral ablation (off by default).  Modifies
        # K for the configured (layer, head) only; everything else is
        # untouched.  See enable_ablation / _apply_ablation.
        if type(self)._ablation_enabled:
            K = self._apply_ablation(Q, K)

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

        # Return arity is version-dependent (detected at import time):
        # - transformers 5.x: decoder unpacks 2 values; cache mutated in-place.
        # - transformers 4.x: decoder unpacks 3 values, needs present_key_value
        #   as the third element (the same DynamicCache mutated in-place above).
        if _HF_RETURNS_PRESENT_KV:
            return attn_output, None, past_key_values
        return attn_output, None

    # ------------------------------------------------------------------
    # Diagnostic capture (off by default — see enable_capture)
    # ------------------------------------------------------------------

    def _record_capture(self, Q: torch.Tensor, K: torch.Tensor) -> None:
        r"""
        Reduced-quantity spectral capture for INS-31 gap-theory validation.

        Stores, per (layer, batch index) at the LAST query position (the one
        a decode step would predict from), small structured statistics:

        - ``d_eff_layer``   : participation ratio of ``Cov(Q)`` across
          heads at this position — the dispatcher-side signal.  Computed
          via the production :func:`compute_d_eff`.
        - ``heads[h]``      : per-head dict ``{delta_H, theta,
          dim_E_minus, top_eigs_neg}``.  ``H = -beta * Cov_p(k)`` with
          :math:`p = \mathrm{softmax}(\beta\,K_h q_h)` and
          ``beta = 1/sqrt(D)`` — the same scale the model itself uses.

        Numerical contract: Q/K are detached and promoted to fp32 inside
        the helper (INS-28); eigensolves on bf16 are unreliable.  No
        gradients flow through this hook.

        Cost: when off, the entire method is bypassed (a single attribute
        read in ``forward`` decides).  When on, per layer per call: one
        :func:`compute_d_eff` plus ``num_heads`` independent eigensolves
        on (D x D) matrices.  Bounded — does not grow with N_kv beyond
        the cost of forming the K x q product.
        """
        import math
        from dcr_attention.dispatcher import compute_d_eff
        from dcr_attention.analysis.gap_metrics import (
            attention_hessian_for_query, spectral_gap, dim_E_minus,
        )

        B, H, N_q, D = Q.shape
        N_kv = K.shape[-2]
        beta = 1.0 / math.sqrt(D)

        Q32 = Q.detach().to(torch.float32)
        K32 = K.detach().to(torch.float32)

        for b in range(B):
            Q_layer = Q32[b, :, -1, :]                # [H, D]
            d_eff_layer, _ = compute_d_eff(Q_layer)
            head_stats = []
            for h in range(H):
                q = Q32[b, h, -1, :]                  # [D]
                K_h = K32[b, h, :, :]                 # [N_kv, D]
                Hh = attention_hessian_for_query(q, K_h, beta=beta)
                delta, theta = spectral_gap(Hh)
                dim_Em = dim_E_minus(Hh, theta)
                # Top-16 most-negative eigenvalues (sorted ascending so they
                # ARE the most negative) for offline histogramming.
                ev = torch.linalg.eigvalsh(Hh).cpu().numpy()
                ev.sort()
                head_stats.append({
                    "head": h,
                    "delta_H": float(delta),
                    "theta": float(theta),
                    "dim_E_minus": int(dim_Em),
                    "top_eigs_neg": [float(x) for x in ev[:16]],
                })

            type(self)._qk_captures.append({
                "layer_idx": int(self.layer_idx),
                "batch_idx": int(b),
                "N_q": int(N_q),
                "N_kv": int(N_kv),
                "d_eff_layer": float(d_eff_layer),
                "heads": head_stats,
            })

    def _apply_ablation(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        r"""
        Replace ``K`` for the configured (layer, head) with an ablated
        version, **matched-Frobenius-perturbation** style.

        Construction at the LAST query position
        =======================================

        1. ``p   = softmax(beta * K_h @ q_last)``                    fp32
        2. ``K_c = K_h - p @ K_h``                                   (p-centred)
        3. ``Cov_p = K_c^T diag(p) K_c``                             [D, D]
        4. eigh(Cov_p) → ascending eigvals, eigvecs columns
        5. ``v_1`` = top eigvec (dominant mode)
           ``v_j`` = a bulk eigvec, ``j`` drawn from
                    ``RandomState(seed + layer_idx*1000 + head)`` once
                    per (layer, head)
        6. ``M_T = ||K_h @ v_1||``   ``M_C = ||K_h @ v_j||``
        7. ``T = min(M_T, M_C)``                                     **matched target**
        8. For the active mode (treatment or control):
             ``alpha = T / M_active  ∈ (0, 1]``
             ``K_h' = K_h - alpha * (K_h @ v_active) ⊗ v_active``
        9. ``||K_h - K_h'||_F = alpha * M_active = T``

        Why this construction
        ---------------------
        The earlier "project + rescale to preserve ||K_h||_F" form, while
        norm-preserving, made the treatment perturbation 10–30 × LARGER
        than the control perturbation (different ``M`` for ``v_1`` vs
        ``v_j``).  A naive treatment-greater-than-control result there
        would have been pure scale-asymmetry, not a causal claim about
        the dominant mode (the task's "uncontrolled ablation is worse
        than no experiment" warning).

        The cost of matching: ``||K_h||_F`` is no longer exactly
        preserved.  The drop is at most ``alpha(2-alpha) * T^2`` —
        log it and verify it is small.  Mass preservation is automatic
        (softmax over modified K still sums to 1 by construction).

        Numerical contract (INS-28): eigsolve on fp32-promoted Cov_p;
        ``torch._C._LinAlgError`` falls back to ``numpy.linalg.eigh``
        on fp64 symmetric copy (INS-33 lesson).
        """
        import math
        import numpy as np

        spec = type(self)._ablation_spec
        if spec is None or self.layer_idx != spec["layer_idx"]:
            return K

        head = spec["head"]
        mode = spec["mode"]
        seed = spec["seed"]

        if head < 0 or head >= K.shape[1]:
            return K

        K_modified = K.clone()
        B, _, T, D = K.shape
        beta = 1.0 / math.sqrt(D)

        for b in range(B):
            q     = Q[b, head, -1, :].detach().to(torch.float32)        # [D]
            K_h32 = K[b, head, :, :].detach().to(torch.float32)         # [T, D]

            p     = torch.softmax(beta * (K_h32 @ q), dim=0)
            k_bar = p @ K_h32
            K_c   = K_h32 - k_bar
            cov_p = (K_c * p[:, None]).T @ K_c
            cov_p = 0.5 * (cov_p + cov_p.T)

            try:
                eigvals, eigvecs = torch.linalg.eigh(cov_p)
            except torch._C._LinAlgError:
                cov_np = cov_p.cpu().double().numpy()
                cov_np = 0.5 * (cov_np + cov_np.T)
                ew, ev = np.linalg.eigh(cov_np)
                eigvals = torch.from_numpy(ew).to(cov_p.device, torch.float32)
                eigvecs = torch.from_numpy(ev).to(cov_p.device, torch.float32)

            # v_1 (treatment) and v_j (control) — compute both for matched
            # target.  j is deterministic per (seed, layer_idx, head).
            v_top = eigvecs[:, -1]
            rng = np.random.RandomState(seed + int(self.layer_idx) * 1000 + head)
            j_used = int(rng.randint(0, D - 1))                  # bulk index
            v_bulk = eigvecs[:, j_used]

            proj_T = K_h32 @ v_top
            proj_C = K_h32 @ v_bulk
            M_T = float(torch.linalg.norm(proj_T))
            M_C = float(torch.linalg.norm(proj_C))
            T_target = min(M_T, M_C)                              # matched

            if mode == "treatment":
                v = v_top
                M_active = M_T
                proj_active = proj_T
            else:
                v = v_bulk
                M_active = M_C
                proj_active = proj_C

            alpha = T_target / max(M_active, 1e-12)              # in (0, 1]
            K_h_new = K_h32 - alpha * proj_active[:, None] * v[None, :]

            delta_F = float(torch.linalg.norm(K_h32 - K_h_new))
            norm_K = float(torch.linalg.norm(K_h32))
            norm_K_new = float(torch.linalg.norm(K_h_new))

            K_modified[b, head, :, :] = K_h_new.to(K.dtype)

            type(self)._ablation_log.append({
                "layer_idx": int(self.layer_idx),
                "batch_idx": int(b),
                "head":      int(head),
                "mode":      mode,
                "j_used":    j_used,
                "lambda_1":  float(eigvals[-1]),
                "lambda_2":  float(eigvals[-2]) if D >= 2 else float("nan"),
                "lambda_j":  float(eigvals[j_used]),
                "M_T":       M_T,
                "M_C":       M_C,
                "T_target":  float(T_target),
                "alpha":     float(alpha),
                "delta_F":   delta_F,
                "norm_K":    norm_K,
                "norm_K_new": norm_K_new,
                "norm_K_frac_drop": float(1.0 - norm_K_new / max(norm_K, 1e-12)),
                "N_kv":      int(T),
            })

        return K_modified

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
            # Slice to actual N_kv: HF may pre-compute a mask longer than the
            # current KV length (e.g. for speculative decoding or prefill with
            # non-zero cache).  Original LlamaAttention does the same slice.
            attn_weights = attn_weights + attention_mask[:, :, :, :N_kv]
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

        Phase 2c — adaptive widening
        ----------------------------
        After Phase 2b N=250 token-trace established that fixed k_window
        catastrophically degrades when ``coverage = k_window / N_kv`` falls
        below ~0.5 (mean Δ_NLL +2.29 nat at coverage 0.4, +4.14 at 0.3),
        the effective window is widened to enforce
        ``coverage ≥ self.cfg.coverage_floor`` (default 0.8).

        If ``self.cfg.enable_adaptive_widening`` is True, after the rank-local
        kernel runs, a heuristic safety check examines projection-space
        spread of selected vs rejected keys.  A "leaky" boundary triggers
        a single widen-and-retry, hard-capped at one widen per call to
        bound worst-case latency at 2×.

        Cost note: PCA is O(N_kv · D²) per head, called once per forward
        (or up to twice with safety net, but the second call reuses the
        already-computed axes).
        """
        B, H, _, D = Q.shape
        N_kv = K.shape[2]

        # Phase 2c Component A: adaptive k_window enforcing coverage floor
        k_eff = adaptive_k_window(
            k_window_min=self.cfg.k_window,
            n_kv=N_kv,
            coverage_floor=self.cfg.coverage_floor,
        )

        # ----- M1 TOP-K Q-AXIS REFERENCE -----
        # axis_source="q_topk_reference": call the M1 pure-PyTorch reference.
        # K/V are already post-repeat_kv ([B, H, N_kv, D]), so H_kv=H (n_per_kv=1)
        # inside topk_qaxis_attention_reference.  Spec §9.3: N>32K falls through to
        # SDPA at the routing layer, so k_eff < N_kv is always satisfied here.
        if self.cfg.axis_source == "q_topk_reference":
            # §6/§9.3: k_eff >= N_kv means top-K degenerates to full attention.
            # M1 rejects N <= k_eff; fall through to SDPA which is equivalent.
            if k_eff >= N_kv:
                return self._sdpa_forward(Q, K, V, None, 1, N_kv)
            Q_m1 = Q.squeeze(-2).to(torch.bfloat16)    # [B, H, D]
            K_m1 = K.to(torch.bfloat16)                # [B, H, N_kv, D]
            V_m1 = V.to(torch.bfloat16)                # [B, H, N_kv, D]
            O = topk_qaxis_attention_reference(Q_m1, K_m1, V_m1, k_eff)
            return O.unsqueeze(-2)                      # [B, H, 1, D]

        # ----- M4 TOP-K Q-AXIS TRITON (M2 + M3 pipeline) -----
        # axis_source="q_topk_triton": drop-in Triton replacement for M1.
        # The Triton kernel expects K/V with H_kv=num_key_value_heads (pre-GQA
        # repeat), because it handles GQA internally with n_per_kv=4.  K/V here
        # are post-repeat_kv ([B, H=32, N_kv, D]); we undo the repeat by taking
        # every n_rep-th head row: K[:, ::n_rep, :, :] → [B, H_kv=8, N_kv, D].
        # This is exact (repeat_kv copies each head n_rep times consecutively).
        if self.cfg.axis_source == "q_topk_triton":
            if k_eff >= N_kv:
                return self._sdpa_forward(Q, K, V, None, 1, N_kv)
            from dcr_attention.kernel.triton.topk_axis import topk_qaxis_attention
            Q_m4 = Q.squeeze(-2).to(torch.bfloat16)                          # [B, H, D]
            K_m4 = K[:, ::self.n_rep, :, :].to(torch.bfloat16).contiguous()  # [B, H_kv, N_kv, D]
            V_m4 = V[:, ::self.n_rep, :, :].to(torch.bfloat16).contiguous()  # [B, H_kv, N_kv, D]
            O = topk_qaxis_attention(Q_m4, K_m4, V_m4, k_eff)
            return O.unsqueeze(-2)                               # [B, H, 1, D]

        # ----- AXIS COMPUTATION -----
        # Phase 2c.4: axis_source="q" uses Q/||Q|| as the ordering axis.
        # Mathematically optimal for decode (N_q=1) — ranking K_j by
        # K_j · Q IS the attention score order, so top-k window directly
        # contains highest-attention keys.  See INSIGHT-AUDIT-3.
        # For prefill (N_q > 1), a single Q-axis is undefined; fall back
        # to PCA(K) per existing behaviour.
        N_q = Q.shape[2]
        use_q_axis = (self.cfg.axis_source == "q") and (N_q == 1)

        if use_q_axis:
            # Q has shape [B, H, 1, D]; squeeze query dim, normalise over D.
            # Cast to fp32 for unit-norm computation precision (Phase 2c.3
            # audit FINDING-2: keep axis chain in fp32).
            q_squeezed = Q.squeeze(-2).float()                   # [B, H, D]
            q_norms = q_squeezed.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            axes = q_squeezed / q_norms                           # [B, H, D] fp32
            # No SVD, no CPU transfer — order of magnitude faster than PCA path.
        else:
            # Phase 2c.3 audit FINDING-2 + FINDING-8:
            # K.cpu().float() halves transfer bandwidth (bf16 first, then upcast)
            # axes kept in fp32 (NOT cast to K.dtype) to avoid bf16 precision loss.
            K_cpu = K.cpu().float()                              # bf16->CPU->fp32
            axes_cpu = torch.zeros(B, H, D, dtype=torch.float32)
            for b in range(B):
                for h in range(H):
                    axis_bh, _ = compute_axis(K_cpu[b, h], positional_embedding=None)
                    axes_cpu[b, h] = axis_bh
            axes = axes_cpu.to(device=K.device)                  # fp32 on GPU

        out = rank_local_attention(
            Q, K, V, axes, k_window=k_eff,
        )

        # Phase 2c Component B: opt-in safety-net widening
        if self.cfg.enable_adaptive_widening and k_eff < N_kv:
            # Compute projections cheaply: scalars per (b, h, position)
            # Q has shape [B, H, 1, D]; axes [B, H, D]
            q_proj = (Q.squeeze(-2) * axes).sum(dim=-1, keepdim=True)  # [B, H, 1]
            k_proj = (K * axes.unsqueeze(-2)).sum(dim=-1)              # [B, H, N_kv]
            sort_idx = torch.argsort(k_proj, dim=-1)                   # [B, H, N_kv]

            leaky = detect_leaky_boundary(
                q_proj=q_proj.float(),
                k_proj=k_proj.float(),
                sort_idx=sort_idx,
                k_window_eff=k_eff,
                slack=0.5,
            )
            # If ANY (b, h) is leaky, widen once and retry on whole batch.
            # This is conservative: per-(b,h) widening would require a
            # variable-window kernel which we don't have.
            if leaky.any().item():
                k_eff_2 = widen_factor(prev_k=k_eff, n_kv=N_kv)
                if k_eff_2 > k_eff:
                    out = rank_local_attention(
                        Q, K, V, axes, k_window=k_eff_2,
                    )

        return out

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
