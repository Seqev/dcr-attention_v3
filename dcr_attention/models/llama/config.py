"""
Configuration for the DCR Llama wrapper.

All thresholds and knobs exposed to the user live in :class:`DCRLlamaConfig`
so experiments can adjust them without touching any implementation file.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Optional


AxisSource = Literal["pca", "positional", "q", "q_topk_reference", "q_topk_triton"]


@dataclass(frozen=True)
class DCRLlamaConfig:
    r"""
    User-facing knobs for ``DCRLlamaAttention`` and the dispatcher.

    Attributes
    ----------
    k_window
        Window size (``2 * half_k``) for the rank-local kernel.  Llama-3
        head dim is 128 — empirically (Phase 1.3 v7, Phase 2-pre v9)
        ``k_window=64`` is the smallest value that gives competitive numbers
        without quality loss in synthetic correctness tests.  The user
        validates against their downstream task.
    T_dispatch
        Threshold on N_kv (= length of the cached K sequence) below which the
        wrapper falls back to dense SDPA even in the decode branch.  From
        Phase 2-pre v9 measurements:

          * N_kv ≤ 1024   : DCR/SDPA = 1.40-2.69 (DCR slower)
          * N_kv ≈ 4096   : DCR/SDPA = 0.52-1.98 (crossover)
          * N_kv ≥ 16384  : DCR/SDPA = 0.13-0.27 (DCR 4-8× faster)

        Default 4096 picks the conservative side of the crossover.  Tune
        per hardware; see ``docs/design/phase2a_llama_wrapper.md`` §3.
    axis_source
        How the rank-window axis is computed each forward.

          * ``"pca"`` (default): top right-singular vector of the post-RoPE
            K matrix per head.  RoPE-aware; recommended for Llama-3.
          * ``"positional"``: low-frequency RoPE pair selection.  Brittle for
            long contexts (N ≥ 128K) and content-agnostic; included for
            ablation only.
          * ``"q"`` (Phase 2c.4 — decode-only): use ``Q / ||Q||`` directly
            as the ordering axis.  For decode (N_q=1) this is mathematically
            optimal: ranking K_j by ``K_j · Q`` IS the attention score order.
            Cost: O(N · D) projection, no SVD.  Falls back to ``"pca"`` for
            prefill (N_q > 1) since a single Q-axis isn't well-defined for
            multi-query batches.  Phase 2c.3 audit INSIGHT-AUDIT-3 motivates
            this option; Phase 2c.4 sweep validates it empirically.

        Phase 2c will add ``"learned"`` (via ``AxisProjector``).
    enable_dcr
        Master switch.  ``False`` → all forward calls fall through to SDPA,
        useful for ablation runs (paper §5).  Bypasses the entire rank-local
        path including the dispatcher.
    layer_filter
        If not ``None``, only the listed ``layer_idx`` values use DCR; all
        other layers always use SDPA.  Layer-wise studies (e.g. "DCR helps
        deeper layers more"): pass e.g. ``frozenset({16, 17, ..., 31})``.
    coverage_floor
        Minimum fraction of the KV context selected at each decode step.
        At each ``_dcr_forward`` call the effective window size is

            k_window_eff = max(k_window, int(n_kv * coverage_floor))

        capped at ``n_kv``.

        **Validated quality tiers (N=32K, Llama-3.2-1B, multi-seed):**

        - ``c ≥ 0.15``: STRICT quality (mean ΔPPL ≤ 0.5%, 5-seed validated)
        - ``c = 0.10``: PERMISSIVE quality (mean ΔPPL +1.078% ± 0.149 pp, 3-seed)
        - ``c < 0.10``: UNUSABLE (ΔPPL > 2% expected)

        Default ``0.15`` is the hero deployment point (HERO_VERIFICATION_REPORT.md).

        Set to ``0.0`` to disable adaptive widening and reproduce the fixed
        ``k_window`` behaviour (Phase 2b baseline; not recommended for
        production).
    enable_adaptive_widening
        Phase 2c — opt-in safety net.  After the rank-local window is
        selected, monitor the spread of projections of rejected vs selected
        keys.  If the nearest rejected key in projection space is closer to
        the query projection than the furthest selected key (with a 0.5
        slack), widen the window once and retry.  Hard-capped at one widen
        per ``_dcr_forward`` call to bound worst-case latency at 2×.

        Default ``False`` because (a) the adaptive coverage_floor already
        prevents the dominant failure mode and (b) the safety net adds
        complexity we want to validate in opt-in ablation runs first.
    """

    k_window: int = 64
    T_dispatch: int = 4096
    axis_source: AxisSource = "pca"
    enable_dcr: bool = True
    layer_filter: Optional[FrozenSet[int]] = None
    coverage_floor: float = 0.15
    enable_adaptive_widening: bool = False

    def __post_init__(self) -> None:
        if self.k_window <= 0 or self.k_window % 2 != 0:
            raise ValueError(
                f"k_window must be a positive even integer, got {self.k_window}"
            )
        if self.T_dispatch < 0:
            raise ValueError(f"T_dispatch must be ≥ 0, got {self.T_dispatch}")
        if self.axis_source not in ("pca", "positional", "q", "q_topk_reference", "q_topk_triton"):
            raise ValueError(
                f"axis_source must be one of 'pca', 'positional', 'q', "
                f"'q_topk_reference', or 'q_topk_triton', got {self.axis_source!r}"
            )
        if not (0.0 <= self.coverage_floor <= 1.0):
            raise ValueError(
                f"coverage_floor must be in [0, 1], got {self.coverage_floor}"
            )
        if self.coverage_floor < 0.10:
            warnings.warn(
                f"coverage_floor={self.coverage_floor} is below validated range. "
                f"Expected ΔPPL > 2% (UNUSABLE quality). "
                f"Recommended minimum: coverage_floor=0.15 (STRICT, hero deployment).",
                RuntimeWarning,
                stacklevel=3,
            )
        elif self.coverage_floor < 0.15:
            warnings.warn(
                f"coverage_floor={self.coverage_floor} is in boundary regime (PERMISSIVE). "
                f"Expected ΔPPL ~1.0-1.3% at N=32K. "
                f"For STRICT quality use coverage_floor=0.15.",
                RuntimeWarning,
                stacklevel=3,
            )


DEFAULT_DCR_LLAMA_CONFIG = DCRLlamaConfig()
