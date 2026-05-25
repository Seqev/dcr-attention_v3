"""
Configuration for the DCR Llama wrapper.

All thresholds and knobs exposed to the user live in :class:`DCRLlamaConfig`
so experiments can adjust them without touching any implementation file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Optional


AxisSource = Literal["pca", "positional"]


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

        Phase 2c will add ``"learned"`` (via ``AxisProjector``).
    enable_dcr
        Master switch.  ``False`` → all forward calls fall through to SDPA,
        useful for ablation runs (paper §5).  Bypasses the entire rank-local
        path including the dispatcher.
    layer_filter
        If not ``None``, only the listed ``layer_idx`` values use DCR; all
        other layers always use SDPA.  Layer-wise studies (e.g. "DCR helps
        deeper layers more"): pass e.g. ``frozenset({16, 17, ..., 31})``.
    """

    k_window: int = 64
    T_dispatch: int = 4096
    axis_source: AxisSource = "pca"
    enable_dcr: bool = True
    layer_filter: Optional[FrozenSet[int]] = None

    def __post_init__(self) -> None:
        if self.k_window <= 0 or self.k_window % 2 != 0:
            raise ValueError(
                f"k_window must be a positive even integer, got {self.k_window}"
            )
        if self.T_dispatch < 0:
            raise ValueError(f"T_dispatch must be ≥ 0, got {self.T_dispatch}")
        if self.axis_source not in ("pca", "positional"):
            raise ValueError(
                f"axis_source must be 'pca' or 'positional', got {self.axis_source!r}"
            )


DEFAULT_DCR_LLAMA_CONFIG = DCRLlamaConfig()
