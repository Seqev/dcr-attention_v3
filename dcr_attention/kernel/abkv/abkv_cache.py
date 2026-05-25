"""
ABKV (Axis-Block Key-Value) cache dataclass.

Pre-sorts KV entries at prefill time by their projection onto a fixed axis
vector so that decode-time top-k selection can be done by scanning a prefix
of the sorted array rather than all N keys.

Design doc: dcr_attention/kernel/abkv/DESIGN.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ABKVCache:
    """
    Pre-sorted KV cache for fast top-k decode attention.

    At prefill time all N keys per head are projected onto `axis_vec` and
    sorted descending so that position 0 has the highest axis score.  Decode
    needs only a prefix of length k_eff from each head to approximate the
    full top-k selection; the quality of that approximation degrades smoothly
    as coverage_floor decreases.

    Shapes (all tensors stored on the same device as the input K_cache):
      K_sorted   : [B, H_kv, N, D]  bf16
      V_sorted   : [B, H_kv, N, D]  bf16
      axis_scores: [B, H_kv, N]     fp32  (pre-sorted descending)
      axis_vec   : [H_kv, D]        fp32  (projection axis per KV head)
      sort_perm  : [B, H_kv, N]     int64 (original index at each sorted pos)
    """

    K_sorted: Tensor     # [B, H_kv, N, D] bf16
    V_sorted: Tensor     # [B, H_kv, N, D] bf16
    axis_scores: Tensor  # [B, H_kv, N]    fp32  sorted descending
    axis_vec: Tensor     # [H_kv, D]       fp32
    sort_perm: Tensor    # [B, H_kv, N]    int64

    num_bins: int
    bin_size: int        # = ceil(N / num_bins)
    N: int
    H_kv: int
    D: int

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def B(self) -> int:
        return int(self.K_sorted.shape[0])

    @property
    def device(self) -> torch.device:
        return self.K_sorted.device

    def bin_score_boundaries(self) -> Tensor:
        """
        Score at the start of each bin per (B, H_kv), shape [B, H_kv, num_bins].

        bin i spans sorted positions [i*bin_size, (i+1)*bin_size).
        boundary[b, h, i] = axis_scores[b, h, i*bin_size].
        """
        starts = [min(i * self.bin_size, self.N - 1) for i in range(self.num_bins)]
        idx = torch.tensor(starts, dtype=torch.long, device=self.device)
        return self.axis_scores[:, :, idx]  # [B, H_kv, num_bins]

    def num_bins_needed(self, k_eff: int) -> int:
        """
        Minimum number of leading bins whose union covers k_eff sorted positions.
        """
        return min(self.num_bins, (k_eff + self.bin_size - 1) // self.bin_size)

    def __repr__(self) -> str:
        return (
            f"ABKVCache(B={self.B}, H_kv={self.H_kv}, N={self.N}, D={self.D}, "
            f"num_bins={self.num_bins}, bin_size={self.bin_size}, "
            f"device={self.device})"
        )
