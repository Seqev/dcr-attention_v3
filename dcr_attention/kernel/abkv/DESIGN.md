# ABKV — Axis-Block Key-Value: Design Document

**Status**: Day 1 CPU prototype complete. Day 2 (Triton kernel) pending GPU free.  
**Last updated**: 2026-05-11

---

## 1. Motivation

M4 (DCR Triton kernel) has a fixed overhead O_fixed that makes its break-even
point N* ≈ 32K tokens at B=1, c=0.10 on RTX 4060 Ti (288 GB/s):

```
P0a measurement (2026-05-10):
  R3: N=32K, B=1, c=0.10 → SDPA 102.3ms, M4 102.8ms → speedup 0.995×
  R2: N=8K,  B=4, c=0.10 → SDPA 107.0ms, M4  95.6ms → speedup 1.12×
```

Phase 2 target is N* ≈ 8-10K at B=1. ABKV achieves this by eliminating
the dominant component of O_fixed: the per-step all-N axis projection scan.

---

## 2. Key Idea

DCR top-K at decode step t:
1. Project all N_kv keys onto u_Q = Q/‖Q‖ → scores[0..N-1]
2. Sort scores → top-k_eff indices
3. Gather K_sel, V_sel, compute attention

Step 1 is O(N·D) every decode step.  For N=32K, D=64, 32 layers: 32K×64×32
= 65M multiplies per token.

ABKV replaces step 1 with a prefill-time pre-sort on a fixed axis vector `v`:

**Prefill** (once, amortised):
- Compute `v` = mean unit-K per head  (O(N·D) total, amortised over decode)
- Project all K onto v → scores  (O(N·D))
- Sort K, V descending by score  (O(N log N))
- Store sorted K_sorted, V_sorted

**Decode** (each step):
- Take K_sorted[:k_eff], V_sorted[:k_eff]  (O(1) with pre-sorted storage)
- Compute attention on k_eff keys  (O(k_eff·D))

Total decode FLOPs: O(k_eff·D) instead of O(N·D + k_eff·D).
At c=0.10 this is 10× fewer key-scoring operations.

---

## 3. Why the Axis Works

The mean-K unit vector `v = mean(K/‖K‖)` approximates the dominant direction
of the key distribution per head.  In transformer KV caches this direction
corresponds to "important" tokens (subject tokens, boundary tokens, rare
vocabulary) which:

1. Tend to be attended to across many query positions (high total attention)
2. Cluster in a narrow cone in key space

When Q is correlated with this dominant direction (typical for queries about
the same semantic content), sorting K by `v` is a strong proxy for sorting by
true Q·K score.

**Empirical validation (Day 1 sweep)**:
```
N=512, important_frac=0.15, amp=6.0
  c=0.10 → cos_sim 0.907  (≥ 0.85 gate PASS)
  c=0.20 → cos_sim 0.999
  c=0.30 → cos_sim 0.999
  c=0.50 → cos_sim 0.999
  c=0.80 → cos_sim 1.000
```

`cos_sim` is invariant to `num_bins` because the sort order determines which
keys are in the prefix — bin boundaries are metadata for the kernel (Day 2).

---

## 4. Data Layout

```
ABKVCache fields:
  K_sorted   [B, H_kv, N, D]  bf16  — keys in descending axis-score order
  V_sorted   [B, H_kv, N, D]  bf16  — values, same permutation
  axis_scores[B, H_kv, N]     fp32  — pre-sorted scores (for binary search)
  axis_vec   [H_kv, D]        fp32  — the projection axis, one per KV head
  sort_perm  [B, H_kv, N]     int64 — original index at each sorted position

  num_bins   int               — number of equal-size bins
  bin_size   int               — ceil(N / num_bins)
  N, H_kv, D int
```

The first `k_eff` entries of `K_sorted[:, h, :, :]` are the candidates for
decode step h.  In Day 2 (Triton kernel), `num_bins` determines how many
full bins are fetched from HBM per step, enabling coalesced block reads.

---

## 5. Algorithm: build_abkv_cache

```
Input: K_cache [B, H_kv, N, D] bf16, V_cache [B, H_kv, N, D] bf16, num_bins

1. If axis_vec is None:
     K_unit = K_cache / ‖K_cache‖ per key
     axis_raw = mean(K_unit, dim=[B, N])  → [H_kv, D]
     axis_vec = axis_raw / ‖axis_raw‖    → [H_kv, D] fp32 unit

2. scores[b, h, n] = sum_d K_cache[b,h,n,d] * axis_vec[h,d]  → [B,H_kv,N] fp32

3. sort_perm = argsort(scores, descending=True, stable=True)  → [B,H_kv,N]

4. K_sorted = gather(K_cache, dim=2, index=sort_perm)
   V_sorted = gather(V_cache, dim=2, index=sort_perm)
   axis_scores_sorted = gather(scores, dim=-1, index=sort_perm)

5. bin_size = ceil(N / num_bins)
   Return ABKVCache(...)
```

---

## 6. Algorithm: abkv_attention_reference

```
Input: Q [B, H, D] bf16, cache: ABKVCache, k_eff: int, scan_budget: int|None

mode A — prefix (scan_budget=None or scan_budget==k_eff):
  K_cand = K_sorted[:, :, :k_eff, :]   # leading prefix, no Q computation
  V_cand = V_sorted[:, :, :k_eff, :]
  → O = online_softmax_attn(Q, expand_GQA(K_cand), expand_GQA(V_cand))

mode B — rerank (scan_budget=S > k_eff):
  K_cand = K_sorted[:, :, :S, :]        # scan more candidates
  qk = Q · K_cand / sqrt(D)            # rerank within S by true Q·K
  topk_idx = topk(qk, k_eff)           # true top-k within S
  K_sel = gather(K_cand, topk_idx)
  V_sel = gather(V_cand, topk_idx)
  → O = online_softmax_attn(Q, K_sel, V_sel)

GQA expansion: K/V [B,H_kv,k,D] → [B,H,k,D] via repeat_interleave(n_per_kv)
```

Mode A (prefix) is the Day 2 target: zero FLOPs for axis projection at decode.  
Mode B (rerank) gives oracle quality at scan_budget=N; useful for ablations.

---

## 7. Quality Model

For structured data where important keys occupy fraction α of the cache:

```
k_eff ≥ α·N  →  all important keys captured  →  cos_sim ≈ 1.0
k_eff < α·N  →  partial coverage of important keys
             →  cos_sim depends on Q·K dominance ratio
```

Quality degrades gracefully as c → 0; the axis quality determines how
far below c=1.0 the output remains useful.

Gate requirements (from DCR project spec):
```
c = 0.80  →  cos_sim ≥ 0.95
c = 0.10  →  cos_sim ≥ 0.85  (with structured data; random data may be lower)
```

---

## 8. Expected N* Reduction

Current M4 fixed overhead: O_fixed/α ≈ 28,800 token-equivalents
  → N*(B=1, c=0.10) ≈ 32K

ABKV eliminates the all-N projection scan (dominant component):
  New O_fixed ≈ bin_lookup overhead only
  → N*(B=1, c=0.10) target ≈ 8-10K

This is a 3-4× improvement in break-even, enabling M4 speedup at
conversational (8-16K) context lengths.

---

## 9. Day 2 Plan (Triton kernel)

1. **Block layout**: Store K_sorted in contiguous bin-aligned blocks.
   Each bin occupies exactly `bin_size × D` bf16 values = one L2 cache block.

2. **Decode kernel** (`abkv_topk_triton`):
   - Compute Q·axis_vec (one dot product, not N)
   - Binary search in `axis_scores` to find bin boundary for coverage c
   - Load `num_bins_needed` contiguous blocks from HBM
   - Run online softmax within loaded keys

3. **Coalesced reads**: bin_size must be a multiple of 128B / sizeof(bf16) = 64
   elements. Recommended: bin_size = 64 or 128 (D=64 → one or two rows).

4. **Integration**: Replace `topk_qaxis_triton` in DCRLlamaAttention with
   `abkv_topk_triton`; cache construction at prefill added to
   `DCRLlamaAttention.forward` when `prefill=True`.

---

## 10. Files

```
dcr_attention/kernel/abkv/
  __init__.py          — public exports
  abkv_cache.py        — ABKVCache dataclass
  abkv_reference.py    — build_abkv_cache, abkv_attention_reference, sdpa_reference
  DESIGN.md            — this document

tests/kernel/
  test_abkv_reference.py    — 20 unit tests, all passing
  test_abkv_bin_size_sweep.py — bin-size sweep, JSON output to /tmp/abkv_day1/

/tmp/abkv_day1/
  bin_size_sweep.json   — sweep results (N=512, 5 bins × 5 coverages)
  bin_size_sweep.log    — console output
```
