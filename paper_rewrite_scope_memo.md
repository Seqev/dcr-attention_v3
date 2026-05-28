# paper_rewrite_scope_memo.md

**Status:** Finalized scope memo for v3.1 paper rewrite
**Date:** 2026-05-28
**Authority:** Architect-level planning document (not the paper itself)
**Dataset basis:** Fully audited via canonical re-measurement (Phase v3.1-FINAL + v3.1-PERPASS)

---

## 0. Purpose

This memo locks the scope, narrative, numbers, and section structure for the
v3.1 paper rewrite before drafting begins. Every quantitative claim here has
survived canonical re-measurement (randomized-order, 50-iter warmup,
3-session protocol). Findings that did not survive are recorded in §5 as an
explicit retraction ledger so they cannot leak back into the manuscript.

The actual paper rewrite (estimated 2-3 weeks) consumes this memo as its
specification.

---

## 1. Executive narrative

The v3.0 paper framed DCR-attention as a sparse-attention speedup. At the
hero configuration (L4: N=32K, B=4, c=0.15) the M4 production kernel actually
ran **slower** than SDPA — 0.870× — a fact the v3.0 manuscript did not fully
confront. The v3.1 narrative is therefore not "look how fast we run." It is:

> *We instrumented why a principled sparse-attention pipeline loses to a
> fused baseline on real hardware, fixed the dominant cost, and — equally
> important — documented every mechanism that does NOT work, including
> failure modes the field routinely overlooks.*

The engineering result is a **parity crossing**: M-class kernel work moved
hero decode from 0.870× (sub-parity) to **1.061× over SDPA** (above parity),
which is **1.220× over the M4 starting point**, against a clean theoretical
ceiling of **1.243×**.

The scientific contribution is larger than the engineering one: eight
characterized negative results, anchored by a methodological finding about
benchmark bias that the discipline of this very project surfaced.

---

## 2. Canonical numbers (LOCKED)

All from `phase_v31_final_hero_measurements.json` and
`phase_v31_perpass_canonical.json`. Protocol: 50 warmup + 30 timed × 3
sessions, randomized (axis, config) order, cold reset between sessions.

### 2.1 Hero latency (L4: N=32000, B=4, c_floor=0.15)

| Path | e2e (ms) | vs SDPA | vs M4 | session variance |
|---|---|---|---|---|
| SDPA reference | 198.64 | 1.000× | 1.150× | 1.17% |
| M4 (q_topk_triton) | 228.43 | 0.870× | 1.000× | 0.12% |
| M6 (q_topk_triton_m6) | 188.20 | 1.056× | 1.214× | 0.78% |
| **M6+M5mixed (production)** | **187.29** | **1.061×** | **1.220×** | **0.098%** |

Hero variance is rock-solid (0.098% across 3 randomized sessions:
187.5 / 187.3 / 187.1 ms). The number that matters is reproducible.

### 2.2 Per-pass integrated cost (L4, canonical)

| Pass | mean ms/call | isolated baseline | integrated/isolated | status |
|---|---|---|---|---|
| u_Q | 0.0153 | 0.042 | 0.365× | small-pass, not load-bearing |
| Pass-1 | 0.6496 | 0.596 | 1.090× | **CONFIRMED** (matches M5.a 1.08×) |
| Pass-2 (M4) | 3.2233 | 3.268 | 0.986× | clean, dominant pass |
| Pass-2 (M6) | 0.7610 | 0.694 | 1.097× | clean |
| Pass-3 | 0.4036 | 0.478 | 0.844× | high variance (5.97%); NOT robust |

Pass-2 dominates M4 attention (3.22 / 4.29 = 75%). M6 reduces it to 0.76 ms.
M4−M6 Pass-2 delta = 2.46 ms/layer × 16 = 39.4 ms ≈ canonical M4−M6 e2e delta
40.23 ms. **The M6 hero saving IS the clean Pass-2 kernel delta — nothing
more, nothing hidden.**

### 2.3 Clean theoretical ceiling

```
sum M4 passes/layer = 0.015 + 0.650 + 3.223 + 0.404 = 4.2918 ms
non_attn = 228.43 − 16 × 4.2918 = 159.76 ms
ceiling vs SDPA = 198.64 / 159.76 = 1.243×
```

Production hero (1.061×) has captured ~26% of the gap between M4-parity and
this ceiling. The remaining headroom requires eliminating attention cost
entirely — precisely what the failed ABKV, abandoned INT4, and deferred
Pass-3 modernization were all chasing.

---

## 3. Section-by-section rewrite plan

### §sec:pass3-bottleneck → CORRECTION

The v3.0 manuscript hypothesized Pass-3 (gather + online softmax) as the
attention bottleneck. Step 0 profiling (and confirmed canonically) shows
**Pass-2 (the sort) dominates at 75% of M4 attention cost**; Pass-3 is ~9%.
This correction motivates the entire M6 direction. Cite the per-pass table
(§2.2). State plainly that the prior hypothesis was wrong and the data
redirected the work.

### §sec:abkv-day1 → REVISION (formal failure characterization)

The Day-1 synthetic experiments validated the ABKV *kernel*, not the ABKV
*axis hypothesis*. State the hypothesis formally:

> ∃ K̄ ∈ ℝ^d : ∀ i,j, top-k(QᵢᵀK) ≈ top-k(K̄ᵀK)

Phase 8.1c falsifies it on real Llama-3.2-1B: top-10-attention overlap =
2.08% (vs ≥90% the "dominant-keys" story requires); hero output cos_sim
0.36 ± 0.45 (threshold 0.95); anti-correlation in some heads (−0.56).

Mechanism: (1) attention-sink phenomenon — early-position tokens dominate
|K| magnitude but receive low attention weight; (2) RoPE phase scrambling —
the angular distribution of rotation phases drives K̄ → 0 by phase averaging
at large N. The synthetic generator placed Q and the important-K cluster on
the same axis by construction; real models have no such alignment.

### §sec:trajectory-stability → NEW

Phase QTRAJ: top-K reshuffles ~26%/step across autoregressive decode
(mean Δ = 0.257), cumulative drift saturates ~47% by step 49, zero stable
(layer, head) pairs out of 512. Verdict NOT_VIABLE — no warm-start
amortization possible. This empirically establishes **per-Q-step necessity**:
the M4 mechanism (per-step Q-axis projection) is not an inefficiency to be
amortized away; it is structurally required. (Middle layers 6-9 marginally
more stable than endpoints — note as an aside, not a claim.)

### §sec:hero-recovery → REWRITE

Lead section. Structure:
1. Pass-2 dominance (the redirect)
2. M6 Triton native top-K: 4.75× Pass-2 kernel speedup, drop-in,
   bit-equivalent to M4 (Tier 1 100% index match)
3. M5 cuBLAS Pass-1: wins single-batch long-context (L3 2.36× isolated),
   loses multi-batch hero (M=4 GQA) → workload-aware mixed dispatch
4. Canonical hero result: 0.870× → 1.061× SDPA (1.220× over M4)
5. Honest scope: the parity crossing, against ceiling 1.243×

Dual framing throughout: "1.06× over SDPA (parity crossing)" as primary,
"1.22× over M4 baseline" as the kernel-improvement number.

### §sec:methodology → NEW (the centerpiece)

The measurement-discipline narrative. Prior intermediate measurements
claimed 1.14-1.15× hero; canonical re-measurement (randomized order,
50-iter warmup, 3 sessions) corrected this to 1.06×. The ~9% bias came from
sequential measurement order (warm-GPU advantage to later-measured paths)
plus insufficient warmup. Baselines (SDPA, M4) reproduced exactly; only the
optimized paths were biased low — exactly the direction that flatters a
new method.

This is the honest, valuable contribution: a project that caught its own
optimistic bias, pre-publication, because it built the discipline to catch
it. Includes the kernel-math reconciliation (M6 saving = clean Pass-2 delta)
as the cross-check that distinguishes signal from artifact.

### §sec:discussion → ENHANCED (eight postulates, §4)

### §sec:future-work → UPDATED

- INT8 KV (safe alternative to infeasible INT4; ~1.16-1.17× expected)
- Pass-3 modernization (ROI now uncertain given clean per-pass numbers)
- Adaptive k_eff (power-law concentration; requires QTRAJ-style validation
  due to Q-dependence risk shared with ABKV)
- **Paper 3 candidate:** KV layout co-design — the Phase 8.1b contiguous-vs-
  gather 2.42× kernel advantage is real but untransferable under the current
  per-step mechanism; recovering it requires storing K in a Q-relevant sorted
  layout that ABKV could not provide

### §sec:related-work → UPDATED

- KIVI / AWQ-K: cite, and add the small-model-brittleness limit (postulate 7)
- cuBLAS / vendor BLAS: cite the M=4 GQA degradation (postulate 6)
- StreamingLLM (attention sinks): mechanistic anchor for postulate 2
- NSA / Quest / InfLLM: block-sparse alternatives, contrasted with the
  per-step Q-axis necessity (postulate 4)

---

## 4. Eight negative postulates (Lessons Learned)

| # | Source | Statement |
|---|---|---|
| 1 | Phase 8.1c | Synthetic-data quality validation does not transfer to real LLMs; the synthetic generator's by-construction alignment is the artifact. |
| 2 | Phase 8.1c | Mean-based K statistics are dominated by attention sinks; any static projection axis (mean-K, PCA-1, calibration prior) is a dead end on real models. |
| 3 | Phase 8.1c | Index-set overlap is a false quality metric; only attention-output cosine similarity is a valid gate. |
| 4 | Phase QTRAJ | Cumulative top-K trajectory drift saturates (~47% by step 50); no long-window amortization exists, confirming per-Q-step necessity. |
| 5 | Cross-cutting | HF dispatch / non-attention overhead is orthogonal to the algorithmic speedup ratio; it shifts absolute latency, not the kernel-efficiency profile. |
| 6 | Phase M5.b | cuBLAS routing degrades at small GEMM dimensions (M=4 GQA): Tensor Cores do not engage and per-launch overhead dominates in multi-batch regimes. Vendor BLAS abstractions are not a free win at small M. |
| 7 | Phase INT4.a | INT4 KV quantization is infeasible on 1B-class models at strict quality thresholds; KIVI's <0.1% PPL claim does not scale down — narrower head distributions concentrate outliers that destroy quantization quality (catastrophic min cos_sim 0.05-0.71 in localized heads, NOT attention sinks). |
| 8 | Phase v3.1-FINAL + PERPASS | Sequential kernel benchmarking with insufficient warmup produces systematic optimistic bias (~9%) for later-measured paths via warm-GPU-state advantage. Rigorous comparison requires randomized-order, high-warmup (≥50 iter), multi-session protocols. Per-pass instrumentation inherits the same bias and must use the same rigor. |

Postulate 8 is the methodological centerpiece and should be foregrounded:
it is the most transferable lesson and the one most likely to change how
others measure.

---

## 5. Retraction ledger (internal discipline record — DO NOT cite in paper)

These findings appeared in intermediate analysis and were caught and dropped
before publication. Recorded here so they do not leak back into the
manuscript.

| Retracted claim | Why it was wrong | Caught by |
|---|---|---|
| Pass-2 "compositional inflation" 1.29× (0.96 ms/layer hidden cost) | Derived from biased M6.d e2e delta (56 ms); canonical M4−M6 = 40.23 ms = clean Pass-2 isolated delta. No inflation existed. | v3.1-FINAL, confirmed by PERPASS (ratio 0.986×) |
| Pass-3 "deflation" 0.77× (L2 cache synergy) | Did not reproduce under canonical protocol; config-scatter 1.00 / 1.44 / 0.84, inconsistent direction; L4's 0.84 within 5.97% session jitter. Artifact + small-pass noise. | v3.1-PERPASS |
| Theoretical ceiling 1.472× | Derived from biased non_attn (133.9 ms). Clean non_attn = 159.76 ms → ceiling 1.243×. | v3.1-PERPASS |

Both retracted "timing asymmetry" findings were going to be a paper section.
They are gone. The methodology section (postulate 8) replaces them and is
strictly stronger: it explains *why* such asymmetries were illusory.

---

## 6. Tables for the manuscript

**Table 12 (hero latency)** — from §2.1. Four rows (SDPA, M4, M6, M6+M5mixed),
columns: e2e ms, speedup vs SDPA, speedup vs M4, session variance. Caption
must state the canonical protocol (50 warmup, 30 timed, 3 randomized sessions).

**Table 13 (per-pass breakdown)** — from §2.2. Note which ratios are robust
(Pass-1, Pass-2) vs high-variance (Pass-3, small passes). Do not present
Pass-3 ratio as a finding.

**Table 14 (negative postulates)** — from §4. The summary that anchors the
discussion.

---

## 7. Future directions (v3.2 / Paper 3)

- **v3.2 algorithmic:** INT8 KV (safe), adaptive k_eff (with validation),
  targeting ~1.18-1.22× — but bounded by the 1.243× ceiling under current
  HF wrapper.
- **v3.2 wall-clock:** non-attention / HF dispatch reduction. Lowers absolute
  latency for both baselines equally (postulate 5) — a serving-throughput
  story, not a kernel-ratio story. Frame honestly as such.
- **Paper 3:** KV layout co-design. The contiguous-vs-gather 2.42× kernel
  advantage (Phase 8.1b) is real but trapped — recovering it requires a
  Q-relevant sorted K layout, which attention-sink phenomenon precludes for
  any static axis. This is a genuinely open architectural problem.

---

## 8. Provenance (artifacts to cite as reproducibility appendix)

| Phase | Artifact | Establishes |
|---|---|---|
| 8.1a-c | phase_8_1c_quality_gate.json | ABKV falsification (postulates 1-3) |
| QTRAJ | phase_qtraj_query_trajectory.json | Trajectory drift (postulate 4) |
| M6.b | phase_m6_b_kernel_bench.json | 4.75× Pass-2 kernel speedup |
| M6.d | phase_m6_d_e2e_latency.json | M6 integration (superseded by canonical) |
| M5.a/b | phase_m5_a/b artifacts | cuBLAS GQA degradation (postulate 6) |
| M5.d-light | phase_m5d_light_e2e.json | Workload-aware mixed dispatch |
| INT4.a | phase_int4_a_quality_spike.json | INT4 infeasibility (postulate 7) |
| v3.1-FINAL | phase_v31_final_hero_measurements.json | **Canonical hero (postulate 8)** |
| v3.1-PERPASS | phase_v31_perpass_canonical.json | **Clean per-pass + ceiling** |

All measurements: Llama-3.2-1B, RTX 4060 Ti, torch 2.5.1+cu121,
triton 3.1.0, seed 0, git e156dc9.

---

## Closing note

The v3.1 paper's value sits in the science, not the speedup. A 1.06×
parity crossing with rigorous methodology and eight characterized negative
results — including a measurement-bias lesson the project caught on itself —
is more credible and more useful to the field than a 1.15× number obtained
through sequential-warm bias. The discipline that produced the smaller,
honest number is itself the contribution.
