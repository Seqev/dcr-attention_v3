# DCR-Attention v3.1

[![DOI](https://zenodo.org/badge/1249514842.svg)](https://doi.org/10.5281/zenodo.20385783)

Sparse-attention KV-cache work on Llama-3.2-1B (RTX 4060 Ti). This repo
documents **both what works and what does not** — including findings we
retracted before publication.

## Result

At the hero configuration (N=32K, B=4, c=0.15), M-class kernel work moved
decode latency from sub-parity to above parity vs SDPA:

| Path | e2e (ms) | vs SDPA | vs M4 |
|---|---|---|---|
| SDPA | 198.64 | 1.000× | — |
| M4 (v3.0) | 228.43 | 0.870× | 1.000× |
| **M6 + M5-mixed (v3.1)** | **187.29** | **1.061×** | **1.220×** |

Clean theoretical ceiling (0-cost attention kernel): **1.243×**. Production
captures ~26% of the M4-parity → ceiling gap.

Numbers are canonical: 50-iter warmup, 30 timed, 3 randomized-order sessions,
hero variance 0.098%.

## What the work actually contributes

The value is in the science, not the speedup. Eight characterized negative
results:

| # | Statement |
|---|---|
| 1 | Synthetic-data quality validation does not transfer to real LLMs. |
| 2 | Mean-K statistics are dominated by attention sinks; static projection axes are a dead end. |
| 3 | Index-set overlap is a false quality metric; only output cosine similarity is a valid gate. |
| 4 | Top-K trajectory drift saturates (~47% by step 50); no warm-start amortization — per-Q-step is structurally necessary. |
| 5 | Dispatch/non-attention overhead is orthogonal to the algorithmic ratio. |
| 6 | cuBLAS degrades at small GEMM dims (M=4 GQA): no Tensor-Core engagement, launch overhead dominates. |
| 7 | INT4 KV is infeasible on 1B-class models; KIVI's <0.1% PPL claim does not scale down (outlier-driven, not sink-driven). |
| 8 | Sequential benchmarking with low warmup produces ~9% optimistic bias for later-measured paths. Rigorous comparison needs randomized-order, high-warmup, multi-session protocols. |

Postulate 8 is the centerpiece: an earlier intermediate claim of 1.14-1.15×
hero was corrected to 1.06× by canonical re-measurement. We caught our own
optimistic bias pre-publication because the project was built to catch it.

## Structure

```
docs/paper_rewrite_scope_memo.md   Scope memo for the v3.1 paper (incl. retraction ledger)
results/                           Canonical measurements + key falsification artifacts
REPRODUCIBILITY.md                 Env, seeds, protocol
```

## Status

Work-in-progress. This drop is the **scope memo + measurement artifacts**;
the full v3.1 manuscript is a separate forthcoming rewrite. The retraction
ledger (scope memo §5) is kept public deliberately as a discipline record.

## Environment

Llama-3.2-1B · RTX 4060 Ti · torch 2.5.1+cu121 · triton 3.1.0 · seed 0
