# Data tree — v3 release

This directory contains the curated measurement JSONs cited by
`paper/main.pdf`. Inclusion scope is **Option C (curated)** per Phase 7.1
gate G4: cited paths plus supporting summaries that anchor specific
claims in the paper.

## Included

| Subdirectory | Contents | Cited by |
|--------------|----------|----------|
| `phase_2_repro/m1_acceptance/` | M1 reference top-K 5-seed at N=2K, c=0.5: 5 per-seed JSONs + summary | §4.1 Tab 9 |
| `phase_2_repro/m4_acceptance/` | M4 Triton fused top-K 5-seed at N=2K, c=0.5: 5 per-seed JSONs + summary + M4-M1 drift | §4.1 Tab 10 |
| `phase_2_repro/v2_hero/` | v2.0 hero re-validation 5-seed at N=20K, c=0.5: 5 per-seed JSONs + summary | §4.2 Tab 11 |
| `hero_verification/` | HERO Phase A 5-seed at N=32K, c=0.15: 4 per-seed JSONs (seed 0 was the original P1a single-seed; values in summary) + summary + Phase B latency L1-L5 | §4.3 Tab 12, §5.1 Tab |
| `pre_phase2/` | Q7 secondary 3-seed at N=32K, c=0.10 | §4.4 Tab 13 |
| `phase_3_causal_full/` | Theorem 3 causal-run aggregate analysis (Wilcoxon, Mann-Whitney, dose-response, α-fit, random-spectrum) | §3.X.3, §3.Y |
| `phase_3_causal_pilot/` | v1-v6 pilot iteration results (transparency / methodology) | §3.X footnote |
| `abkv_day1/` | ABKV synthetic-data bin-size sweep (feasibility) | §5.3 Tab |

## Not included

- **1,500 per-cell causal checkpoint JSONs** (`phase_3_causal_full/checkpoints/`).
  These are the per-(layer, head, prompt, context) raw measurement files
  consumed by `scripts/theorem3_causal_analysis.py` to produce the
  aggregate `analysis.json` included above. Excluded to keep repo size
  reasonable (~1500 small files); re-derivable by re-running the causal
  measurement script in `tests/integration/test_theorem3_causal_full.py`.

- **Phase-1.5 / pre-phase-2 transient measurements** beyond Q7 stability.
  These motivated the multi-seed protocol but are not cited as
  load-bearing in the paper.

- **Run logs** (`data/internal/*.log`). Internal-process artifacts not
  cited by the paper.

## Reproducibility

Each per-seed JSON contains `seed`, `ppl_baseline_sdpa`, `ppl_dcr` (or
`ppl_m1_topk` / `ppl_m4_topk`), `delta_ppl_pct`, and seed-state hashes
(`torch_seed`, `numpy_seed`, `python_seed_state_hash`) for exact
reproducibility. The summary JSONs aggregate these into 5-seed mean,
std, min, max, classification, and verdict fields.

The Phase 6 audit (see Phase 6 audit report in the Zenodo / supplementary
deposit, when available) verified every paper-cited number against the
canonical value in the corresponding JSON.

## Citation of specific JSONs

Paper tables explicitly cite paths like
`data/raw/phase_2_repro/m1_acceptance/`; the table captions point
readers here. The Phase 6 audit's Pass 1 verified zero
paper-to-JSON mismatch in the released paper version (`paper/main.pdf`,
v3 final).
