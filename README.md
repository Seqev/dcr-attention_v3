# DCR-attention

Top-K sparse attention for long-context decode on Llama-3.2-1B.

## Headline result (v3)

Multi-seed hero deployment point: **N = 32,000 tokens of context,
coverage floor c = 0.15**, with quality degradation
**ΔPPL = +0.428% ± 0.096 pp** (5 seeds: {0, 1, 2, 42, 100}; STRICT
classification on the deployment-tier scale, i.e. ΔPPL ≤ 0.5%).

Latency on the reference hardware (RTX 4060 Ti, bf16 inference) at the
hero operating point: **0.895× SDPA** — characterized as
*HeroQualityOnly*: quality multi-seed validated; speedup partial. The
named bottleneck (Pass-3 of the fused 3-pass kernel) and the
architectural response (ABKV, synthetic-data feasibility demonstrated)
are documented in §5 of the paper. End-to-end speedup recovery is
explicit future work.

Pre-registered matched-magnitude causal test of the underlying
entropy-concentration mechanism returns **DESCRIPTIVE** (Wilcoxon
p = 0.7645, Mann-Whitney U bias check p = 0.272 — generalizable). The
random-spectrum baseline (untrained K projections) gives α = 1.0000
± 0.00002 vs α_trained = 0.387, locating α < 1 as a fact about training
rather than softmax algebra.

Full paper: [`paper/main.pdf`](paper/main.pdf).

## What's here

| Directory | Contents |
|-----------|----------|
| `paper/` | v3 paper source (`main.tex`), compiled PDF (`main.pdf`), per-section `.tex` files, figures |
| `dcr_attention/` | M4 Triton fused top-K kernel + reference implementations + Llama integration |
| `tests/` | acceptance tests + integration tests + the six causal-pilot iteration scripts |
| `scripts/` | analysis scripts (Wilcoxon / Mann-Whitney / dose-response), figure generators, the Phase 5.5 paper-merge script |
| `data/raw/` | curated measurement JSONs cited by the paper (see `data/raw/README.md` for scope) |

## Status

- **v3 release (this repo):** multi-seed validated, pre-registered
  DESCRIPTIVE causal verdict, honest sub-parity systems characterization.
- v1.0 / v2.0 Zenodo DOIs: deleted (sober reset prior to v3 — the
  single-seed v2.0 numerical headline was reproduced exactly on seed 0
  in the multi-seed re-validation but sits at the high end of the
  distribution; the 5-seed mean places it ~30× lower, consistent with
  a tier-boundary measurement that benefits from multi-seed protocol).

## Reproducing measurements

Setup (CUDA-capable GPU recommended; CPU-only mode falls back to small
contexts):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # see project for exact pin set
# the project is a standalone package under dcr_attention/
```

Single-seed acceptance run (the cheapest reproducible measurement, ≈10
minutes on consumer GPU):

```bash
PYTHONPATH=. python tests/kernel/test_m1_acceptance.py
```

Full 5-seed protocol — Phase 2 acceptance configuration (N = 2000,
c = 0.5):

```bash
PYTHONPATH=. python tests/integration/test_phase_2_m1_multiseed.py
PYTHONPATH=. python tests/integration/test_phase_2_m4_multiseed.py
```

Hero re-run at N = 32K, c = 0.15 (≈4 hours per seed on consumer GPU):

```bash
PYTHONPATH=. python tests/integration/test_hero_verification_p1a.py
```

Theorem 3 causal test (≈2–3 GPU-hours for the full grid; the analysis
script consumes the per-cell JSONs):

```bash
PYTHONPATH=. python tests/integration/test_theorem3_causal_full.py
python scripts/theorem3_causal_analysis.py
```

Each script writes per-seed / per-cell JSONs to `data/raw/...` matching
the paths cited by the paper. Aggregated summaries are the artifacts
released here.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

The project memory commits to a *forward-only kernel* under Apache 2.0;
this is the canonical license, no internal subdirectory overrides.

## Citation

```bibtex
@article{dcr-attention-v3,
  title = {DCR-attention: top-K sparse attention for long-context decode
           on Llama-3.2-1B},
  year  = {2026},
  note  = {Zenodo DOI / arXiv ID to be added upon Phase 7.2 / 7.3 deposit}
}
```

(Zenodo DOI and arXiv ID will be added in subsequent release phases.)

## Reproducibility chain

Every quantitative claim in `paper/main.pdf` is traced to a JSON under
`data/raw/...` and verifiable by re-running the corresponding script
under `tests/` or `scripts/`. The paper's §4 Tables 9–13 cite per-seed
JSON paths in their captions; the Phase 6 audit verified zero
unresolved cross-references in the compiled paper.

## Scope

DCR-attention is one model (Llama-3.2-1B), one dataset (WikiText-2),
one task (next-token NLL), one hardware class (RTX 4060 Ti). Numerical
hero values may shift on other configurations; the multi-seed protocol
itself transfers. See paper §6.3 Limitations for the explicit scope
boundaries.
