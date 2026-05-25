"""Phase 6 Pass 1 — dump canonical numerical claims from raw artifacts."""
import json
from pathlib import Path

P = lambda p: json.load(open(p))
R = Path("/home/user/dcr-attention")

m1 = P(R / "data/raw/phase_2_repro/m1_acceptance/m1_5seed_summary.json")
m4 = P(R / "data/raw/phase_2_repro/m4_acceptance/m4_5seed_summary.json")
drift = P(R / "data/raw/phase_2_repro/m4_acceptance/m4_m1_drift.json")
v2h = P(R / "data/raw/phase_2_repro/v2_hero/v2_hero_5seed_summary.json")
hero = P(R / "data/raw/hero_verification/p1a_5seed_summary.json")
q7 = P(R / "data/raw/pre_phase2/p0b_q7_stability.json")
lat = P(R / "data/raw/hero_verification/latency_c015_c030.json")
abkv = P(R / "data/raw/abkv_day1/bin_size_sweep.json")
causal = P(R / "data/raw/phase_3_causal_full/analysis.json")
rand = P(R / "data/raw/phase_3_causal_full/random_spectrum_results.json")

print("CANONICAL VALUES FROM JSON:\n")
print("=== Phase 2 acceptance ===")
ms = m1["summary"]
print(f"  M1 mean: {ms['mean_delta_pct']:.6f}%   std: {ms['std_delta_pct']:.6f}")
print(f"  M1 per-seed: {ms['per_seed']}")
print(f"  M1 classification: {ms['classification']}")
ms4 = m4["summary"]
print(f"  M4 mean: {ms4['mean_delta_pct']:.6f}%   std: {ms4['std_delta_pct']:.6f}")
print(f"  M4 per-seed: {ms4['per_seed']}")
print(f"  M4 classification: {ms4['classification']}")
print(f"  Drift M4-M1: {drift['drift_m4_minus_m1_pp']:.6f} pp   verdict: {drift['verdict']}")

print("\n=== v2.0 hero re-val (N=20K, c=0.5) ===")
print(f"  mean: {v2h['mean_delta_pct']:.6f}%   std: {v2h['std_delta_pct']:.6f}")
print(f"  classification: {v2h['classification']}   drift_direction: {v2h['drift_direction']}")
print(f"  drift_from_published (+0.308%): {v2h['drift_from_published']:.6f} pp")
print(f"  per-seed deltas: {v2h['deltas_ppl_pct']}")
print(f"  v2_0_published reference: {v2h['v2_0_published']}")

print("\n=== HERO (N=32K, c=0.15) ===")
hs = hero["summary"]
print(f"  mean: {hs['mean_delta_pct']:.6f}%   std: {hs['std_delta_pct']:.6f}")
print(f"  classification: {hs['classification']}   verdict: {hs['verdict']}")
print(f"  deltas: {hs['deltas_ppl_pct']}")
print(f"  single_seed_ref: {hs['single_seed_ref_pct']}   drift: {hs['drift_from_single_seed_pp']:.6f}")

print("\n=== Q7 (N=32K, c=0.10) ===")
print(f"  mean: {q7['mean_delta_pct']:.6f}%   std: {q7['std_delta_pct']:.6f}   drift: {q7['drift_from_phase15_ref']:.6f}")
print(f"  deployability: {q7['deployability']}   single_seed_ref: {q7['phase_1_5_single_seed_ref']}")
print(f"  per-seed deltas:", [r["delta_ppl_pct"] for r in q7["per_seed"]])
print(f"  k_eff at c=0.10: {q7['per_seed'][0]['k_eff']}")

print("\n=== Latency L1-L5 ===")
for r in lat:
    print(f"  {r['config_id']}: N={r['N']:>5} B={r['batch']} c={r['c_floor']:.2f}  SDPA={r['sdpa_latency_ms']:>7.2f}ms  M4={r['m4_latency_ms']:>7.2f}ms  speedup={r['speedup_vs_sdpa']:.4f}")

print("\n=== Causal run analysis (registered λ=0.15) ===")
pp = causal["primary_pooled"]
bp = causal["bias_check_pooled"]
dp = causal["dose_response_pooled"]
ap = causal["alpha_fit_descriptive"]
print(f"  PRIMARY: n={pp['n']}  median(D)={pp['median']:.4e}  CI95=[{pp['median_ci95_lo']:.4e}, {pp['median_ci95_hi']:.4e}]")
print(f"           Wilcoxon p={pp['p_value']:.6f}   verdict={pp['verdict']}")
print(f"  BIAS: n_primary={bp['n_primary']}  n_secondary={bp['n_secondary']}")
print(f"        median_p={bp['median_primary']:.4e}  median_s={bp['median_secondary']:.4e}")
print(f"        Mann-Whitney U p={bp['p_value']:.6f}   verdict={bp['verdict']}")
print(f"  DOSE: slope dD/dλ = {dp['slope_dD_dlambda']:.4e}")
for s in dp["sweep"]:
    print(f"    λ={s['lambda']:.2f}: n={s['n']} median={s['median_D']:+.4e} CI95=[{s['ci95_lo']:+.4e}, {s['ci95_hi']:+.4e}]")
print(f"  ALPHA_TRAINED (descriptive fit): α={ap['alpha']:.6f}   β={ap['beta']:.6f}")
print(f"    contexts: {ap['contexts']}   mean_base_H: {[round(x, 4) for x in ap['mean_base_H']]}")

print("\n=== Random spectrum ===")
print(f"  α_random per seed: {rand['alpha_random_seeds']}")
print(f"  α_random mean ± std: {rand['alpha_random_mean']:.6f} ± {rand['alpha_random_std']:.6e}")
print(f"  config: {rand['config']}")

print("\n=== ABKV Day-1 ===")
print(f"  config: N={abkv['config']['N']} B={abkv['config']['B']} H={abkv['config']['H']} H_kv={abkv['config']['H_kv']}")
print(f"  important_frac={abkv['config']['important_frac']}  amplitude={abkv['config']['amplitude']}")
print(f"  num_bins sweep: {[r['num_bins'] for r in abkv['results']]}")
print("  Per (nb, c): cos_sim, attn_ms")
for r in abkv["results"]:
    for cov in r["coverages"]:
        print(f"    nb={r['num_bins']:>3} c={cov['c']:.1f}  cos={cov['cos_sim']:.6f}  attn_ms={cov['attn_ms']:.2f}")
