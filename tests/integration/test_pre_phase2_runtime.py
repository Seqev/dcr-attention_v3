"""
Pre-Phase-2 P0a — M4 runtime sweep at c=0.10 (critical speedup measurement).

Bandwidth model predicts 1.4× speedup at N=32K, c=0.10. This script measures
actual latency to validate or refute that estimate.

Configs:
  R1: N=8K,  B=1, c=0.10  — single-batch baseline
  R2: N=8K,  B=4, c=0.10  — batch=4 (Phase 1 win regime)
  R3: N=32K, B=1, c=0.10  — hero scale, Phase 2 primary target

Output: /tmp/pre_phase2/p0a_runtime_c10.json

Run:
  cd /home/user/dcr-attention
  PYTHONPATH=/home/user/dcr-attention python tests/integration/test_pre_phase2_runtime.py \
    2>&1 | tee /tmp/pre_phase2/p0a_runtime.log
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

CONFIGS = [
    # (cfg_id, N,     c,    batch, label)
    ("R1",    8000,  0.10,  1,    "N8K_c10_b1"),
    ("R2",    8000,  0.10,  4,    "N8K_c10_b4"),
    ("R3",   32000,  0.10,  1,    "N32K_c10_b1"),
]

OUT_DIR = Path("/tmp/pre_phase2")

# Bandwidth-model prediction (Phase 1 estimate)
_PREDICTED_SPEEDUP_R3 = 1.4


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import load_model_and_data, time_trace

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"P0a — M4 Runtime Sweep at c=0.10", flush=True)
    print(f"GPU: {gpu}", flush=True)

    results = []

    for cfg_id, N, c, batch, label in CONFIGS:
        print(f"\n{'='*60}", flush=True)
        print(f"=== {cfg_id} {label}: N={N}, c={c}, B={batch} ===", flush=True)
        print(f"{'='*60}", flush=True)

        torch.cuda.empty_cache()
        model, tok, ids = load_model_and_data(N=N + 1, seed=0)

        # SDPA baseline
        cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
        torch.cuda.reset_peak_memory_stats()
        try:
            sdpa_lat = time_trace(model, cfg_sdpa, ids, N=N, batch_size=batch)
            sdpa_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            sdpa_status = "ok"
        except torch.cuda.OutOfMemoryError:
            sdpa_lat = sdpa_peak_gb = None
            sdpa_status = "OOM"
            print(f"  SDPA OOM at N={N} B={batch}", flush=True)

        torch.cuda.empty_cache()

        # M4 Triton at c=0.10
        cfg_m4 = DCRLlamaConfig(
            axis_source="q_topk_triton",
            coverage_floor=c,
            T_dispatch=64,
            k_window=64,
            enable_dcr=True,
            enable_adaptive_widening=False,
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            m4_lat = time_trace(model, cfg_m4, ids, N=N, batch_size=batch)
            m4_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            m4_status = "ok"
        except torch.cuda.OutOfMemoryError:
            m4_lat = m4_peak_gb = None
            m4_status = "OOM"
            print(f"  M4 OOM at N={N} B={batch}", flush=True)

        speedup = (sdpa_lat / m4_lat) if (sdpa_lat and m4_lat) else None
        vram_ratio = (sdpa_peak_gb / m4_peak_gb) if (sdpa_peak_gb and m4_peak_gb) else None

        sdpa_str = f"{sdpa_lat:.2f} ms ({sdpa_peak_gb:.2f} GB)" if sdpa_lat else sdpa_status
        m4_str  = f"{m4_lat:.2f} ms ({m4_peak_gb:.2f} GB)"  if m4_lat  else m4_status
        spd_str = f"{speedup:.3f}×" if speedup else "N/A"

        print(f"  SDPA:   {sdpa_str}", flush=True)
        print(f"  M4:     {m4_str}", flush=True)
        print(f"  Speedup:{spd_str}", flush=True)
        if cfg_id == "R3" and speedup:
            pred = _PREDICTED_SPEEDUP_R3
            print(
                f"  vs bw-model prediction {pred:.1f}×: "
                f"{'HOLDS' if speedup >= 1.2 else 'WRONG' if speedup < 1.0 else 'MARGINAL'}",
                flush=True,
            )

        result = {
            "config_id": cfg_id,
            "label": label,
            "N": N,
            "c_floor": c,
            "batch": batch,
            "sdpa_latency_ms": round(sdpa_lat, 3) if sdpa_lat else None,
            "sdpa_peak_vram_gb": round(sdpa_peak_gb, 3) if sdpa_peak_gb else None,
            "sdpa_status": sdpa_status,
            "m4_latency_ms": round(m4_lat, 3) if m4_lat else None,
            "m4_peak_vram_gb": round(m4_peak_gb, 3) if m4_peak_gb else None,
            "m4_status": m4_status,
            "speedup_vs_sdpa": round(speedup, 4) if speedup else None,
            "vram_reduction_factor": round(vram_ratio, 3) if vram_ratio else None,
            "bw_model_prediction": _PREDICTED_SPEEDUP_R3 if cfg_id == "R3" else None,
            "gpu": gpu,
        }
        results.append(result)

        del model, tok, ids
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"{'ID':>4}  {'Label':<18}  {'SDPA ms':>9}  {'M4 ms':>9}  {'Speedup':>8}  {'VRAM red.':>10}", flush=True)
    print(f"{'-'*70}", flush=True)
    for r in results:
        sdpa = f"{r['sdpa_latency_ms']:.1f}" if r["sdpa_latency_ms"] else "OOM"
        m4   = f"{r['m4_latency_ms']:.1f}"   if r["m4_latency_ms"]   else "OOM"
        spd  = f"{r['speedup_vs_sdpa']:.3f}×" if r["speedup_vs_sdpa"] else "N/A"
        vram = f"{r['vram_reduction_factor']:.2f}×" if r["vram_reduction_factor"] else "N/A"
        print(f"{r['config_id']:>4}  {r['label']:<18}  {sdpa:>9}  {m4:>9}  {spd:>8}  {vram:>10}", flush=True)

    out_path = OUT_DIR / "p0a_runtime_c10.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)

    # Decision logic output
    r3 = next((r for r in results if r["config_id"] == "R3"), None)
    if r3 and r3["speedup_vs_sdpa"]:
        spd = r3["speedup_vs_sdpa"]
        if spd >= 1.2:
            branch = "A  (bandwidth model holds — full GREEN Phase 2)"
        elif spd >= 1.0:
            branch = "C  (marginal speedup — narrow scope)"
        elif spd >= 0.8:
            branch = "B  (no speedup — memory+quality narrative)"
        else:
            branch = "B  (bandwidth model wrong — definite pivot)"
        print(f"\nP0a verdict → Branch {branch}", flush=True)
        print(f"R3 speedup: {spd:.3f}×  (predicted {_PREDICTED_SPEEDUP_R3:.1f}×)", flush=True)


if __name__ == "__main__":
    main()
