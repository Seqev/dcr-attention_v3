"""
M4 latency at c=0.15 and c=0.30 — hero deployment validation.

L1: N=8K  B=1 c=0.15
L2: N=8K  B=4 c=0.15
L3: N=32K B=1 c=0.15
L4: N=32K B=4 c=0.15  <-- DECIDING measurement
L5: N=32K B=1 c=0.30  -- Theorem 3 regime comparison

Runtime: ~30 minutes.

Run:
  cd /home/user/dcr-attention
  source /home/user/dcr-venv/bin/activate
  PYTHONPATH=/home/user/dcr-attention python tests/integration/test_hero_verification_latency.py \
    2>&1 | tee /tmp/hero_verification/latency_c015_c030.log
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

OUT_DIR = Path("/tmp/hero_verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("L1", 8_000,  0.15, 1),
    ("L2", 8_000,  0.15, 4),
    ("L3", 32_000, 0.15, 1),
    ("L4", 32_000, 0.15, 4),
    ("L5", 32_000, 0.30, 1),
]


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import load_model_and_data, time_trace

    _print("Hero Verification — M4 Latency at c=0.15 and c=0.30")
    _print(f"Output: {OUT_DIR}")
    _print("=" * 60)

    results = []

    for cfg_id, N, c, batch in CONFIGS:
        label = f"N{N // 1000}K_c{int(c * 100):02d}_b{batch}"
        _print(f"\n=== {cfg_id} {label}: N={N}, c={c}, B={batch} ===")

        _print(f"  Loading model…")
        model, tok, ids = load_model_and_data(N=N + 1, seed=0)
        gpu_name = torch.cuda.get_device_name(0)

        # SDPA baseline
        cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sdpa_lat = sdpa_peak = sdpa_status = None
        try:
            sdpa_lat = time_trace(model, cfg_sdpa, ids, N=N, batch_size=batch)
            sdpa_peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            sdpa_status = "ok"
            _print(f"  SDPA: {sdpa_lat:.2f} ms  peak={sdpa_peak:.3f} GB")
        except torch.cuda.OutOfMemoryError:
            sdpa_status = "OOM"
            _print(f"  SDPA: OOM")

        # M4 Triton
        cfg_m4 = DCRLlamaConfig(
            axis_source="q_topk_triton",
            coverage_floor=c,
            enable_dcr=True,
            enable_adaptive_widening=False,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m4_lat = m4_peak = m4_status = None
        try:
            m4_lat = time_trace(model, cfg_m4, ids, N=N, batch_size=batch)
            m4_peak = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            m4_status = "ok"
            _print(f"  M4:   {m4_lat:.2f} ms  peak={m4_peak:.3f} GB")
        except torch.cuda.OutOfMemoryError:
            m4_status = "OOM"
            _print(f"  M4: OOM")

        speedup = round(sdpa_lat / m4_lat, 4) if (sdpa_lat and m4_lat) else None
        if speedup:
            _print(f"  Speedup: {speedup:.3f}×")

        result = {
            "config_id": cfg_id,
            "label": label,
            "N": N,
            "c_floor": c,
            "batch": batch,
            "sdpa_latency_ms": round(sdpa_lat, 3) if sdpa_lat else None,
            "sdpa_peak_vram_gb": sdpa_peak,
            "sdpa_status": sdpa_status,
            "m4_latency_ms": round(m4_lat, 3) if m4_lat else None,
            "m4_peak_vram_gb": m4_peak,
            "m4_status": m4_status,
            "speedup_vs_sdpa": speedup,
            "gpu": gpu_name,
        }
        results.append(result)

        del model
        torch.cuda.empty_cache()

    out_path = OUT_DIR / "latency_c015_c030.json"
    out_path.write_text(json.dumps(results, indent=2))
    _print(f"\nSaved: {out_path}")

    # Summary table
    _print("\nSUMMARY TABLE:")
    _print(f"{'ID':<4} {'Label':<22} {'SDPA':>8} {'M4':>8} {'Speedup':>8}")
    _print("-" * 56)
    for r in results:
        sdpa_s = f"{r['sdpa_latency_ms']:.1f}ms" if r["sdpa_latency_ms"] else r["sdpa_status"]
        m4_s = f"{r['m4_latency_ms']:.1f}ms" if r["m4_latency_ms"] else r["m4_status"]
        sp_s = f"{r['speedup_vs_sdpa']:.3f}×" if r["speedup_vs_sdpa"] else "—"
        _print(f"{r['config_id']:<4} {r['label']:<22} {sdpa_s:>8} {m4_s:>8} {sp_s:>8}")
    _print("DONE.")


if __name__ == "__main__":
    main()
