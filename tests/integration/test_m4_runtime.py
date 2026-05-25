"""
M4 runtime sweep: end-to-end Llama-3.2-1B latency.

Compares SDPA baseline vs Triton DCR path at multiple (N, batch) configs.
Output: /tmp/m4_runtime/runtime_sweep.json

Acceptance criteria (§5.4):
  1. All 5 configs complete or graceful OOM report
  2. M4 Triton completes ALL configs without OOM (Track 3 closure)
  3. At N=20000 B=1: M4 latency < SDPA latency
  4. At N=5000 B=8: M4 peak VRAM < 5 GB
"""

from __future__ import annotations

import json
import pytest
import torch
from pathlib import Path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="M4 runtime sweep requires CUDA"
    ),
]

MODEL_ID = "meta-llama/Llama-3.2-1B"

CONFIGS = [
    # (N, batch, label)
    (5000,  1, "N5K_b1"),
    (5000,  4, "N5K_b4"),
    (5000,  8, "N5K_b8_PRODUCTION"),
    (8000,  1, "N8K_b1"),
    (20000, 1, "N20K_b1"),
]

T_DISPATCH = 64
COVERAGE_FLOOR = 0.5
K_WINDOW = 64


@pytest.mark.slow
def test_m4_runtime_sweep():
    """Sweep N × batch, measure SDPA vs Triton end-to-end latency per decode step."""
    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import load_model_and_data, time_trace

    out_dir = Path("/tmp/m4_runtime")
    out_dir.mkdir(exist_ok=True)

    results = []

    for N, batch, label in CONFIGS:
        print(f"\n--- {label}: N={N}, batch={batch} ---", flush=True)
        torch.cuda.empty_cache()

        model, _tok, ids = load_model_and_data(N=N)

        # SDPA baseline
        cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
        torch.cuda.reset_peak_memory_stats()
        try:
            sdpa_lat = time_trace(model, cfg_sdpa, ids, N=N, batch_size=batch)
            sdpa_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            sdpa_status = "ok"
        except torch.cuda.OutOfMemoryError:
            sdpa_lat = None
            sdpa_peak_gb = None
            sdpa_status = "OOM"
            print(f"  SDPA OOM at N={N} batch={batch}")

        torch.cuda.empty_cache()

        # M4 Triton
        cfg_m4 = DCRLlamaConfig(
            axis_source="q_topk_triton",
            k_window=K_WINDOW,
            coverage_floor=COVERAGE_FLOOR,
            T_dispatch=T_DISPATCH,
            enable_dcr=True,
            enable_adaptive_widening=False,
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            m4_lat = time_trace(model, cfg_m4, ids, N=N, batch_size=batch)
            m4_peak_gb = torch.cuda.max_memory_allocated() / 1e9
            m4_status = "ok"
        except torch.cuda.OutOfMemoryError:
            m4_lat = None
            m4_peak_gb = None
            m4_status = "OOM"
            print(f"  M4 Triton OOM at N={N} batch={batch}")

        speedup = (sdpa_lat / m4_lat) if (sdpa_lat and m4_lat) else None

        result = {
            "label": label,
            "N": N,
            "batch": batch,
            "sdpa_latency_ms_per_step": round(sdpa_lat, 3) if sdpa_lat else None,
            "sdpa_peak_vram_gb": round(sdpa_peak_gb, 3) if sdpa_peak_gb else None,
            "sdpa_status": sdpa_status,
            "m4_triton_latency_ms_per_step": round(m4_lat, 3) if m4_lat else None,
            "m4_peak_vram_gb": round(m4_peak_gb, 3) if m4_peak_gb else None,
            "m4_status": m4_status,
            "speedup_vs_sdpa": round(speedup, 3) if speedup else None,
        }
        results.append(result)

        sdpa_str = f"{sdpa_lat:.2f} ms ({sdpa_peak_gb:.2f} GB)" if sdpa_lat else "OOM"
        m4_str = f"{m4_lat:.2f} ms ({m4_peak_gb:.2f} GB)" if m4_lat else "OOM"
        spd_str = f"{speedup:.2f}×" if speedup else "N/A"
        print(f"  SDPA:   {sdpa_str}  [{sdpa_status}]")
        print(f"  Triton: {m4_str}  [{m4_status}]")
        print(f"  Speedup: {spd_str}")

        # Free model before next config (each config loads fresh)
        del model
        torch.cuda.empty_cache()

    with open(out_dir / "runtime_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'runtime_sweep.json'}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Config':<26} {'SDPA':<16} {'Triton':<16} {'Speedup':<10} {'VRAM'}")
    print("=" * 75)
    for r in results:
        sdpa = f"{r['sdpa_latency_ms_per_step']:.1f}ms" if r["sdpa_latency_ms_per_step"] else "OOM"
        triton = f"{r['m4_triton_latency_ms_per_step']:.1f}ms" if r["m4_triton_latency_ms_per_step"] else "OOM"
        speedup = f"{r['speedup_vs_sdpa']:.2f}×" if r["speedup_vs_sdpa"] else "N/A"
        vram = f"{r['m4_peak_vram_gb']:.2f}GB" if r["m4_peak_vram_gb"] else "—"
        print(f"{r['label']:<26} {sdpa:<16} {triton:<16} {speedup:<10} {vram}")

    # Acceptance gate §5.4 criterion 2: M4 Triton completes ALL configs
    m4_ooms = [r["label"] for r in results if r["m4_status"] == "OOM"]
    assert not m4_ooms, (
        f"M4 Triton OOM on: {m4_ooms}. "
        "A2 protocol: STOP — Track 3 closure failed. Surface to architect."
    )

    # Criterion 4: B=8 N=5K VRAM < 5 GB
    b8_result = next((r for r in results if r["label"] == "N5K_b8_PRODUCTION"), None)
    if b8_result and b8_result["m4_peak_vram_gb"] is not None:
        assert b8_result["m4_peak_vram_gb"] < 5.0, (
            f"M4 B=8 N=5K peak VRAM {b8_result['m4_peak_vram_gb']:.2f} GB exceeds 5 GB. "
            "Track 3 closure criterion §5.4 failed."
        )

    # Criterion 3: N=20K B=1 M4 < SDPA
    n20k_result = next((r for r in results if r["label"] == "N20K_b1"), None)
    if n20k_result and n20k_result["speedup_vs_sdpa"] is not None:
        assert n20k_result["speedup_vs_sdpa"] >= 1.0, (
            f"M4 Triton slower than SDPA at N=20K B=1: "
            f"speedup = {n20k_result['speedup_vs_sdpa']:.3f}×. "
            "Investigate kernel overhead."
        )
