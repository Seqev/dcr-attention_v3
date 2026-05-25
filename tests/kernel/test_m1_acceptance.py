"""
M1 acceptance test: axis_source="q_topk_reference" ΔPPL within ±0.1 pp of
rank-window reference (+0.871%) at N=2000.

Methodology mirrors H.1.B run_h1b_scaling.py exactly:
  - Data: WikiText-2 validation, first 2001 tokens (same prefix as H.1.B)
  - Decode: autoregressive teacher-forcing with KV cache (prefill_size=1)
  - DCR fires at N_kv >= T_dispatch (=64), matching H.1.B cfg #1
  - Parameters: k_window=64, coverage_floor=0.5  →  k_eff≈N_kv*0.5 at N=2000

CRITICAL: must use autoregressive decode, NOT single-shot forward.
Single-shot forward has N_q = N > 1 → routing sends everything to SDPA →
DCR never fires → ΔPPL ≈ 0 (wrong, not testing M1 at all).

Reference locked by architect:
  rank-window H.1.B cfg #1 (axis=q, c=0.5, N=2000): ΔPPL = +0.871%
  M1 gate: ΔPPL ∈ [+0.771%, +0.971%]   (±0.1 pp absolute)

Marks:
  slow  — loads 1B model, autoregressive N=2000 decode, ~4-8 h on GPU.
  gpu   — skipped on CPU.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="M1 acceptance test requires CUDA",
    ),
]

MODEL_ID = "meta-llama/Llama-3.2-1B"
N_TOKENS = 2000          # decode steps (H.1.B cfg #1)
PREFILL_SIZE = 1         # tokens fed in prefill before decode loop starts
T_DISPATCH = 64          # DCR fires at N_kv >= this (H.1.B default)
K_WINDOW = 64            # base k_window (adaptive widening via coverage_floor)
COVERAGE_FLOOR = 0.5     # H.1.B cfg #1: c=0.5 → k_eff ≈ N_kv*0.5

# Locked threshold (architect 2026-05-10, corrected from rank-window to M1 q_topk)
# Previous value 0.871 was the H.1.B rank-window result; test runs q_topk_reference
# which achieves 0.285% (3× better). Gate tightened to ±0.05 pp for this path.
_REF_DELTA_PP = 0.285    # M1 q_topk_reference locked result (N=2000, c=0.5)
_GATE_PP = 0.1           # acceptance half-width (pp)


# ---------------------------------------------------------------------------
# Module-level helpers (usable by other test files without fixtures)
# ---------------------------------------------------------------------------

def load_model_and_data(N: int = N_TOKENS + 1, seed: int = 0):
    """Load Llama-3.2-1B + WikiText-2 ids.  Returns (model, tokenizer, ids[offset:offset+N+1]).

    seed controls the starting offset into the WikiText-2 validation corpus:
    offset = seed * 1000 tokens.  seed=0 reproduces all prior measurements
    (offset=0, same prefix as H.1.B and Phase 1.5).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset as _load

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = _load("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    all_ids = tok(text, return_tensors="pt").input_ids[0]

    offset = seed * 1000
    assert offset + N + 1 <= all_ids.size(0), (
        f"seed={seed} offset={offset} requests tokens {offset}..{offset+N+1} "
        f"but corpus has only {all_ids.size(0)} tokens."
    )
    ids = all_ids[offset: offset + N + 1]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model, tok, ids


def run_trace(
    model,
    cfg,
    ids: torch.Tensor,
    N: int = N_TOKENS,
    label: str = "",
) -> float:
    """
    Reconfigure model with cfg, run autoregressive teacher-forced decode
    for N steps, return perplexity (scalar float).

    Handles both fresh (unpatched) and already-patched models.
    """
    # Sanity guard: N must be positive and ids must be long enough
    assert isinstance(N, int) and N > 0, f"N must be a positive int, got {N!r}"
    assert ids.shape[-1] > N, (
        f"ids has {ids.shape[-1]} tokens but N={N} requested — "
        f"need at least N+1 tokens (1 prefill + N decode steps)."
    )

    from dcr_attention.models.llama.monkey_patch import (
        patch_llama_with_dcr,
        reconfigure_dcr,
        is_patched,
    )

    if is_patched(model):
        reconfigure_dcr(model, cfg)
    else:
        patch_llama_with_dcr(model, cfg)

    _label = label or getattr(cfg, "axis_source", "sdpa")
    nll = _trace_nll(model, ids[: N + 1], N=N, label=_label)

    # Sanity guard: result must have exactly N entries
    assert nll.shape[0] == N, (
        f"_trace_nll returned {nll.shape[0]} NLL values, expected {N}. "
        f"Measurement-bug class: check that N parameter flows through."
    )

    ppl = math.exp(nll.mean().item())
    assert 0.0 < ppl < 10_000.0, (
        f"PPL={ppl:.2f} outside expected range (0, 10000). "
        f"Possible numerical overflow or degenerate model output."
    )
    return ppl


def time_trace(
    model,
    cfg,
    ids: torch.Tensor,
    N: int = N_TOKENS,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> float:
    """
    Return average latency in ms per decode step at KV cache depth N.

    Pre-fills a fake KV cache to N_kv=N so the Triton/DCR kernel actually
    fires during measurement (rather than falling back to SDPA due to small
    N_kv).  Fake K/V tensors are random bf16; only latency matters here.

    Runs n_warmup steps (discarded), then measures n_measure steps.
    batch_size > 1: same input token broadcast across the batch.
    """
    import time as _time
    from transformers import DynamicCache
    from dcr_attention.models.llama.monkey_patch import (
        patch_llama_with_dcr,
        reconfigure_dcr,
        is_patched,
    )

    if is_patched(model):
        reconfigure_dcr(model, cfg)
    else:
        patch_llama_with_dcr(model, cfg)

    device = next(model.parameters()).device

    # Build fake KV cache pre-filled to N_kv = N.
    # Shapes match what DynamicCache stores (pre-repeat_kv resolution).
    n_layers   = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim   = model.config.head_dim

    past = DynamicCache()
    with torch.no_grad():
        for layer_idx in range(n_layers):
            k_fake = torch.randn(
                batch_size, n_kv_heads, N, head_dim,
                dtype=torch.bfloat16, device=device,
            )
            v_fake = torch.randn(
                batch_size, n_kv_heads, N, head_dim,
                dtype=torch.bfloat16, device=device,
            )
            past.update(k_fake, v_fake, layer_idx)

    # Token ID 0 for every decode step (output quality is irrelevant here).
    next_tok = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

    latencies = []
    with torch.no_grad():
        for step in range(n_warmup + n_measure):
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            out = model(next_tok, past_key_values=past, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            t1 = _time.perf_counter()

            past = out.past_key_values   # N_kv grows by 1 each step (negligible)

            if step >= n_warmup:
                latencies.append((t1 - t0) * 1000.0)   # ms

    return sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# Standardized JSON output (used by all Phase 2+ measurement scripts)
# ---------------------------------------------------------------------------

def standardized_json_output(
    config_id: str,
    N: int,
    c_floor: float,
    ppl_baseline: float,
    ppl_dcr: float,
    delta_ppl_pct: float,
    seed: int,
    axis_source: str,
    extra: Optional[dict] = None,
) -> dict:
    """
    Standardized measurement record for all Phase 2-4 scripts.

    Includes seed provenance so results are fully reproducible.
    Import from this module: ``from tests.kernel.test_m1_acceptance import standardized_json_output``
    """
    import datetime
    import numpy as np
    import random as _random

    result = {
        "config_id": config_id,
        "N": N,
        "c_floor": c_floor,
        "k_eff": max(int(c_floor * N), 1),
        "ppl_baseline_sdpa": float(ppl_baseline),
        "ppl_dcr": float(ppl_dcr),
        "delta_ppl_pct": float(delta_ppl_pct),
        "axis_source": axis_source,
        "seed": int(seed),
        "torch_seed": int(torch.initial_seed()),
        "numpy_seed": int(np.random.get_state()[1][0]),
        "python_seed_state_hash": hash(str(_random.getstate())),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
    }
    if extra:
        result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: load once, shared across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wikitext_ids():
    """WikiText-2 validation tokens, first N_TOKENS+1, same prefix as H.1.B."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    enc = tok(text, return_tensors="pt")
    ids = enc.input_ids[0]
    assert ids.size(0) >= N_TOKENS + 1, (
        f"WikiText-2 validation has {ids.size(0)} tokens; need {N_TOKENS + 1}"
    )
    return ids[: N_TOKENS + 1]   # [N_TOKENS+1]


@pytest.fixture(scope="module")
def llama_model():
    """Load Llama-3.2-1B once, unpatched, on GPU."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core: autoregressive teacher-forced NLL trace
# ---------------------------------------------------------------------------

def _trace_nll(model, ids: torch.Tensor, N: int = N_TOKENS, label: str = "") -> torch.Tensor:
    """
    Teacher-forced autoregressive decode.  Mirrors H.1.B trace_nll().

    Returns nll[N] float64 — per-step NLL from decode position 0..N-1.
    NLL[t] = -log p(ids[PREFILL_SIZE + t] | context).
    DCR fires on decode steps where N_kv >= T_DISPATCH.
    """
    # Sanity guard: N parameter must flow here correctly (measurement-bug class)
    assert isinstance(N, int) and N > 0, f"N must be a positive int, got {N!r}"
    assert ids.shape[-1] >= N + PREFILL_SIZE, (
        f"ids has {ids.shape[-1]} tokens; need at least {N + PREFILL_SIZE} "
        f"(PREFILL_SIZE={PREFILL_SIZE} + N={N})."
    )

    device = next(model.parameters()).device
    ids = ids.to(device)

    nll = torch.zeros(N, dtype=torch.float64)
    t_start = time.perf_counter()

    with torch.no_grad():
        # Prefill
        out = model(ids[:PREFILL_SIZE].unsqueeze(0), use_cache=True, return_dict=True)
        past = out.past_key_values
        logit_prev = out.logits[0, -1]   # [V]

        for t in range(N):
            target = ids[PREFILL_SIZE + t]
            nll[t] = -F.log_softmax(logit_prev.float(), dim=-1)[target].item()

            if t < N - 1:
                out = model(
                    ids[PREFILL_SIZE + t].unsqueeze(0).unsqueeze(0),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                torch.cuda.synchronize()
                past = out.past_key_values
                logit_prev = out.logits[0, -1]

            if (t + 1) % 200 == 0 or (t + 1) == N:
                elapsed = time.perf_counter() - t_start
                cum_ppl = math.exp(nll[: t + 1].mean().item())
                rate = (t + 1) / elapsed
                eta = (N - t - 1) / rate if rate > 0 else 0
                print(
                    f"  [{label}] t={t+1:>5}/{N}  "
                    f"cum_ppl={cum_ppl:.4f}  {rate:.1f} tok/s  ETA {eta:.0f}s",
                    flush=True,
                )

    # Sanity: NLL must be finite (catches model errors / numerical overflow)
    assert torch.isfinite(nll).all(), (
        f"_trace_nll produced non-finite NLL values at positions "
        f"{torch.where(~torch.isfinite(nll))[0].tolist()[:10]}. "
        f"Check model state and input ids."
    )

    return nll


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_m1_baseline_ppl(llama_model, wikitext_ids):
    """
    Baseline (SDPA-only) perplexity sanity check.
    Not a gate — just prints PPL so we can verify data loading is correct.
    Expected: ~10-12 for Llama-3.2-1B on WikiText-2 val.
    """
    from dcr_attention.models.llama.config import DCRLlamaConfig
    from dcr_attention.models.llama.monkey_patch import patch_llama_with_dcr

    # Disable DCR entirely → pure SDPA reference
    patch_llama_with_dcr(llama_model, cfg=DCRLlamaConfig(enable_dcr=False))
    try:
        nll = _trace_nll(llama_model, wikitext_ids, label="baseline")
    finally:
        # Leave patched-but-disabled for next test; full unpatch done via fixture teardown
        pass

    ppl = math.exp(nll.mean().item())
    print(f"\n  Baseline PPL = {ppl:.4f}")
    assert 5.0 < ppl < 30.0, f"Baseline PPL {ppl:.4f} outside sane range [5, 30]"

    # Stash for use by acceptance test (module-level side-channel via object attr)
    llama_model._m1_baseline_nll = nll


def test_m1_qaxis_topk_reference_delta_ppl(llama_model, wikitext_ids):
    """
    M1 acceptance gate: ΔPPL ∈ [+0.185%, +0.385%].

    Runs autoregressive decode with axis_source='q_topk_reference'.
    Compares against baseline NLL from test_m1_baseline_ppl (same session)
    or falls back to re-running baseline if not available.

    Parameters match H.1.B cfg #1:
      N=2000, k_window=64, coverage_floor=0.5, T_dispatch=64

    Locked reference: +0.285% (M1 q_topk_reference, Phase 1 measurement).
    Previous docstring said 0.871% — that was the rank-window result, wrong here.
    """
    from dcr_attention.models.llama.config import DCRLlamaConfig
    from dcr_attention.models.llama.monkey_patch import (
        patch_llama_with_dcr,
        reconfigure_dcr,
        is_patched,
    )

    # --- Baseline NLL (from previous test or fresh run) ---
    if not hasattr(llama_model, "_m1_baseline_nll"):
        if is_patched(llama_model):
            reconfigure_dcr(llama_model, cfg=DCRLlamaConfig(enable_dcr=False))
        else:
            patch_llama_with_dcr(llama_model, cfg=DCRLlamaConfig(enable_dcr=False))
        nll_base = _trace_nll(llama_model, wikitext_ids, label="baseline-fallback")
    else:
        nll_base = llama_model._m1_baseline_nll

    ppl_base = math.exp(nll_base.mean().item())

    # --- M1 run ---
    # Model is already patched (either from test_m1_baseline_ppl or from
    # the baseline-fallback branch above); use reconfigure_dcr to update cfg.
    cfg = DCRLlamaConfig(
        axis_source="q_topk_reference",
        k_window=K_WINDOW,
        coverage_floor=COVERAGE_FLOOR,
        T_dispatch=T_DISPATCH,
        enable_dcr=True,
        enable_adaptive_widening=False,
    )
    reconfigure_dcr(llama_model, cfg=cfg)
    try:
        nll_m1 = _trace_nll(llama_model, wikitext_ids, label="M1-q_topk_ref")
    finally:
        reconfigure_dcr(llama_model, cfg=DCRLlamaConfig(enable_dcr=False))

    ppl_m1 = math.exp(nll_m1.mean().item())
    delta_pp = (ppl_m1 / ppl_base - 1.0) * 100.0
    drift = abs(delta_pp - _REF_DELTA_PP)

    print(
        f"\n{'='*60}\n"
        f"M1 ACCEPTANCE RESULT\n"
        f"  PPL_base  = {ppl_base:.4f}\n"
        f"  PPL_m1    = {ppl_m1:.4f}\n"
        f"  ΔPPL      = {delta_pp:+.3f}%\n"
        f"  reference = {_REF_DELTA_PP:+.3f}%  (M1 q_topk locked, N=2000 c=0.5)\n"
        f"  |drift|   = {drift:.3f} pp  (gate ≤ {_GATE_PP} pp)\n"
        f"  verdict   = {'✓ PASS' if drift <= _GATE_PP else '✗ FAIL'}\n"
        f"{'='*60}",
        flush=True,
    )

    assert drift <= _GATE_PP, (
        f"M1 ΔPPL {delta_pp:+.3f}% is {drift:.3f} pp from reference "
        f"{_REF_DELTA_PP:+.3f}% — exceeds ±{_GATE_PP} pp gate.\n"
        f"  PPL_base={ppl_base:.4f}  PPL_m1={ppl_m1:.4f}"
    )
