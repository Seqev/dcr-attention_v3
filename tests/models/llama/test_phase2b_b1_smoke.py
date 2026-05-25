"""
Phase 2b — B1 smoke gate (GPU-only).

Validates that DCRLlamaAttention's DCR branch actually triggers in real
HF Llama-3.2 1B inference, produces finite output, and matches expected
shapes.  Cheap (sec) but CRITICAL: catches silent SDPA-fallback bugs
that won't show in unit tests but break Phase 2c/2d.

Acceptance:
  * test_dcr_branch_actually_invoked_when_routing_says_dcr — counter > 0
  * test_dcr_decode_produces_finite_output — torch.isfinite(.).all()
  * test_dcr_decode_no_shape_regression — output shape matches unpatched

Skipped on CPU and machines without HF gated repo access.
"""
from __future__ import annotations
import os
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 2b smoke tests require CUDA",
)

MODEL_ID = "meta-llama/Llama-3.2-1B"

# Use synthetic random tokens for B1 — valid vocab range, deterministic seed.
# Real-text prompts are reserved for B2 (perplexity) and B3 (long-context).
VOCAB_SIZE = 128256  # Llama-3.2 vocab size
SEED = 0


@pytest.fixture(scope="module")
def llama_1b():
    """Load Llama-3.2 1B once per module (~2 sec from cache)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def _make_random_input_ids(n_tokens: int, device: str = "cuda") -> torch.Tensor:
    """Reproducible random input_ids of given length."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    ids = torch.randint(0, VOCAB_SIZE, (1, n_tokens), generator=g, dtype=torch.long)
    return ids.to(device)


# ---------------------------------------------------------------------------
# B1.1 — DCR branch actually invoked
# ---------------------------------------------------------------------------

def test_dcr_branch_actually_invoked_when_routing_says_dcr(llama_1b):
    """
    With T_dispatch < N_kv during decode, every layer should hit DCR.
    Counter must be > 0.  Catches: silent SDPA fallback bugs.

    Setup:
      - N_ctx = 2500 tokens (prefill)
      - T_dispatch = 2048
      - decode 1 step (N_q=1, N_kv=2501 ≥ T_dispatch)
      - 16 layers × 1 decode step = 16 expected DCR invocations
    """
    from dcr_attention.models.llama import (
        patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig,
    )
    from dcr_attention.models.llama.attention import DCRLlamaAttention

    model, _ = llama_1b
    n_ctx = 2500
    input_ids = _make_random_input_ids(n_ctx)

    cfg = DCRLlamaConfig(T_dispatch=2048, k_window=64, enable_dcr=True)
    patch_llama_with_dcr(model, cfg)
    try:
        DCRLlamaAttention.reset_counters()
        with torch.no_grad():
            # Prefill — should be SDPA (N_q > 1)
            out_pre = model(input_ids=input_ids, use_cache=True)
            past = out_pre.past_key_values

            counters_after_prefill = DCRLlamaAttention.get_counters()

            # Decode 1 token — N_kv = 2501 ≥ T_dispatch, must be DCR
            new_id = torch.tensor([[1]], device="cuda")
            model(input_ids=new_id, past_key_values=past, use_cache=True)

            counters_after_decode = DCRLlamaAttention.get_counters()
    finally:
        unpatch_llama(model)

    n_layers = 16  # Llama-3.2 1B
    # Prefill: 16 layers × 1 prefill = 16 SDPA invocations expected.
    assert counters_after_prefill["sdpa"] == n_layers, (
        f"Prefill should hit SDPA on every layer; got {counters_after_prefill}"
    )
    assert counters_after_prefill["dcr"] == 0, (
        f"Prefill should NOT hit DCR (N_q > 1); got {counters_after_prefill}"
    )
    # Decode: +16 DCR invocations (one per layer).
    delta_dcr = counters_after_decode["dcr"] - counters_after_prefill["dcr"]
    assert delta_dcr == n_layers, (
        f"Decode with N_kv >= T_dispatch must invoke DCR on every layer; "
        f"got delta_dcr={delta_dcr}, expected {n_layers}. "
        f"Counters: pre={counters_after_prefill}, post={counters_after_decode}. "
        f"Possible silent fallback to SDPA — check routing logic."
    )


# ---------------------------------------------------------------------------
# B1.2 — DCR decode produces finite output
# ---------------------------------------------------------------------------

def test_dcr_decode_produces_finite_output(llama_1b):
    """
    Decode 32 tokens through DCR branch.  All logits must be finite
    (no NaN/inf).  Catches: numerical instabilities in DCR kernel
    integration with Llama RoPE+GQA at scale.
    """
    from dcr_attention.models.llama import (
        patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig,
    )
    from dcr_attention.models.llama.attention import DCRLlamaAttention

    model, _ = llama_1b
    n_ctx = 2500
    n_decode = 32

    cfg = DCRLlamaConfig(T_dispatch=2048, k_window=64, enable_dcr=True)
    patch_llama_with_dcr(model, cfg)
    try:
        DCRLlamaAttention.reset_counters()
        with torch.no_grad():
            input_ids = _make_random_input_ids(n_ctx)
            out = model(input_ids=input_ids, use_cache=True)
            assert torch.isfinite(out.logits).all(), "prefill logits not finite"
            past = out.past_key_values

            for step in range(n_decode):
                new_id = torch.tensor([[step % VOCAB_SIZE]], device="cuda")
                out = model(input_ids=new_id, past_key_values=past, use_cache=True)
                past = out.past_key_values
                assert torch.isfinite(out.logits).all(), (
                    f"Decode step {step}: logits not finite "
                    f"(NaN={torch.isnan(out.logits).any().item()}, "
                    f"inf={torch.isinf(out.logits).any().item()})"
                )

        counters = DCRLlamaAttention.get_counters()
    finally:
        unpatch_llama(model)

    # Sanity: DCR was actually used for these decode steps.
    assert counters["dcr"] >= 16 * n_decode, (
        f"Expected >= {16 * n_decode} DCR invocations across {n_decode} "
        f"decode steps; got {counters['dcr']}"
    )


# ---------------------------------------------------------------------------
# B1.3 — DCR decode shape matches unpatched
# ---------------------------------------------------------------------------

def test_dcr_decode_no_shape_regression(llama_1b):
    """
    Patched output shape must match unpatched output shape on every step.
    Catches: shape bugs that would break downstream tokenizer/sampling.
    """
    from dcr_attention.models.llama import (
        patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig,
    )

    model, _ = llama_1b
    n_ctx = 2500
    input_ids = _make_random_input_ids(n_ctx)

    # Reference: unpatched shapes
    with torch.no_grad():
        out_ref_pre = model(input_ids=input_ids, use_cache=True)
        ref_pre_shape = out_ref_pre.logits.shape
        new_id = torch.tensor([[1]], device="cuda")
        out_ref_dec = model(
            input_ids=new_id,
            past_key_values=out_ref_pre.past_key_values,
            use_cache=True,
        )
        ref_dec_shape = out_ref_dec.logits.shape

    # Patched: same input/decode, compare shapes
    cfg = DCRLlamaConfig(T_dispatch=2048, k_window=64, enable_dcr=True)
    patch_llama_with_dcr(model, cfg)
    try:
        with torch.no_grad():
            out_pat_pre = model(input_ids=input_ids, use_cache=True)
            assert out_pat_pre.logits.shape == ref_pre_shape, (
                f"Prefill shape regression: ref={ref_pre_shape}, "
                f"patched={out_pat_pre.logits.shape}"
            )
            out_pat_dec = model(
                input_ids=new_id,
                past_key_values=out_pat_pre.past_key_values,
                use_cache=True,
            )
            assert out_pat_dec.logits.shape == ref_dec_shape, (
                f"Decode shape regression: ref={ref_dec_shape}, "
                f"patched={out_pat_dec.logits.shape}"
            )
    finally:
        unpatch_llama(model)
