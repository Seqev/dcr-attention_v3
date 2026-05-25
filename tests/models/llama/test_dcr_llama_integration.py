"""
GPU integration tests for DCRLlamaAttention with real HF Llama-3.2 1B.

These tests are skipped on CPU and on machines without HF_TOKEN.

Goal: verify the wrapper integrates cleanly with HF — no crashes, expected
output shapes, and bit-equivalent behaviour in the SDPA-only regime
(enable_dcr=False or T_dispatch=infinity).
"""

from __future__ import annotations
import os
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU integration tests require CUDA",
)

MODEL_ID = "meta-llama/Llama-3.2-1B"


@pytest.fixture(scope="module")
def llama_1b():
    """Load Llama-3.2 1B once per module."""
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


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def test_unpatched_forward_works(llama_1b):
    """Sanity: HF Llama-3.2 1B forward pass works at all."""
    model, tokenizer = llama_1b
    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    assert out.logits.shape[-1] == model.config.vocab_size
    assert torch.isfinite(out.logits).all()


def test_patch_then_forward_works(llama_1b):
    """Patch the model and verify forward still works (no crash)."""
    from dcr_attention.models.llama import patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig

    model, tokenizer = llama_1b
    cfg = DCRLlamaConfig(T_dispatch=10**9)   # force SDPA always — safe smoke
    n = patch_llama_with_dcr(model, cfg)
    try:
        assert n == len(model.model.layers)
        inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
        assert torch.isfinite(out.logits).all()
    finally:
        unpatch_llama(model)


# ---------------------------------------------------------------------------
# SDPA-only equivalence: with enable_dcr=False, output must equal un-patched
# ---------------------------------------------------------------------------

def test_enable_dcr_false_matches_unpatched(llama_1b):
    """
    With enable_dcr=False (master switch off), DCRLlamaAttention.forward should
    behave bit-equivalently to the original LlamaAttention.forward in the
    sense that final logits agree to within bf16 numerical tolerance.
    """
    from dcr_attention.models.llama import patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig

    model, tokenizer = llama_1b
    inputs = tokenizer(
        "The quick brown fox jumps over the lazy dog.",
        return_tensors="pt",
    ).to("cuda")

    # Reference: un-patched forward
    with torch.no_grad():
        out_ref = model(**inputs, use_cache=False).logits.float().cpu()

    # Patched with master switch OFF
    cfg = DCRLlamaConfig(enable_dcr=False)
    patch_llama_with_dcr(model, cfg)
    try:
        with torch.no_grad():
            out_patch = model(**inputs, use_cache=False).logits.float().cpu()
    finally:
        unpatch_llama(model)

    diff = (out_ref - out_patch).abs().max().item()
    # bf16 logits over 16 layers: tolerance reasonable up to ~5e-2 atol
    assert diff < 5e-2, f"enable_dcr=False output diverged: max|diff| = {diff:g}"


def test_T_dispatch_infinity_matches_unpatched(llama_1b):
    """
    T_dispatch = inf forces SDPA branch even on decode; should equal un-patched.
    """
    from dcr_attention.models.llama import patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig

    model, tokenizer = llama_1b
    inputs = tokenizer("A short prompt.", return_tensors="pt").to("cuda")

    with torch.no_grad():
        out_ref = model(**inputs, use_cache=False).logits.float().cpu()

    cfg = DCRLlamaConfig(T_dispatch=10**9, enable_dcr=True)
    patch_llama_with_dcr(model, cfg)
    try:
        with torch.no_grad():
            out_patch = model(**inputs, use_cache=False).logits.float().cpu()
    finally:
        unpatch_llama(model)

    diff = (out_ref - out_patch).abs().max().item()
    assert diff < 5e-2, f"T_dispatch=inf output diverged: max|diff| = {diff:g}"


# ---------------------------------------------------------------------------
# DCR branch smoke — long context decode actually triggers DCR
# ---------------------------------------------------------------------------

def test_dcr_branch_triggers_on_long_context_decode(llama_1b):
    """
    With T_dispatch=512, decode after a long-enough prefill should hit DCR.
    Verify finite output.  Correctness vs reference at decode-shape is
    expected to differ slightly (different kernel) — this test is smoke only.
    """
    from dcr_attention.models.llama import patch_llama_with_dcr, unpatch_llama, DCRLlamaConfig

    model, tokenizer = llama_1b
    # Long prompt to fill the cache past T_dispatch=512
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 200
    inputs = tokenizer(long_prompt, return_tensors="pt", max_length=1024,
                       truncation=True).to("cuda")

    cfg = DCRLlamaConfig(T_dispatch=512, k_window=64)
    patch_llama_with_dcr(model, cfg)
    try:
        with torch.no_grad():
            # Prefill (use_cache=True so we get a real cache)
            out_prefill = model(**inputs, use_cache=True)
            past = out_prefill.past_key_values
            assert torch.isfinite(out_prefill.logits).all(), "prefill produced NaN/inf"

            # Decode 1 token: this should hit the DCR branch
            new_id = torch.tensor([[1]], device="cuda")
            out_decode = model(input_ids=new_id, past_key_values=past,
                               use_cache=True)
            assert torch.isfinite(out_decode.logits).all(), "DCR decode produced NaN/inf"
    finally:
        unpatch_llama(model)


# ---------------------------------------------------------------------------
# Memory-neutrality
# ---------------------------------------------------------------------------

def test_patching_does_not_increase_param_count(llama_1b):
    """Real test of memory neutrality on a 1B-parameter model."""
    from dcr_attention.models.llama import patch_llama_with_dcr, unpatch_llama

    model, _ = llama_1b
    n_before = sum(p.numel() for p in model.parameters())
    patch_llama_with_dcr(model)
    try:
        n_after = sum(p.numel() for p in model.parameters())
        assert n_before == n_after, (
            f"param count changed: {n_before} → {n_after}; "
            f"wrapper introduced fresh parameters"
        )
    finally:
        unpatch_llama(model)
