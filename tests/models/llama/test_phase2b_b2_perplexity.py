"""
Phase 2b Round 2 — perplexity logic unit tests.

Smoke-tests `compute_window_singleshot` and `compute_window_autoregressive`
on a minimal mock LM that returns deterministic logits.  Verifies:

  1. Both methods compute identical NLL on the same tokens (since the LM
     is deterministic and teacher-forced).
  2. Counters increment as expected.
  3. NLL formula correct: -sum(log p(x_t | x_<t)).

These tests run on CPU, sub-second, no Llama load required.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.phase2b.perplexity import (
    compute_window_singleshot,
    compute_window_autoregressive,
    aggregate,
)


class TinyDeterministicLM(nn.Module):
    """A minimal LM whose logits are a fixed function of input tokens.

    We make logits depend on input position only (not content) so that
    single-shot and autoregressive forward give identical logits per
    position.  This isolates testing the NLL accumulation logic.
    """
    def __init__(self, vocab_size: int = 16, hidden: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        # Position-only logit head: logit[v] depends on position p only.
        # Use a single linear over a positional encoding.
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kw):
        # Compute embeddings, simple identity logits (sum of embeddings as score)
        x = self.embed(input_ids)                         # [B, N, H]
        logits = self.head(x)                              # [B, N, V]

        class Out:
            pass
        out = Out()
        out.logits = logits
        # Fake "cache": just track total tokens seen
        if use_cache:
            class FakeCache:
                def __init__(self, n=0):
                    self.n = n
            if past_key_values is None:
                out.past_key_values = FakeCache(input_ids.size(1))
            else:
                out.past_key_values = FakeCache(past_key_values.n + input_ids.size(1))
        return out


def test_singleshot_returns_finite_nll():
    """Basic: NLL should be finite on a small window."""
    torch.manual_seed(0)
    model = TinyDeterministicLM(vocab_size=16)
    model.eval()
    window = torch.randint(0, 16, (8,))
    r = compute_window_singleshot(model, window)
    assert math.isfinite(r.nll_sum)
    assert r.n_predicted == 7  # window length minus first token
    assert r.ppl > 0


def test_autoregressive_returns_finite_nll():
    """Basic: autoregressive NLL also finite."""
    torch.manual_seed(0)
    model = TinyDeterministicLM(vocab_size=16)
    model.eval()
    window = torch.randint(0, 16, (8,))
    r = compute_window_autoregressive(model, window, prefill_size=1)
    assert math.isfinite(r.nll_sum)
    assert r.n_predicted == 7  # same as single-shot
    assert r.ppl > 0


def test_singleshot_and_autoregressive_match():
    """
    With a deterministic LM where logit[t] depends only on input_ids[t]
    (not on history), both methods must yield identical NLL.

    This protects against off-by-one errors in either implementation.
    """
    torch.manual_seed(0)
    model = TinyDeterministicLM(vocab_size=16)
    model.eval()
    window = torch.randint(0, 16, (16,))

    r_ss = compute_window_singleshot(model, window)
    r_ar = compute_window_autoregressive(model, window, prefill_size=1)

    assert r_ss.n_predicted == r_ar.n_predicted, (
        f"Methods predict different number of positions: "
        f"singleshot={r_ss.n_predicted}, autoregressive={r_ar.n_predicted}"
    )
    delta_per_token = abs(r_ss.nll_sum - r_ar.nll_sum) / r_ss.n_predicted
    assert delta_per_token < 1e-5, (
        f"Methods disagree on NLL: ss={r_ss.nll_sum:.6f}, "
        f"ar={r_ar.nll_sum:.6f}, |Δ/tok|={delta_per_token:.2e}"
    )


def test_aggregate_sums_correctly():
    """aggregate() across windows = sum of NLLs / sum of counts."""
    torch.manual_seed(0)
    model = TinyDeterministicLM(vocab_size=16)
    model.eval()
    windows = [torch.randint(0, 16, (4,)) for _ in range(3)]
    results = [compute_window_singleshot(model, w) for w in windows]

    agg = aggregate(results)
    expected_nll = sum(r.nll_sum for r in results)
    expected_n = sum(r.n_predicted for r in results)
    assert agg["nll_sum"] == expected_nll
    assert agg["n_predicted"] == expected_n
    assert math.isclose(agg["ppl"], math.exp(expected_nll / expected_n))


def test_nll_formula_matches_cross_entropy():
    """Compare our NLL to F.cross_entropy on the same data — must match."""
    torch.manual_seed(0)
    model = TinyDeterministicLM(vocab_size=16)
    model.eval()
    window = torch.randint(0, 16, (10,))

    r = compute_window_singleshot(model, window)

    # Reference: explicit cross_entropy on shifted logits
    with torch.no_grad():
        out = model(input_ids=window.unsqueeze(0))
    shifted = out.logits[0, :-1, :].float()
    targets = window[1:]
    ref_nll = F.cross_entropy(shifted, targets, reduction="sum").item()

    assert abs(r.nll_sum - ref_nll) < 1e-4, (
        f"NLL mismatch: ours={r.nll_sum}, F.cross_entropy={ref_nll}"
    )
