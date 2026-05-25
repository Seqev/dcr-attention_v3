"""
Tests for the environment-capture utility.

We verify the schema (keys present, types correct), not the exact values —
those depend on the host machine.
"""

from __future__ import annotations

from dcr_attention.reproducibility import record_environment


_REQUIRED_KEYS = {
    "timestamp_utc", "git_sha", "python", "torch", "triton",
    "device", "cuda_available", "seed", "platform",
}


def test_record_environment_has_all_required_keys() -> None:
    env = record_environment(seed=0)
    missing = _REQUIRED_KEYS - env.keys()
    assert not missing, f"missing keys: {missing}"


def test_record_environment_seed_is_passed_through() -> None:
    env = record_environment(seed=12345)
    assert env["seed"] == 12345


def test_record_environment_seed_none_is_allowed() -> None:
    env = record_environment(seed=None)
    assert env["seed"] is None


def test_extra_payload_is_attached() -> None:
    env = record_environment(seed=1, extra={"k_window": 64, "model": "synth"})
    assert env["extra"] == {"k_window": 64, "model": "synth"}


def test_no_extra_means_no_extra_key() -> None:
    env = record_environment(seed=1)
    assert "extra" not in env


def test_record_is_json_serialisable() -> None:
    """Used by benchmark scripts to append into ``benchmarks/results.json``."""
    import json
    env = record_environment(seed=1)
    json.dumps(env)                                 # must not raise


def test_cuda_available_is_bool() -> None:
    env = record_environment(seed=0)
    assert isinstance(env["cuda_available"], bool)
