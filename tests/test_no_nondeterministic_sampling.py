"""
PATCH-03 I_DET_1 — Deterministic-sampling discipline guard.

Phase 4b episode: a single ``torch.randperm(n)[:200]`` inside Stage A
landmark sampler caused phantom +5.35 pt accuracy lift that turned out to
be measurement noise.  The structural fix is a repository-wide ban on
non-deterministic sampling calls inside ``dcr_attention/`` source.

This test AST-walks every .py file in the package and flags any call to:
    torch.randperm  /  torch.rand  /  torch.randn  /  torch.randint  /  torch.normal
that does not pass ``generator=...``.  Tests are excluded — they use
``torch.manual_seed`` for global determinism, which is the standard pattern.

Failure here means a future contributor introduced a Phase-4b-class footgun.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# Functions that produce non-deterministic output without an explicit generator.
_BANNED_TORCH_FNS = {"randperm", "rand", "randn", "randint", "normal", "bernoulli"}


def _torch_call_target(node: ast.Call) -> str | None:
    """
    Return ``"torch.X"`` if ``node`` is a call to ``torch.X(...)``,
    otherwise ``None``.

    Handles the patterns ``torch.X(...)`` and ``torch.cuda.X(...)``; we only
    care about the first level (``torch.<fn>``).
    """
    fn = node.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        if fn.value.id == "torch":
            return f"torch.{fn.attr}"
    return None


def _has_generator_kwarg(node: ast.Call) -> bool:
    return any(kw.arg == "generator" for kw in node.keywords)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (lineno, banned-call-string) violations in ``path``."""
    tree = ast.parse(path.read_text())
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _torch_call_target(node)
        if target is None:
            continue
        fn_name = target.split(".", 1)[1]
        if fn_name not in _BANNED_TORCH_FNS:
            continue
        if _has_generator_kwarg(node):
            continue
        violations.append((node.lineno, target))
    return violations


def _package_root() -> Path:
    here = Path(__file__).resolve()
    # tests/test_no_nondeterministic_sampling.py is two levels above the package?
    # Actually structure is:  /repo/dcr_attention/  vs  /repo/tests/...
    return here.parent.parent / "dcr_attention"


def test_no_bare_random_sampling_in_package_source() -> None:
    """
    Walk every .py file under ``dcr_attention/`` and assert no banned
    sampling call lacks a ``generator=`` kwarg.
    """
    root = _package_root()
    assert root.is_dir(), f"package root not found at {root}"

    all_violations: list[tuple[Path, int, str]] = []
    for py_file in sorted(root.rglob("*.py")):
        for lineno, target in _scan_file(py_file):
            all_violations.append((py_file, lineno, target))

    if all_violations:
        msg_lines = ["Non-deterministic sampling without generator= found:"]
        for path, lineno, target in all_violations:
            rel = path.relative_to(root.parent)
            msg_lines.append(f"  {rel}:{lineno}  {target}(...)  — add generator=")
        msg_lines.append("")
        msg_lines.append(
            "Phase 4b discovery: a bare torch.randperm in Stage A produced "
            "phantom +5.35 pt accuracy that was actually measurement noise. "
            "Pass an explicit torch.Generator to keep results reproducible."
        )
        pytest.fail("\n".join(msg_lines))


def test_guard_actually_detects_violations(tmp_path) -> None:
    """
    Negative test: synthesize a .py file with a bare torch.randperm and
    verify ``_scan_file`` flags it.  Without this, the guard could silently
    pass when the AST walk is broken.
    """
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def f(n):\n"
        "    return torch.randperm(n)[:200]\n"
    )
    violations = _scan_file(bad)
    assert violations, "guard failed to detect bare torch.randperm"
    assert violations[0][1] == "torch.randperm"


def test_guard_accepts_generator_kwarg(tmp_path) -> None:
    """Positive control: with ``generator=`` the call is allowed."""
    good = tmp_path / "good.py"
    good.write_text(
        "import torch\n"
        "def f(n, g):\n"
        "    return torch.randperm(n, generator=g)\n"
    )
    assert _scan_file(good) == []
