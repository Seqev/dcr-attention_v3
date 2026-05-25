"""
Reproducibility primitives — minimal subset of ABC1 (TECHNICAL_PATCH_v1 PATCH-01).

Provides ``record_environment(seed)`` that returns a JSON-serialisable dict
suitable for stamping into ``benchmarks/results.json`` records.

Scope decision: we adopt only the foot-stamping piece of ABC1, not the
4-hash composition + run_set_hash + seed_metric formula.  See
``docs/decisions/technical_patch_v1_review.md``.
"""

from __future__ import annotations
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, Optional
import os
import platform
import subprocess

import torch


def _safe_version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _git_sha() -> str:
    """
    Best-effort git SHA of the repo housing this file.  Returns
    ``"no-git"`` when not in a git checkout (e.g. during pip install).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "no-git"


def _device_name() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return f"cpu:{platform.processor() or platform.machine()}"


def record_environment(
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""
    Snapshot of the execution environment.

    Returned keys:
      * ``timestamp_utc``   — ISO 8601 UTC.
      * ``git_sha``         — short SHA, ``"no-git"`` if unavailable.
      * ``python``          — interpreter version string.
      * ``torch``, ``triton`` — package versions; ``"not-installed"`` if absent.
      * ``device``          — descriptive device string.
      * ``cuda_available``  — bool.
      * ``seed``            — integer seed if supplied, else ``None``.
      * ``platform``        — short OS / arch string.

    Adding hash-composition (ABC1's ``conditions_hash64`` etc.) is deferred
    until we actually have multi-condition statistical comparisons.
    """
    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": _safe_version("triton"),
        "device": _device_name(),
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": seed,
        "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
    }
    if extra:
        env["extra"] = extra
    return env
