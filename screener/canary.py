"""Reference-portfolio health check for every generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


REFERENCE_ALLOCATION = {"UPRO": 0.55, "TLT": 0.45}
REFERENCE_BASELINE = {
    "cagr": 19.29,
    "max_drawdown": -49.9,
    "sharpe": 0.78,
}


@dataclass(frozen=True)
class CanaryResult:
    healthy: bool
    observed: dict[str, float]
    differences: dict[str, float]
    message: str


def check_canary(
    observed: dict[str, float],
    *,
    tolerances: dict[str, float] | None = None,
) -> CanaryResult:
    """Compare one sequentially obtained canary result to the known baseline."""
    tolerances = tolerances or {"cagr": 1.0, "max_drawdown": 3.0, "sharpe": 0.08}
    differences = {
        key: float(observed[key] - REFERENCE_BASELINE[key])
        for key in REFERENCE_BASELINE
        if key in observed
    }
    missing = set(REFERENCE_BASELINE) - set(observed)
    healthy = not missing and all(abs(differences[key]) <= tolerances[key] for key in differences)
    message = "OK" if healthy else "BROKEN: canary drift or missing metrics"
    return CanaryResult(healthy, observed, differences, message)


def run_canary(fetch: Callable[[dict[str, float]], dict[str, float]]) -> CanaryResult:
    """Run exactly one canary fetch; callers must not parallelize it."""
    return check_canary(fetch(REFERENCE_ALLOCATION))
