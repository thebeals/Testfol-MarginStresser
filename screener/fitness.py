"""Contribution-timed bootstrap MWRR fitness and plateau stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bakeoff import BLOCK_SIZE, block_bootstrap, mwrr


@dataclass(frozen=True)
class FitnessResult:
    score: float
    mean_mwrr: float
    mwrr_std: float
    plateau_stability: float


def _portfolio_returns(returns: pd.DataFrame, allocation: dict[str, float]) -> np.ndarray:
    tickers = list(allocation)
    weights = np.asarray([allocation[ticker] for ticker in tickers], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("allocation weights must be non-negative and sum to one")
    return returns.loc[:, tickers].to_numpy() @ weights


def bootstrap_mwrr_score(
    returns: pd.DataFrame,
    allocation: dict[str, float],
    seed: int,
    *,
    bootstraps: int = 24,
    block_size: int = BLOCK_SIZE,
) -> float:
    """Return the bootstrap mean MWRR with a dispersion penalty."""
    history = _portfolio_returns(returns, allocation)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(bootstraps):
        scores.append(mwrr(block_bootstrap(history, rng), returns.index))
    values = np.asarray(scores)
    return float(values.mean() - 0.25 * values.std())


def plateau_stability(
    returns: pd.DataFrame,
    allocation: dict[str, float],
    seed: int,
    *,
    perturbation: float = 0.02,
) -> float:
    """Measure how much score survives small weight perturbations."""
    base = bootstrap_mwrr_score(returns, allocation, seed)
    values = []
    tickers = list(allocation)
    for index, ticker in enumerate(tickers):
        if len(tickers) == 1 or allocation[ticker] <= perturbation:
            continue
        other = tickers[(index + 1) % len(tickers)]
        changed = dict(allocation)
        changed[ticker] -= perturbation
        changed[other] += perturbation
        values.append(bootstrap_mwrr_score(returns, changed, seed + index + 1))
    if not values:
        return 1.0
    return float(np.mean(np.asarray(values) >= base - 0.01))


def score_candidate(
    returns: pd.DataFrame,
    allocation: dict[str, float],
    seed: int,
) -> FitnessResult:
    """Combine bootstrap MWRR with a plateau-stability multiplier."""
    history = _portfolio_returns(returns, allocation)
    rng = np.random.default_rng(seed)
    values = np.asarray([mwrr(block_bootstrap(history, rng), returns.index) for _ in range(24)])
    mean = float(values.mean())
    std = float(values.std())
    stability = plateau_stability(returns, allocation, seed + 10_000)
    return FitnessResult(mean - 0.25 * std + 0.05 * stability, mean, std, stability)
