"""Continuous allocation search using the Phase 1 CMA-ES winner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from .fitness import bootstrap_mwrr_score
from .diversify import check_diversification


@dataclass(frozen=True)
class SearchResult:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    score: float
    evaluations: int


def _normalise(values: list[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 0.0:
        array.fill(1.0 / len(array))
    else:
        array /= total
    return tuple(float(value) for value in array)


def search_weights(
    returns: pd.DataFrame,
    tickers: tuple[str, ...],
    *,
    n_trials: int = 120,
    seed: int = 42,
    rebalance_freq: str = "None",
) -> SearchResult:
    """Search non-negative weights for one prescreened subset."""
    if not tickers:
        raise ValueError("at least one ticker is required")
    sampler = optuna.samplers.CmaEsSampler(seed=seed, with_margin=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        values = [trial.suggest_float(f"weight_{index}", 0.001, 1.0) for index in range(len(tickers))]
        weights = _normalise(values)
        allocation = dict(zip(tickers, weights))
        if not check_diversification(returns, allocation).passed:
            return -1.0
        return bootstrap_mwrr_score(
            returns,
            allocation,
            seed + trial.number,
            rebalance_freq=rebalance_freq,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    values = [study.best_trial.params[f"weight_{index}"] for index in range(len(tickers))]
    weights = _normalise(values)
    return SearchResult(tickers, weights, float(study.best_value), n_trials)
