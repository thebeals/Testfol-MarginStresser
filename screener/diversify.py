"""Variance-share and stress-window diversification checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import FACTOR_MAP, WRAPPER_FACTORS, Factor

STRESS_WINDOWS = {
    "2008_gfc": ("2007-10-09", "2009-03-09"),
    "2020_covid": ("2020-02-19", "2020-03-23"),
    "2022_rates": ("2022-01-03", "2022-10-12"),
}
VARIANCE_SHARE_CAP = 0.50
FACTOR_BREADTH_THRESHOLD = 0.02
MIN_FACTOR_BREADTH = 4


@dataclass(frozen=True)
class DiversificationReport:
    factor_variance_share: dict[str, float]
    factor_breadth: int
    stress_correlations: dict[str, float]
    passed: bool
    violations: tuple[str, ...]


def _ticker_factors(ticker: str) -> tuple[Factor, ...]:
    if ticker in FACTOR_MAP:
        return (FACTOR_MAP[ticker],)
    if ticker in WRAPPER_FACTORS:
        return WRAPPER_FACTORS[ticker]
    raise KeyError(f"unknown ticker: {ticker}")


def _factor_returns(returns: pd.DataFrame, allocation: dict[str, float]) -> pd.DataFrame:
    result = pd.DataFrame(index=returns.index)
    for factor in Factor:
        columns = []
        weights = []
        for ticker, weight in allocation.items():
            factors = _ticker_factors(ticker)
            if factor in factors:
                columns.append(ticker)
                weights.append(weight / len(factors))
        if columns:
            factor_weights = np.asarray(weights)
            factor_weights /= factor_weights.sum()
            result[factor.value] = returns.loc[:, columns].to_numpy() @ factor_weights
    return result


def variance_share(returns: pd.DataFrame, allocation: dict[str, float]) -> dict[str, float]:
    """Compute each factor's contribution to portfolio variance."""
    factors = _factor_returns(returns, allocation)
    if factors.empty:
        return {}
    covariance = factors.cov().to_numpy()
    weights = np.asarray([
        sum(allocation[ticker] / len(_ticker_factors(ticker)) for ticker in allocation if factor in _ticker_factors(ticker))
        for factor in factors.columns
    ])
    total = float(weights @ covariance @ weights)
    if total <= 0:
        return {factor: 0.0 for factor in factors.columns}
    contributions = weights * (covariance @ weights)
    return {factor: float(max(contribution / total, 0.0)) for factor, contribution in zip(factors.columns, contributions)}


def stress_correlation(returns: pd.DataFrame, allocation: dict[str, float]) -> dict[str, float]:
    """Report the maximum absolute factor correlation in each stress window."""
    factors = _factor_returns(returns, allocation)
    result = {}
    for name, (start, end) in STRESS_WINDOWS.items():
        window = factors.loc[start:end]
        corr = window.corr().to_numpy()
        values = np.abs(corr[np.triu_indices_from(corr, k=1)]) if len(corr) > 1 else np.array([0.0])
        valid = values[np.isfinite(values)]
        result[name] = float(valid.max()) if valid.size else 0.0
    return result


def check_diversification(
    returns: pd.DataFrame,
    allocation: dict[str, float],
    *,
    variance_cap: float = VARIANCE_SHARE_CAP,
    stress_correlation_cap: float = 0.95,
    factor_breadth_threshold: float = FACTOR_BREADTH_THRESHOLD,
    min_factor_breadth: int = MIN_FACTOR_BREADTH,
) -> DiversificationReport:
    shares = variance_share(returns, allocation)
    correlations = stress_correlation(returns, allocation)
    factor_breadth = sum(share > factor_breadth_threshold for share in shares.values())
    violations = tuple(
        [f"factor:{factor}" for factor, share in shares.items() if share > variance_cap]
        + ([f"factor_breadth:{factor_breadth}/{min_factor_breadth}"] if factor_breadth < min_factor_breadth else [])
        + [f"stress:{window}" for window, value in correlations.items() if value > stress_correlation_cap]
    )
    return DiversificationReport(shares, factor_breadth, correlations, not violations, violations)
