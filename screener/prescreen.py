"""Mean-variance proxy prescreen for ticker subsets.

The prescreen is deliberately a shortlist, not a hard production rejection:
the real objective is contribution-timed MWRR and is evaluated later.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .universe import FACTOR_MAP, WRAPPER_FACTORS


@dataclass(frozen=True)
class SubsetScore:
    tickers: tuple[str, ...]
    score: float
    expected_return: float
    variance: float


def _proxy_score(returns: pd.DataFrame, tickers: tuple[str, ...], risk_aversion: float) -> SubsetScore:
    subset = returns.loc[:, list(tickers)]
    mean = subset.mean().to_numpy()
    covariance = subset.cov().to_numpy()
    weights = np.full(len(tickers), 1.0 / len(tickers))
    expected_return = float(weights @ mean)
    variance = float(weights @ covariance @ weights)
    return SubsetScore(tickers, expected_return - risk_aversion * variance, expected_return, variance)


def _factor_signature(tickers: tuple[str, ...]) -> tuple[str, ...]:
    factors = set()
    for ticker in tickers:
        direct = FACTOR_MAP.get(ticker)
        mapped = (direct,) if direct else WRAPPER_FACTORS.get(ticker, ())
        factors.update(factor.value for factor in mapped)
    return tuple(sorted(factors))


def prescreen_subsets(
    returns: pd.DataFrame,
    *,
    min_tickers: int = 2,
    max_tickers: int = 6,
    max_candidates: int = 100,
    risk_aversion: float = 10.0,
    min_factors: int = 3,
) -> list[SubsetScore]:
    """Return the best proxy subsets using a bounded branch-and-bound scan.

    The optimistic bound is the best singleton proxy score among remaining
    assets. Branches whose bound cannot beat the current shortlist floor are
    skipped; the output remains a candidate shortlist rather than a hard gate.
    """
    if returns.empty or len(returns.columns) < min_tickers:
        return []
    columns = tuple(str(column) for column in returns.columns)
    singletons = {
        ticker: _proxy_score(returns, (ticker,), risk_aversion).score for ticker in columns
    }
    scores: list[SubsetScore] = []
    floor = float("-inf")
    for size in range(min_tickers, min(max_tickers, len(columns)) + 1):
        for tickers in itertools.combinations(columns, size):
            factors = set()
            for ticker in tickers:
                direct = FACTOR_MAP.get(ticker)
                mapped = (direct,) if direct else WRAPPER_FACTORS.get(ticker, ())
                factors.update(factor.value for factor in mapped)
            if len(factors) < min_factors:
                continue
            optimistic = max(singletons[ticker] for ticker in tickers)
            if len(scores) >= max_candidates and optimistic < floor:
                continue
            scores.append(_proxy_score(returns, tickers, risk_aversion))
            scores.sort(key=lambda item: item.score, reverse=True)
            if len(scores) > max_candidates:
                scores.pop()
            floor = scores[-1].score if len(scores) >= max_candidates else float("-inf")
    # Keep proxy selection from collapsing onto one attractive defensive mix.
    # This is coverage only; the final allocation strategy remains downstream.
    by_signature: dict[tuple[str, ...], SubsetScore] = {}
    for score in scores:
        signature = _factor_signature(score.tickers)
        if signature not in by_signature:
            by_signature[signature] = score
    covered = list(by_signature.values())[:max_candidates]
    selected = {score.tickers for score in covered}
    for score in scores:
        if len(covered) >= max_candidates:
            break
        if score.tickers not in selected:
            covered.append(score)
            selected.add(score.tickers)
    return sorted(covered, key=lambda item: item.score, reverse=True)
