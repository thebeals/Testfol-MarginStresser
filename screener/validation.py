"""Anti-overfitting validation suite ported from the verified reference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats


class ConfidenceBadge(str, Enum):
    ROBUST = "Robust"
    CAUTION = "Caution"
    LIKELY_OVERFIT = "Likely Overfit"


@dataclass
class ValidationReport:
    train_sharpe: float
    test_sharpe: float
    walk_forward_sharpes: list[float]
    bootstrap_sharpe_ci: tuple[float, float]
    dsr: float
    permutation_p_value: float
    badge: ConfidenceBadge
    details: dict


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def train_test_split(returns: pd.Series, train_frac: float = 0.7) -> tuple[pd.Series, pd.Series]:
    r = returns.dropna()
    if r.empty:
        return r, r
    split = max(int(len(r) * train_frac), 1)
    return r.iloc[:split], r.iloc[split:]


def walk_forward(
    returns: pd.Series, *, train_size: int = 252, test_size: int = 63, step: int | None = None
) -> list[float]:
    r = returns.dropna()
    step = step or test_size
    sharpes = []
    for start in range(train_size, len(r) - test_size + 1, step):
        sharpes.append(sharpe_ratio(r.iloc[start : start + test_size]))
    return sharpes


def block_bootstrap_monte_carlo(
    returns: pd.Series, *, n_sims: int = 1000, block_size: int = 21, seed: int = 42
) -> np.ndarray:
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return np.array([sharpe_ratio(pd.Series(r))])
    block = min(block_size, max(len(r) // 2, 1))
    rng = np.random.default_rng(seed)
    result = np.empty(n_sims)
    for simulation in range(n_sims):
        starts = rng.integers(0, len(r), size=int(np.ceil(len(r) / block)))
        sample = np.concatenate([r[(start + np.arange(block)) % len(r)] for start in starts])[: len(r)]
        result[simulation] = sharpe_ratio(pd.Series(sample))
    return result


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    if n_obs < 3 or n_trials < 1:
        return 0.0
    gamma = 0.5772156649
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
        expected_max = (1 - gamma) * z1 + gamma * z2
    else:
        expected_max = 0.0
    denominator = np.sqrt(max(1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2, 1e-12))
    statistic = (observed_sharpe - expected_max) * np.sqrt(n_obs - 1) / denominator
    return float(stats.norm.cdf(statistic))


def permutation_test(returns: pd.Series, *, n_perms: int = 500, seed: int = 42) -> float:
    r = returns.dropna()
    if len(r) < 10:
        return 1.0
    observed = abs(sharpe_ratio(r))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perms):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(r))
        if abs(sharpe_ratio(pd.Series(r.to_numpy() * signs))) >= observed:
            count += 1
    return (count + 1) / (n_perms + 1)


def confidence_badge(*, test_sharpe: float, train_sharpe: float, dsr: float, perm_p: float, bootstrap_ci: tuple[float, float]) -> ConfidenceBadge:
    ratio = test_sharpe / train_sharpe if train_sharpe > 0 else 0.0
    if dsr >= 0.95 and perm_p < 0.05 and ratio >= 0.5 and bootstrap_ci[0] > 0:
        return ConfidenceBadge.ROBUST
    if dsr < 0.5 or ratio < 0.25 or perm_p > 0.20:
        return ConfidenceBadge.LIKELY_OVERFIT
    return ConfidenceBadge.CAUTION


def run_validation_suite(
    returns: pd.Series, *, n_trials: int = 1, n_bootstrap: int = 1000, n_perms: int = 500
) -> ValidationReport:
    train, test = train_test_split(returns)
    train_sr = sharpe_ratio(train)
    test_sr = sharpe_ratio(test)
    walk_forward_sharpes = walk_forward(returns)
    bootstrap = block_bootstrap_monte_carlo(returns, n_sims=n_bootstrap)
    ci = (float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5)))
    clean_train = train.dropna()
    skew = float(stats.skew(clean_train)) if len(clean_train) > 2 else 0.0
    kurtosis = float(stats.kurtosis(clean_train, fisher=False)) if len(clean_train) > 3 else 3.0
    dsr = deflated_sharpe_ratio(train_sr, n_trials, len(clean_train), skew, kurtosis)
    permutation_p = permutation_test(returns, n_perms=n_perms)
    badge = confidence_badge(
        test_sharpe=test_sr,
        train_sharpe=train_sr,
        dsr=dsr,
        perm_p=permutation_p,
        bootstrap_ci=ci,
    )
    return ValidationReport(
        train_sharpe=train_sr,
        test_sharpe=test_sr,
        walk_forward_sharpes=walk_forward_sharpes,
        bootstrap_sharpe_ci=ci,
        dsr=dsr,
        permutation_p_value=permutation_p,
        badge=badge,
        details={"n_obs": len(returns.dropna()), "n_trials": n_trials, "walk_forward_mean": float(np.mean(walk_forward_sharpes)) if walk_forward_sharpes else 0.0},
    )
