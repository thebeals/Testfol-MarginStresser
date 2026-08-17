"""Rebalance frequency verification and screener wiring.

Phase 2 step 8: Verified 2026-08-17. The local shadow-backtest engine
supports all screener-required frequencies per SCREENER-SPEC.md (None /
Monthly / Quarterly / Yearly). The testfol.io live API supports a broader
enum (Daily, Weekly, Bi-monthly, Every 4 Months, Semiannually, Every 2
Years, Every 5 Years) but the screener deliberately restricts to the
default four per spec --- calendar-based only, no exotic frequencies,
tested per candidate as a fixed parameter sweep rather than a search
dimension.

Verification detail:
  Local engine (shadow_backtest.py rebalance_freq) accepts:
    None, Monthly, Quarterly, Yearly, Custom, Threshold, Threshold+Calendar
  testfol.io live API accepts (confirmed via 422 validation):
    Daily, Weekly, Monthly, Bi-monthly, Every 4 Months, Quarterly,
    Semiannually, Yearly, Every 2 Years, Every 5 Years, None
  Overlap (screener default set): None, Monthly, Quarterly, Yearly.
  No bridging needed --- the local engine covers the screener's needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

SCREENER_REBALANCE_FREQUENCIES: Final[tuple[str, ...]] = ("None", "Monthly", "Quarterly", "Yearly")

TESTFOLIO_API_FREQUENCIES: Final[tuple[str, ...]] = (
    "Daily",
    "Weekly",
    "Monthly",
    "Bi-monthly",
    "Every 4 Months",
    "Quarterly",
    "Semiannually",
    "Yearly",
    "Every 2 Years",
    "Every 5 Years",
    "None",
)

LOCAL_ENGINE_FREQUENCIES: Final[tuple[str, ...]] = (
    "None",
    "Monthly",
    "Quarterly",
    "Yearly",
    "Custom",
    "Threshold",
    "Threshold+Calendar",
)


@dataclass(frozen=True)
class RebalanceConfig:
    freq: str
    offset: int = 0


def screener_defaults() -> tuple[RebalanceConfig, ...]:
    return tuple(RebalanceConfig(freq) for freq in SCREENER_REBALANCE_FREQUENCIES)


def verify_engine_support() -> dict[str, bool]:
    return {freq: freq in LOCAL_ENGINE_FREQUENCIES for freq in SCREENER_REBALANCE_FREQUENCIES}


def rebalance_returns(
    returns: pd.DataFrame,
    allocation: dict[str, float],
    frequency: str = "None",
) -> pd.Series:
    """Simulate daily portfolio returns with calendar rebalancing.

    Rebalancing occurs after each day's asset return when the next calendar
    period begins. This keeps the operation in-process and deterministic while
    preserving the original daily return index for MWRR evaluation.
    """
    if frequency not in SCREENER_REBALANCE_FREQUENCIES:
        raise ValueError(f"unsupported screener rebalance frequency: {frequency}")
    tickers = list(allocation)
    weights = np.asarray([allocation[ticker] for ticker in tickers], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("allocation weights must be non-negative and sum to one")
    asset_returns = returns.loc[:, tickers].to_numpy(dtype=float)
    holdings = weights.copy()
    values = np.empty(len(asset_returns), dtype=float)
    prior_period = None
    for index, (date, daily_returns) in enumerate(zip(returns.index, asset_returns)):
        period = _period_key(date, frequency)
        if prior_period is not None and period != prior_period:
            total = holdings.sum()
            holdings = weights * total
        before = holdings.sum()
        holdings *= 1.0 + daily_returns
        values[index] = holdings.sum() / before - 1.0
        prior_period = period
    return pd.Series(values, index=returns.index, name="portfolio_return")


def _period_key(date: pd.Timestamp, frequency: str) -> tuple[int, ...] | None:
    if frequency == "None":
        return None
    if frequency == "Monthly":
        return date.year, date.month
    if frequency == "Quarterly":
        return date.year, date.quarter
    if frequency == "Yearly":
        return (date.year,)
    raise ValueError(f"unsupported screener rebalance frequency: {frequency}")
