"""Pre-inception return models with observable-overlap validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeverageSpec:
    leverage: float
    expense_ratio_pct: float
    swap_exposure: float = 1.0
    spread_pct: float = 0.5
    funding_rate_pct: float | pd.Series = 4.0


def leveraged_returns(underlying_returns: pd.Series, spec: LeverageSpec) -> pd.Series:
    """Apply daily-reset leverage, financing, spread, and expense costs."""
    returns = pd.to_numeric(underlying_returns, errors="coerce").dropna().astype(float)
    if isinstance(spec.funding_rate_pct, pd.Series):
        funding = pd.to_numeric(spec.funding_rate_pct, errors="coerce").reindex(returns.index, method="ffill").fillna(4.0)
    else:
        funding = float(spec.funding_rate_pct)
    funding_daily = funding / 100.0 / 252.0
    spread_daily = spec.spread_pct / 100.0 / 252.0
    expense_daily = spec.expense_ratio_pct / 100.0 / 252.0
    result = (
        spec.leverage * returns
        - spec.swap_exposure * (spec.leverage - 1.0) * (funding_daily + spread_daily)
        - expense_daily
    ).clip(lower=-1.0)
    return result.rename(underlying_returns.name)


def tracking_metrics(actual_returns: pd.Series, simulated_returns: pd.Series) -> dict[str, float | int | str]:
    """Compare a simulated series with observed ETF returns on common dates."""
    aligned = pd.concat(
        {"actual": actual_returns, "simulated": simulated_returns}, axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("actual and simulated returns have no common observations")
    actual = aligned["actual"]
    simulated = aligned["simulated"]
    actual_curve = (1.0 + actual).cumprod()
    simulated_curve = (1.0 + simulated).cumprod()

    def max_drawdown(curve: pd.Series) -> float:
        return float((curve / curve.cummax() - 1.0).min())

    return {
        "start": str(aligned.index[0].date()),
        "end": str(aligned.index[-1].date()),
        "observations": len(aligned),
        "daily_correlation": float(actual.corr(simulated)),
        "annualized_tracking_error": float((actual - simulated).std() * np.sqrt(252.0)),
        "actual_cumulative_return": float(actual_curve.iloc[-1] - 1.0),
        "simulated_cumulative_return": float(simulated_curve.iloc[-1] - 1.0),
        "actual_max_drawdown": max_drawdown(actual_curve),
        "simulated_max_drawdown": max_drawdown(simulated_curve),
    }
