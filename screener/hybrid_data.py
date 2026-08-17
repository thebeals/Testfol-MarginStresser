"""Hybrid observed/simulated price histories for validated LETF backfills."""

from __future__ import annotations

import pandas as pd

from .synthetic import LeverageSpec, leveraged_returns


LETF_SPECS: dict[str, tuple[str, LeverageSpec]] = {
    "UPRO": ("SPY", LeverageSpec(3.0, 0.91, spread_pct=0.40)),
    "SSO": ("SPY", LeverageSpec(2.0, 0.89, spread_pct=0.40)),
    "TQQQ": ("QQQ", LeverageSpec(3.0, 0.84, spread_pct=0.65)),
    "TMF": ("TLT", LeverageSpec(3.0, 1.05, spread_pct=0.50)),
    "TYD": ("IEF", LeverageSpec(3.0, 0.95, spread_pct=0.50)),
    "UBT": ("TLT", LeverageSpec(2.0, 0.95, spread_pct=0.50)),
}


def extend_validated_letfs(prices: pd.DataFrame, funding_rate_pct: pd.Series | None) -> tuple[pd.DataFrame, list[str]]:
    """Backfill mapped LETFs from their underlying and splice observed returns."""
    result = prices.copy()
    simulated: list[str] = []
    for ticker, (underlying, base_spec) in LETF_SPECS.items():
        if underlying not in result.columns or result[underlying].dropna().empty:
            continue
        underlying_prices = result[underlying].dropna().sort_index()
        underlying_returns = underlying_prices.pct_change(fill_method=None).dropna()
        spec = LeverageSpec(
            leverage=base_spec.leverage,
            expense_ratio_pct=base_spec.expense_ratio_pct,
            swap_exposure=base_spec.swap_exposure,
            spread_pct=base_spec.spread_pct,
            funding_rate_pct=funding_rate_pct if funding_rate_pct is not None else base_spec.funding_rate_pct,
        )
        synthetic_returns = leveraged_returns(underlying_returns, spec)
        if ticker in result.columns and result[ticker].notna().any():
            observed = result[ticker].dropna().sort_index().pct_change(fill_method=None).dropna()
            combined_returns = pd.concat([synthetic_returns, observed]).groupby(level=0).last().sort_index()
        else:
            combined_returns = synthetic_returns
        result[ticker] = 100.0 * (1.0 + combined_returns).cumprod()
        simulated.append(ticker)
    return result, simulated
