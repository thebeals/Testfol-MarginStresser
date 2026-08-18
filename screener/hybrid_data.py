"""Hybrid observed/simulated price histories for validated LETF backfills."""

from __future__ import annotations

import json
from pathlib import Path

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

TESTFOL_CACHE_PATH = Path(__file__).parents[1] / "data" / "testfol-simulated-returns.json"


def load_testfol_simulated_prices(path: str | Path = TESTFOL_CACHE_PATH) -> pd.DataFrame:
    """Load cached Testfol daily percentage returns as normalized price series."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames: dict[str, pd.Series] = {}
    for ticker, metadata in payload.get("series", {}).items():
        rows = metadata.get("daily_returns_pct", [])
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["date", "return_pct", "value"])
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["return_pct"] = pd.to_numeric(frame["return_pct"], errors="raise") / 100.0
        returns = frame.drop_duplicates("date", keep="last").set_index("date")["return_pct"].sort_index()
        frames[ticker] = 100.0 * (1.0 + returns).cumprod()
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


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


def extend_hybrid_prices(
    prices: pd.DataFrame,
    funding_rate_pct: pd.Series | None,
    *,
    testfol_cache_path: str | Path = TESTFOL_CACHE_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add validated LETF and cached Testfol histories with source provenance."""
    result, letf_simulated = extend_validated_letfs(prices, funding_rate_pct)
    source_by_ticker = {str(ticker): "observed" for ticker in prices.columns}
    for ticker in letf_simulated:
        source_by_ticker[ticker] = "letf_simulated"

    testfol_prices = load_testfol_simulated_prices(testfol_cache_path)
    for ticker in testfol_prices.columns:
        result[ticker] = testfol_prices[ticker]
        source_by_ticker[str(ticker)] = "testfol_simulated"

    return result, {
        "source_by_ticker": source_by_ticker,
        "letf_simulated_tickers": letf_simulated,
        "testfol_simulated_tickers": [str(ticker) for ticker in testfol_prices.columns],
    }
