"""Pricing helpers that keep constituent identity separate from price symbols."""

from __future__ import annotations

import pandas as pd


def ensure_price_tickers(weights: pd.DataFrame) -> pd.DataFrame:
    """Return weights with a normalized PriceTicker column."""
    result = weights.copy()
    result["Ticker"] = result["Ticker"].astype(str).str.strip().str.upper()
    if "PriceTicker" not in result:
        result["PriceTicker"] = result["Ticker"]
    else:
        result["PriceTicker"] = (
            result["PriceTicker"]
            .where(result["PriceTicker"].notna(), result["Ticker"])
            .astype(str)
            .str.strip()
            .str.upper()
        )
        result.loc[result["PriceTicker"].eq(""), "PriceTicker"] = result["Ticker"]
    return result


def price_ticker_map(rows: pd.DataFrame) -> pd.Series:
    """Map audited membership symbols to their identity-safe pricing symbols."""
    if rows.empty:
        return pd.Series(dtype=str)
    normalized = ensure_price_tickers(rows)
    return (
        normalized.drop_duplicates("Ticker", keep="first")
        .set_index("Ticker")["PriceTicker"]
    )


def collapse_to_pricing_weights(
    security_weights: pd.Series,
    ticker_map: pd.Series,
    available_tickers=None,
) -> pd.Series:
    """Collapse membership weights onto available pricing series and normalize."""
    if security_weights is None or security_weights.empty:
        return pd.Series(dtype=float)

    frame = security_weights.rename("Weight").to_frame()
    frame["PriceTicker"] = frame.index.to_series().map(ticker_map).fillna(
        frame.index.to_series()
    )
    if available_tickers is not None:
        available = set(available_tickers)
        frame = frame[frame["PriceTicker"].isin(available)]
    if frame.empty:
        return pd.Series(dtype=float)

    result = frame.groupby("PriceTicker")["Weight"].sum().astype(float)
    total = float(result.sum())
    return result / total if total > 0 else pd.Series(dtype=float)
