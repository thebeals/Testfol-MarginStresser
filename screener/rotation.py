"""Fixed, never-tuned survivor-pool rotation overlays."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RotationVariant:
    name: str
    signal_asset: str
    period: int
    average: str


ROTATION_VARIANTS = (
    RotationVariant("sma200_spy", "SPY", 200, "sma"),
    RotationVariant("ema100_spy", "SPY", 100, "ema"),
)


def rotation_signal(prices: pd.DataFrame, variant: RotationVariant) -> pd.Series:
    """Return a boolean above-trend signal using fixed spec parameters."""
    signal = prices[variant.signal_asset]
    average = signal.rolling(variant.period).mean() if variant.average == "sma" else signal.ewm(span=variant.period, adjust=False).mean()
    return (signal > average).fillna(False)


def apply_rotation(
    prices: pd.DataFrame,
    above: dict[str, float],
    below: dict[str, float],
    variant: RotationVariant,
) -> pd.DataFrame:
    """Create a daily allocation frame for an already validated survivor."""
    signal = rotation_signal(prices, variant)
    tickers = sorted(set(above) | set(below))
    allocations = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
    for date, is_above in signal.items():
        allocation = above if is_above else below
        allocations.loc[date, list(allocation)] = list(allocation.values())
    return allocations.fillna(0.0)
