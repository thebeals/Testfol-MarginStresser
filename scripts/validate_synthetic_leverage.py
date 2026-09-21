"""Validate synthetic LETF returns against observed ETF histories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from screener.synthetic import LeverageSpec, leveraged_returns, tracking_metrics
from app.services.data_service import get_fed_funds_rate


ANCHORS = {
    "UPRO_vs_SPY": ("UPRO", "SPY", 3.0, 0.91),
    "TQQQ_vs_QQQ": ("TQQQ", "QQQ", 3.0, 0.84),
    "TMF_vs_TLT": ("TMF", "TLT", 3.0, 1.05),
}


def _load_returns(start: str, end: str):
    symbols = sorted({symbol for actual, underlying, _, _ in ANCHORS.values() for symbol in (actual, underlying)})
    prices = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, threads=False)["Close"]
    return prices.pct_change(fill_method=None).dropna(how="all")


def _fit_spread(actual, underlying, base_spec: LeverageSpec, calibration_end) -> float:
    calibration_actual = actual.loc[:calibration_end]
    calibration_underlying = underlying.loc[:calibration_end]
    best_spread = base_spec.spread_pct
    best_error = float("inf")
    for spread in np.arange(0.0, 2.01, 0.05):
        spec = LeverageSpec(**{**base_spec.__dict__, "spread_pct": float(spread)})
        simulated = leveraged_returns(calibration_underlying, spec)
        common = calibration_actual.index.intersection(simulated.index)
        if len(common) < 252:
            continue
        error = float(np.mean((calibration_actual.loc[common] - simulated.loc[common]) ** 2))
        if error < best_error:
            best_error = error
            best_spread = float(spread)
    return best_spread


def validate(start: str, end: str, calibration_fraction: float) -> dict[str, object]:
    returns = _load_returns(start, end)
    funding = get_fed_funds_rate()
    result: dict[str, object] = {"date_range": {"start": start, "end": end}, "anchors": {}}
    for name, (actual_ticker, underlying_ticker, leverage, expense) in ANCHORS.items():
        actual = returns[actual_ticker].dropna()
        underlying = returns[underlying_ticker].dropna()
        common = actual.index.intersection(underlying.index)
        actual = actual.loc[common]
        underlying = underlying.loc[common]
        split = max(1, int(len(common) * calibration_fraction))
        calibration_end = common[split - 1]
        base = LeverageSpec(leverage=leverage, expense_ratio_pct=expense, funding_rate_pct=funding if funding is not None else 4.0)
        fitted_spread = _fit_spread(actual, underlying, base, calibration_end)
        fitted = LeverageSpec(**{**base.__dict__, "spread_pct": fitted_spread})
        simulated = leveraged_returns(underlying, fitted)
        holdout_start = common[min(split, len(common) - 1)]
        result["anchors"][name] = {
            "actual": actual_ticker,
            "underlying": underlying_ticker,
            "calibration_end": str(calibration_end.date()),
            "holdout_start": str(holdout_start.date()),
            "fitted_spread_pct": fitted_spread,
            "calibration": tracking_metrics(actual.loc[:calibration_end], simulated.loc[:calibration_end]),
            "holdout": tracking_metrics(actual.loc[holdout_start:], simulated.loc[holdout_start:]),
            "full_period": tracking_metrics(actual, simulated),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--output", default="data/synthetic-leverage-validation.json")
    args = parser.parse_args()
    result = validate(args.start, args.end, args.calibration_fraction)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
