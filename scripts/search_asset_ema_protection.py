"""Test individual-asset EMA protection and confirmation thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from scripts.build_rebalance_research import _dca_metrics, _metrics, _method_returns, _recovery_metrics


ROOT = Path(__file__).parents[1]
SHORTLIST = ROOT / "data" / "rebalance-expert-shortlist.json"
UNIVERSE = ROOT / "data" / "allocation-feasibility.json"
OUTPUT = ROOT / "data" / "expert-asset-ema-results.json"
MEMO = ROOT / "data" / "expert-asset-ema-report.md"
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-08-17")


def _simulate(prices: pd.DataFrame, allocation: dict[str, float], window: int, threshold: int, mode: str) -> pd.Series:
    tickers = list(allocation)
    returns = prices[tickers].pct_change().fillna(0.0)
    ema = prices[tickers].ewm(span=window, adjust=False, min_periods=window).mean()
    values = {ticker: 100.0 * weight for ticker, weight in allocation.items()}
    cash = 0.0
    active = {ticker: True for ticker in tickers}
    full_cash = False
    output = []
    for index, timestamp in enumerate(prices.index):
        before = sum(values.values()) + cash
        for ticker in tickers:
            if active[ticker]:
                values[ticker] *= 1.0 + float(returns.loc[timestamp, ticker])
        total = sum(values.values()) + cash
        output_return = total / before - 1.0 if before else 0.0
        signals_ready = index > 0 and ema.loc[timestamp].notna().all()
        if signals_ready:
            below = sum(float(prices.loc[timestamp, ticker]) < float(ema.loc[timestamp, ticker]) for ticker in tickers)
            if mode == "full_cash":
                should_cash = below >= threshold
                if should_cash and not full_cash:
                    cash = total
                    values = {ticker: 0.0 for ticker in tickers}
                    full_cash = True
                elif not should_cash and full_cash:
                    values = {ticker: total * weight for ticker, weight in allocation.items()}
                    cash = 0.0
                    full_cash = False
            else:
                next_active = {ticker: float(prices.loc[timestamp, ticker]) >= float(ema.loc[timestamp, ticker]) for ticker in tickers}
                if sum(not state for state in next_active.values()) >= threshold:
                    active = next_active
                    investable = total
                    active_weight_total = sum(allocation[ticker] for ticker in tickers if active[ticker])
                    if active_weight_total:
                        values = {ticker: investable * allocation[ticker] / active_weight_total if active[ticker] else 0.0 for ticker in tickers}
                        cash = 0.0
                    else:
                        values = {ticker: 0.0 for ticker in tickers}
                        cash = investable
                else:
                    active = {ticker: True for ticker in tickers}
                    values = {ticker: total * allocation[ticker] for ticker in tickers}
                    cash = 0.0
        output.append(output_return)
    return pd.Series(output, index=prices.index)


def main() -> None:
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))["shortlist"]
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for row in shortlist for ticker in row["allocation"]})
    prices = prices.dropna(subset=required).loc[START:END]
    results = []
    for row in shortlist:
        allocation = row["allocation"]
        baseline = _method_returns(prices, allocation, row["method"])
        for window in (50, 100, 200):
            for threshold in (1, 2):
                for mode in ("individual", "full_cash"):
                    protected = _simulate(prices, allocation, window, threshold, mode)
                    base_metrics = _metrics(baseline)
                    metrics = _metrics(protected)
                    results.append({"expert_rank": row["expert_rank"], "allocation_id": row["allocation_id"], "method": row["method"], "window": window, "threshold": threshold, "mode": mode, "baseline": base_metrics, "protected": metrics, "cagr_delta": metrics["cagr"] - base_metrics["cagr"], "dd_delta": metrics["max_drawdown"] - base_metrics["max_drawdown"], "recovery": _recovery_metrics(protected), "dca": _dca_metrics(protected)})
    robust = [row for row in results if row["cagr_delta"] >= -0.01 and row["dd_delta"] >= 0]
    robust.sort(key=lambda row: row["cagr_delta"] + 0.35 * row["dd_delta"], reverse=True)
    report = {"rules": "Signals use each asset's prior-day close versus its EMA. Changes apply on the next session. Individual mode removes under-EMA assets and redistributes to remaining active targets; full_cash mode exits everything when the below-EMA count reaches the threshold.", "tested": len(results), "robust": robust[:100], "all": results}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Individual-Asset EMA Protection Report", "", report["rules"], "", f"Tested **{len(results)}** configurations across the expert top 10.", "", "| Rank | Window | Below EMA Trigger | Mode | CAGR Change | DD Improvement |", "|---:|---:|---:|---|---:|---:|"]
    for row in robust[:30]:
        lines.append(f"| {row['expert_rank']} | {row['window']} | {row['threshold']} assets | {row['mode']} | {row['cagr_delta']:+.2%} | {row['dd_delta']:+.2%} |")
    lines.extend(["", "## Interpretation", "", "Individual mode preserves assets that remain above the EMA while moving weak sleeves to cash. Full-cash mode is more defensive but can discard good holdings because of one or two weak components. Re-entry can lag sharp rebounds. These results are historical and require separate out-of-sample validation."])
    MEMO.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tested": len(results), "robust": len(robust)}, indent=2))


if __name__ == "__main__":
    main()
