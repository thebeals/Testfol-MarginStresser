"""Search deterministic daily EMA exits and rotation rules for the expert shortlist."""

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
OUTPUT = ROOT / "data" / "expert-ema-exit-rotation-results.json"
MEMO = ROOT / "data" / "expert-ema-exit-rotation-report.md"
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-08-17")


def _atr(prices: pd.DataFrame, length: int) -> pd.DataFrame:
    previous = prices.shift(1)
    true_range = pd.concat([prices - previous, (prices - previous).abs(), (prices - previous).abs()], axis=1)
    return true_range.T.groupby(level=0).max().T.rolling(length, min_periods=length).mean()


def _signals(prices: pd.DataFrame, strategy: str, length: int, multiple: float, filter_length: int, filter_exit: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy == "atr_band":
        price = prices.rolling(2).mean()
        average = prices.ewm(span=length, adjust=False, min_periods=length).mean()
        width = _atr(prices, length) * multiple
        bull = (price > average + width) & (price.shift(1) <= (average + width).shift(1))
        bear = (price < average - width) & (price.shift(1) >= (average - width).shift(1))
        return bull.fillna(False), bear.fillna(False)
    fast = prices.ewm(span=9, adjust=False, min_periods=20).mean()
    slow = prices.ewm(span=20, adjust=False, min_periods=20).mean()
    filter_average = prices.ewm(span=filter_length, adjust=False, min_periods=filter_length).mean()
    bull = (fast > slow) & (fast.shift(1) <= slow.shift(1)) & (prices > filter_average)
    bear = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    if filter_exit:
        bear |= prices < filter_average
    return bull.fillna(False), bear.fillna(False)


def _simulate(prices: pd.DataFrame, allocation: dict[str, float], strategy: str, length: int, multiple: float, filter_length: int, filter_exit: bool, rotation: str) -> pd.Series:
    tickers = list(allocation)
    returns = prices[tickers].pct_change().fillna(0.0)
    buys, sells = _signals(prices[tickers], strategy, length, multiple, filter_length, filter_exit)
    positions = {ticker: 100.0 * weight for ticker, weight in allocation.items()}
    cash = 0.0
    last_buy: str | None = None
    output: list[float] = []
    for index, timestamp in enumerate(prices.index):
        before = sum(positions.values()) + cash
        for ticker in tickers:
            positions[ticker] *= 1.0 + float(returns.loc[timestamp, ticker])
        if index > 0:
            sold = [ticker for ticker in tickers if bool(sells.loc[timestamp, ticker]) and positions[ticker] > 0]
            proceeds = sum(positions[ticker] for ticker in sold)
            for ticker in sold:
                positions[ticker] = 0.0
            if proceeds:
                if rotation == "cash":
                    cash += proceeds
                else:
                    eligible = [ticker for ticker in tickers if ticker not in sold and positions[ticker] > 0]
                    if rotation == "winner" and eligible:
                        winner = max(eligible, key=lambda ticker: float(prices[ticker].loc[timestamp] / prices[ticker].loc[prices.index[max(0, index - 63)]] - 1.0))
                        positions[winner] += proceeds
                    elif rotation == "last_buy" and last_buy in eligible:
                        positions[last_buy] += proceeds
                    else:
                        cash += proceeds
            for ticker in tickers:
                if bool(buys.loc[timestamp, ticker]):
                    last_buy = ticker
                    if positions[ticker] == 0 and cash > 0:
                        amount = cash * allocation[ticker] / sum(allocation[name] for name in tickers if positions[name] > 0 or name == ticker)
                        positions[ticker] = amount
                        cash -= amount
        after = sum(positions.values()) + cash
        output.append(after / before - 1.0 if before else 0.0)
    return pd.Series(output, index=prices.index)


def main() -> None:
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))["shortlist"]
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    raw = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)
    prices, _ = extend_hybrid_prices(raw["Close"].reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for row in shortlist for ticker in row["allocation"]})
    prices = prices.dropna(subset=required).loc[START:END]
    results = []
    for row in shortlist:
        allocation = row["allocation"]
        baseline = _method_returns(prices, allocation, row["method"])
        base_metrics = _metrics(baseline)
        configs = [("atr_band", length, multiple, 0, False) for length in (21, 55, 100) for multiple in (0.5, 1.0, 1.5)]
        configs += [("cross_filter", 0, 0.0, filter_length, filter_exit) for filter_length in (100, 200) for filter_exit in (False, True)]
        for strategy, length, multiple, filter_length, filter_exit in configs:
            for rotation in ("cash", "winner", "last_buy"):
                protected = _simulate(prices, allocation, strategy, length, multiple, filter_length, filter_exit, rotation)
                metrics = _metrics(protected)
                results.append({"expert_rank": row["expert_rank"], "allocation_id": row["allocation_id"], "method": row["method"], "strategy": strategy, "length": length, "multiple": multiple, "filter_length": filter_length, "filter_exit": filter_exit, "rotation": rotation, "baseline": base_metrics, "protected": metrics, "cagr_delta": metrics["cagr"] - base_metrics["cagr"], "dd_delta": metrics["max_drawdown"] - base_metrics["max_drawdown"], "recovery": _recovery_metrics(protected), "dca": _dca_metrics(protected)})
    results.sort(key=lambda row: row["cagr_delta"] + 0.35 * row["dd_delta"], reverse=True)
    robust = [row for row in results if row["cagr_delta"] >= -0.01 and row["dd_delta"] >= 0]
    report = {"tested": len(results), "robust": robust, "all": results}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# EMA Exit and Rotation Study", "", "Daily signals use prior-day information and execute on the next session. The Pine-style strategy uses `SMA(close, 2)` crossing EMA +/- ATR. The crossover strategy uses 9/20 EMA crosses with 100/200 EMA filters.", "", f"Tested **{len(results)}** configurations across the expert top 10.", "", "| Rank | Strategy | Parameters | Rotation | CAGR Change | DD Improvement |", "|---:|---|---|---|---:|---:|"]
    for row in robust[:40]:
        params = f"{row['length']} / ATR {row['multiple']}" if row["strategy"] == "atr_band" else f"9/20 + {row['filter_length']} ({'filter exit' if row['filter_exit'] else 'cross exit'})"
        lines.append(f"| {row['expert_rank']} | {row['strategy']} | {params} | {row['rotation']} | {row['cagr_delta']:+.2%} | {row['dd_delta']:+.2%} |")
    lines += ["", "No fees, taxes, slippage, or market impact are modeled. Rotation is deterministic: `winner` sends proceeds to the held asset with the strongest trailing 63-session return; `last_buy` sends proceeds to the most recent asset with a buy signal, otherwise cash."]
    MEMO.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tested": len(results), "robust": len(robust)}, indent=2))


if __name__ == "__main__":
    main()
