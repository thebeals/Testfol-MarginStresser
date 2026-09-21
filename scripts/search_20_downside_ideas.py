"""Search 20 deterministic downside-protection ideas, ten variants each."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from scripts.build_rebalance_research import _metrics, _method_returns

ROOT = Path(__file__).parents[1]
SHORTLIST = ROOT / "data" / "rebalance-expert-shortlist.json"
UNIVERSE = ROOT / "data" / "allocation-feasibility.json"
OUTPUT = ROOT / "data" / "expert-20-downside-ideas-results.json"
REPORT = ROOT / "data" / "expert-20-downside-ideas-report.md"
START, END = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-08-17")


IDEAS = [
    ("EMA trend exit", "Exit an asset below its EMA and re-enter above it.", "ema", [(n, 0.0) for n in (20, 50, 100, 150, 200, 250, 300, 400, 500, 600)]),
    ("EMA cross exit", "Exit on a fast/slow EMA bearish cross; re-enter on a bullish cross.", "cross", [(f, s) for f, s in ((5, 20), (8, 21), (9, 20), (10, 30), (12, 26), (20, 50), (30, 100), (50, 150), (50, 200), (100, 200))]),
    ("EMA ATR buffer", "Exit only when price falls below EMA minus an ATR buffer; re-enter above EMA.", "atr", [(n, m) for n, m in ((20, .5), (20, 1), (50, .5), (50, 1), (100, .5), (100, 1), (100, 1.5), (200, .5), (200, 1), (200, 1.5))]),
    ("Donchian exit", "Exit below the prior rolling low and re-enter above the rolling high.", "donchian", [(n, 0.0) for n in (10, 20, 30, 55, 75, 100, 126, 150, 200, 252)]),
    ("Trailing peak stop", "Exit after a fixed drawdown from each asset's rolling peak; re-enter after recovery.", "trail", [(n, d) for n, d in ((20,.08),(30,.10),(50,.12),(63,.15),(100,.15),(126,.18),(150,.20),(200,.20),(252,.25),(300,.30))]),
    ("Asset drawdown brake", "Exit when an asset's drawdown from its all-time observed high exceeds the threshold.", "asset_dd", [(d, 0.0) for d in (.08,.10,.12,.15,.18,.20,.25,.30,.35,.40)]),
    ("Consecutive down days", "Exit after consecutive negative days; re-enter after consecutive positive days.", "streak", [(n, n) for n in range(1, 11)]),
    ("RSI regime", "Exit in weak RSI regimes and re-enter when RSI recovers above a fixed level.", "rsi", [(n, x) for n, x in ((5,35),(7,35),(10,35),(14,35),(14,40),(14,45),(21,40),(21,45),(30,40),(30,45))]),
    ("MACD regime", "Exit while MACD is below its signal line; re-enter on a bullish cross.", "macd", [(f, s) for f, s in ((5,20),(8,21),(9,26),(12,26),(15,30),(20,50),(26,52),(30,90),(50,100),(50,200))]),
    ("Momentum lookback", "Exit negative trailing momentum and re-enter when trailing momentum turns positive.", "momentum", [(n, 0.0) for n in (5,10,20,30, 60, 90, 126, 150, 200, 252)]),
    ("Volatility shock", "Exit an asset when volatility spikes and its return is negative; re-enter after volatility normalizes.", "volshock", [(n, m) for n, m in ((10,1.25),(10,1.5),(20,1.25),(20,1.5),(20,2),(30,1.5),(30,2),(60,1.5),(60,2),(60,2.5))]),
    ("Mean-reversion guard", "Exit deep weakness below an SMA; re-enter only after price recovers above the SMA.", "meanrev", [(n, d) for n, d in ((20,.05),(20,.10),(50,.05),(50,.10),(100,.05),(100,.10),(150,.10),(200,.10),(200,.15),(252,.15))]),
    ("EMA slope", "Exit when the long EMA slopes down; re-enter when its slope turns positive.", "slope", [(n, w) for n, w in ((20,3),(50,3),(50,10),(100,3),(100,10),(150,10),(200,10),(200,20),(300,20),(400,20))]),
    ("Two-factor vote", "Exit when at least two of EMA, momentum, and channel signals are bearish.", "vote", [(n, n) for n in (20,30,50,75,100,126,150,200,252,300)]),
    ("Breadth gate", "Move the portfolio to cash when too many allocated assets are below their EMA.", "breadth", [(n, k) for n, k in ((20,1),(50,1),(100,1),(100,2),(100,3),(200,1),(200,2),(200,3),(200,4),(252,3))]),
    ("Relative strength", "Exit the weakest momentum assets and concentrate proceeds in the strongest eligible asset.", "relative", [(n, k) for n, k in ((20,1),(20,2),(50,1),(50,2),(100,1),(100,2),(126,1),(126,2),(252,1),(252,2))]),
    ("Volatility weighting", "Reduce or exit high-volatility assets and redistribute to lower-volatility holdings.", "volweight", [(n, x) for n, x in ((10,1.25),(20,1.25),(20,1.5),(30,1.5),(60,1.5),(60,2),(90,1.5),(90,2),(126,2),(252,2))]),
    ("Scale-out trend", "Sell half on a soft trend break and fully exit on a harder break; re-enter above trend.", "scale", [(n, d) for n, d in ((20,.03),(20,.05),(50,.03),(50,.05),(100,.03),(100,.05),(100,.10),(200,.05),(200,.10),(252,.10))]),
    ("Cooldown re-entry", "Use an EMA exit, then require a fixed number of sessions before re-entry.", "cooldown", [(n, c) for n, c in ((20,3),(50,3),(50,5),(100,3),(100,5),(100,10),(150,5),(200,5),(200,10),(252,20))]),
    ("Portfolio catastrophe brake", "Force all positions to cash after a portfolio drawdown or rolling-loss event, then wait for recovery.", "catastrophe", [(d, w) for d, w in ((.08,5),(.10,5),(.12,5),(.15,5),(.15,10),(.20,10),(.20,20),(.25,10),(.25,20),(.30,20))]),
]


def _rsi(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    change = prices.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _signals(prices: pd.DataFrame, kind: str, a: float, b: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if kind == "catastrophe":
        empty = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        return empty, empty
    if kind == "asset_dd":
        peak = prices.cummax()
        return prices < peak * (1 - a), prices >= peak * (1 - a / 2)
    if kind == "breadth":
        average = prices.ewm(span=int(a), adjust=False, min_periods=int(a)).mean()
        return prices < average, prices >= average
    ema = prices.ewm(span=int(a) if kind not in ("cross", "macd") else int(b), adjust=False, min_periods=int(a) if kind not in ("cross", "macd") else int(b)).mean()
    if kind == "ema":
        return prices < ema, prices >= ema
    if kind == "cross":
        fast, slow = prices.ewm(span=int(a), adjust=False).mean(), prices.ewm(span=int(b), adjust=False).mean()
        return (fast < slow), (fast >= slow)
    if kind == "atr":
        atr = prices.diff().abs().rolling(int(a), min_periods=int(a)).mean()
        return prices < ema - atr * b, prices >= ema
    if kind in ("donchian", "channel"):
        low, high = prices.shift(1).rolling(int(a), min_periods=int(a)).min(), prices.shift(1).rolling(int(a), min_periods=int(a)).max()
        return prices < low, prices > high
    if kind == "trail":
        peak = prices.rolling(int(a), min_periods=int(a)).max()
        return prices < peak * (1 - b), prices >= peak * (1 - b / 2)
    if kind == "streak":
        down = (prices.pct_change() < 0).rolling(int(a)).sum() >= int(a)
        up = (prices.pct_change() > 0).rolling(int(b)).sum() >= int(b)
        return down, up
    if kind == "rsi":
        value = _rsi(prices, int(a))
        return value < b, value >= 50
    if kind == "macd":
        fast, slow = prices.ewm(span=int(a), adjust=False).mean(), prices.ewm(span=int(b), adjust=False).mean()
        signal = (fast - slow).ewm(span=9, adjust=False).mean()
        return fast - slow < signal, fast - slow >= signal
    if kind == "momentum":
        mom = prices.pct_change(int(a))
        return mom < 0, mom >= 0
    if kind == "volshock":
        vol = prices.pct_change().rolling(int(a)).std()
        baseline = vol.rolling(int(a) * 3).median()
        return (vol > baseline * b) & (prices.pct_change() < 0), vol <= baseline * b
    if kind == "meanrev":
        average = prices.rolling(int(a)).mean()
        return prices < average * (1 - b), prices >= average
    if kind == "slope":
        slope = ema.diff(int(b))
        return slope < 0, slope >= 0
    if kind == "vote":
        mom = prices.pct_change(int(a)) < 0
        below = prices < ema
        low = prices < prices.shift(1).rolling(int(a)).min()
        bearish = below.astype(int) + mom.astype(int) + low.astype(int) >= 2
        return bearish, ~bearish
    if kind == "relative":
        mom = prices.pct_change(int(a))
        rank = mom.rank(axis=1, ascending=False, method="min")
        return rank > prices.shape[1] - int(b), rank <= int(b)
    if kind == "volweight":
        vol = prices.pct_change().rolling(int(a)).std()
        median = vol.median(axis=1)
        return vol.gt(median * b, axis=0), vol.le(median, axis=0)
    if kind == "scale":
        return prices < ema * (1 - b), prices >= ema
    if kind == "cooldown":
        return prices < ema, prices >= ema
    raise ValueError(kind)


def _simulate(prices: pd.DataFrame, allocation: dict[str, float], kind: str, a: float, b: float) -> pd.Series:
    tickers = list(allocation)
    returns = prices[tickers].pct_change().fillna(0)
    sell, buy = _signals(prices[tickers], kind, a, b)
    positions = {ticker: 100 * weight for ticker, weight in allocation.items()}
    cash, catastrophe, last_exit = 0.0, False, -99999
    catastrophe_threshold = float(a) if kind == "catastrophe" else 0.20
    catastrophe_window = int(b) if kind == "catastrophe" else 5
    wealth = 100.0
    high = 100.0
    output = []
    for i, date in enumerate(prices.index):
        before = sum(positions.values()) + cash
        for ticker in tickers:
            positions[ticker] *= 1 + float(returns.loc[date, ticker])
        total = sum(positions.values()) + cash
        wealth = total
        high = max(high, wealth)
        portfolio_loss = wealth / high - 1
        rolling_loss = (np.prod([1 + value for value in output[-catastrophe_window:]]) - 1) if len(output) >= catastrophe_window else 0
        if not catastrophe and (portfolio_loss <= -catastrophe_threshold or rolling_loss <= -0.15):
            cash, positions, catastrophe, last_exit = total, {ticker: 0.0 for ticker in tickers}, True, i
        elif catastrophe and i - last_exit >= 10 and wealth >= high * 0.97:
            positions, cash, catastrophe = {ticker: wealth * weight for ticker, weight in allocation.items()}, 0.0, False
        elif i > 0 and not catastrophe:
            if kind == "breadth" and int(sell.loc[date].sum()) >= int(b):
                exits = [ticker for ticker in tickers if positions[ticker] > 0]
            else:
                exits = [ticker for ticker in tickers if bool(sell.loc[date, ticker]) and positions[ticker] > 0]
            proceeds = sum(positions[ticker] for ticker in exits)
            for ticker in exits:
                positions[ticker] = 0.0
            if kind == "scale":
                proceeds *= 0.5
            cash += proceeds
            eligible = [ticker for ticker in tickers if bool(buy.loc[date, ticker])]
            if eligible and cash:
                weights = sum(allocation[ticker] for ticker in eligible)
                for ticker in eligible:
                    positions[ticker] += cash * allocation[ticker] / weights
                cash = 0.0
        output.append(total / before - 1 if before else 0.0)
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
        baseline = _metrics(_method_returns(prices, row["allocation"], row["method"]))
        for idea, description, kind, variants in IDEAS:
            for variant, (a, b) in enumerate(variants, 1):
                protected = _simulate(prices, row["allocation"], kind, a, b)
                metrics = _metrics(protected)
                results.append({"idea": idea, "description": description, "variant": variant, "expert_rank": row["expert_rank"], "parameters": [a, b], "baseline": baseline, "protected": metrics, "cagr_delta": metrics["cagr"] - baseline["cagr"], "dd_delta": metrics["max_drawdown"] - baseline["max_drawdown"]})
    results.sort(key=lambda x: x["cagr_delta"] + 0.5 * x["dd_delta"], reverse=True)
    robust = [x for x in results if x["cagr_delta"] >= -0.01 and x["dd_delta"] >= 0]
    report = {"ideas": [{"id": i + 1, "name": x[0], "description": x[1], "variants": x[3]} for i, x in enumerate(IDEAS)], "tested": len(results), "robust": robust, "all": results}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# 20 Downside Ideas x 10 Variants", "", "Every simulation uses prior-day signals, next-session execution, and a portfolio catastrophe brake: force cash at 20% drawdown or a 15% five-session loss, then require ten sessions and recovery to within 3% of the high before re-entry.", "", f"Tested **{len(results)}** configurations across the 10 expert portfolios.", "", "## Ideas"]
    for i, (name, description, _, _) in enumerate(IDEAS, 1):
        lines.append(f"{i}. **{name}:** {description}")
    lines += ["", "## Best Balanced Results", "", "| Idea | Variant | Portfolio | CAGR Change | DD Improvement | Protected CAGR | Protected DD |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x in robust[:30]:
        lines.append(f"| {x['idea']} | {x['variant']} | {x['expert_rank']} | {x['cagr_delta']:+.2%} | {x['dd_delta']:+.2%} | {x['protected']['cagr']:.2%} | {x['protected']['max_drawdown']:.2%} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tested": len(results), "robust": len(robust)}, indent=2))


if __name__ == "__main__":
    main()
