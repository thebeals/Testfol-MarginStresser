"""Backtest practical downside overlays for the expert shortlist and write a memo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from scripts.build_rebalance_research import _dca_metrics, _method_returns, _metrics, _recovery_metrics


ROOT = Path(__file__).parents[1]
SHORTLIST_PATH = ROOT / "data" / "rebalance-expert-shortlist.json"
UNIVERSE_PATH = ROOT / "data" / "allocation-feasibility.json"
RESULT_PATH = ROOT / "data" / "rebalance-expert-protection.json"
MEMO_PATH = ROOT / "data" / "expert-top10-investment-memo.md"
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2026-08-17")


def _period_key(timestamp: pd.Timestamp) -> tuple[int, int]:
    return timestamp.year, timestamp.month


def _overlay(
    base_returns: pd.Series,
    asset_returns: pd.DataFrame,
    cash_returns: pd.Series,
    mode: str,
) -> tuple[pd.Series, dict[str, float | int]]:
    base_returns = base_returns.dropna()
    asset_returns = asset_returns.reindex(base_returns.index).fillna(0.0)
    cash_returns = cash_returns.reindex(base_returns.index).fillna(0.0)
    base_wealth = (1.0 + base_returns).cumprod()
    ema = base_wealth.ewm(span=100, adjust=False, min_periods=100).mean()
    invested = True
    prior_period = None
    output = []
    cash_days = 0
    exits = 0
    for timestamp in base_returns.index:
        period = _period_key(timestamp)
        monthly_check = prior_period is not None and period != prior_period
        if not invested:
            cash_days += 1
            daily = float(cash_returns.loc[timestamp])
            if monthly_check and pd.notna(ema.loc[timestamp]) and base_wealth.loc[timestamp] >= ema.loc[timestamp]:
                invested = True
        else:
            daily = float(base_returns.loc[timestamp])
            if mode == "ema100" and monthly_check and pd.notna(ema.loc[timestamp]) and base_wealth.loc[timestamp] < ema.loc[timestamp]:
                invested = False
                exits += 1
            elif mode == "crash10" and (asset_returns.loc[timestamp] <= -0.10).any():
                invested = False
                exits += 1
        output.append(daily)
        prior_period = period
    return pd.Series(output, index=base_returns.index), {"cash_days": cash_days, "cash_pct": cash_days / len(base_returns), "exits": exits}


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _candidate_memo(row: dict[str, object], results: dict[str, dict[str, object]]) -> str:
    classes = row["diversification"]["asset_classes"]
    leveraged = row["diversification"]["leveraged_weight"]
    base = results["baseline"]["test"]
    ema = results["ema100_cash"]["test"]
    crash = results["crash10_cash"]["test"]
    strengths = row["rationale"]["why_consider"]
    risks = row["rationale"]["what_can_go_wrong"]
    protection = []
    if leveraged > 0:
        protection.append("Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.")
    if "gold" in classes or "managed futures" in classes or "commodities" in classes:
        protection.append("Retain the diversifier sleeves; they are the intended non-equity shock absorbers.")
    if "bonds" in classes or "cash" in classes:
        protection.append("Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.")
    protection.append("Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.")
    return f"""### #{row['expert_rank']}: {row['allocation_label']}\n\n**Method:** `{row['method']}`  \n**Asset classes:** {', '.join(classes)}  \n**Leveraged ETF weight:** {_fmt_pct(leveraged)}  \n**Exact Testfol method:** {'Yes' if row['testfol_validated'] else 'No'}\n\n**Why it is good**\n\n""" + "\n".join(f"- {item}" for item in strengths) + "\n\n**Downside risks**\n\n" + "\n".join(f"- {item}" for item in risks) + "\n\n**Protection I would use**\n\n" + "\n".join(f"- {item}" for item in protection) + f"\n\n**Protection test, 2021-01-01 through 2026-08-17**\n\n| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |\n|---|---:|---:|---:|---:|---:|\n| Baseline | {_fmt_pct(base['cagr'])} | {_fmt_pct(base['max_drawdown'])} | {base['sharpe']:.2f} | {results['baseline']['recovery']['recovery_days'] or 'Not recovered'} | 0.0% |\n| EMA100 monthly cash | {_fmt_pct(ema['cagr'])} | {_fmt_pct(ema['max_drawdown'])} | {ema['sharpe']:.2f} | {results['ema100_cash']['recovery']['recovery_days'] or 'Not recovered'} | {_fmt_pct(results['ema100_cash']['overlay']['cash_pct'])} |\n| 10% component crash cash | {_fmt_pct(crash['cagr'])} | {_fmt_pct(crash['max_drawdown'])} | {crash['sharpe']:.2f} | {results['crash10_cash']['recovery']['recovery_days'] or 'Not recovered'} | {_fmt_pct(results['crash10_cash']['overlay']['cash_pct'])} |\n\n**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.\n"""


def build(prices: pd.DataFrame, cash: pd.Series) -> dict[str, object]:
    shortlist = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))["shortlist"]
    output = []
    for row in shortlist:
        allocation = row["allocation"]
        base = _method_returns(prices.loc[START:END], allocation, row["method"]).dropna()
        assets = prices.loc[START:END, list(allocation)].pct_change().reindex(base.index).fillna(0.0)
        variants = {}
        for name, returns, overlay in (
            ("baseline", base, {"cash_days": 0, "cash_pct": 0.0, "exits": 0}),
            ("ema100_cash", *_overlay(base, assets, cash, "ema100")),
            ("crash10_cash", *_overlay(base, assets, cash, "crash10")),
        ):
            variants[name] = {
                "test": _metrics(returns),
                "recovery": _recovery_metrics(returns),
                "dca": _dca_metrics(returns),
                "overlay": overlay,
            }
        output.append({"expert_rank": row["expert_rank"], "allocation_id": row["allocation_id"], "method": row["method"], "allocation_label": row["allocation_label"], "source": row, **variants})
    return {"period": {"start": START.date().isoformat(), "end": END.date().isoformat()}, "cash_proxy": "SHV total-return series", "rules": {"ema100_cash": "At the monthly check, move the whole portfolio to SHV when the portfolio wealth index is below its EMA100; re-enter when it is above EMA100.", "crash10_cash": "Move the whole portfolio to SHV on the next session after any allocated component loses at least 10% in one day; re-enter at a monthly check when portfolio wealth is above EMA100."}, "candidates": output}


def main() -> None:
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    shortlist = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))["shortlist"]
    required = sorted({ticker for row in shortlist for ticker in row["allocation"]})
    prices = prices.dropna(subset=required)
    cash = prices["SHV"].pct_change().fillna(0.0) if "SHV" in prices else pd.Series(0.0, index=prices.index)
    report = build(prices, cash)
    RESULT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memo = ["# Expert Top-10 Investment Memo", "", "This memo reviews the expert shortlist through an investor-risk lens and then tests two explicit cash overlays. Results are historical simulations, not guarantees or personalized advice.", "", "## Shortlist", "", "The shortlist candidates all beat the locked SPY test CAGR and pass the method-level 20% drawdown gate. The selection balances return, drawdown, diversification, leverage, crisis behavior, rolling consistency, and Testfol validation.", "", "## Protection Rules Tested", "", f"- **EMA100 monthly cash:** {report['rules']['ema100_cash']}", f"- **10% component-crash cash:** {report['rules']['crash10_cash']}", "- Cash uses SHV total-return data where available.", "- No margin, tax, slippage, or leverage was added by these overlays.", ""]
    for candidate in report["candidates"]:
        memo.append(_candidate_memo(candidate["source"], {key: candidate[key] for key in ("baseline", "ema100_cash", "crash10_cash")}))
    memo.extend(["## Bottom Line", "", "A cash overlay can reduce drawdown, but it can also sell after a shock and miss the rebound. The component-crash rule is especially prone to false exits in portfolios containing TQQQ, UPRO, TMF, TYD, or SSO. The EMA100 rule is slower and more systematic, but it can still suffer a large loss before the monthly exit and can whipsaw in sideways markets.", "", "The practical protection is position sizing, limiting leveraged overlap, preserving liquidity, and accepting that no rule guarantees protection from a synchronized selloff."])
    MEMO_PATH.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(report["candidates"]), "memo": str(MEMO_PATH)}, indent=2))


if __name__ == "__main__":
    main()
