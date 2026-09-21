"""Build a drawdown-aware research report for distinct rebalance candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from screener.rebalance import rebalance_returns
from scripts.run_rebalance_matrix import _band_returns, _ema_returns


START = pd.Timestamp("2006-01-01")
END = pd.Timestamp("2026-08-17")
TEST_START = pd.Timestamp("2021-01-01")
MONTHLY_CONTRIBUTION = 1000.0
REGIMES = {
    "2008_financial_crisis": ("2008-01-01", "2009-03-31"),
    "2011_risk_off": ("2011-05-01", "2011-10-31"),
    "2015_2016_selloff": ("2015-08-01", "2016-02-29"),
    "2018_q4_selloff": ("2018-10-01", "2018-12-31"),
    "2020_covid_shock": ("2020-02-01", "2020-04-30"),
    "2022_inflation_bear": ("2022-01-01", "2022-10-31"),
    "2025_2026_recent": ("2025-01-01", "2026-08-17"),
}


def _allocation_label(allocation: dict[str, float]) -> str:
    return " / ".join(f"{ticker} {weight:.1%}" for ticker, weight in allocation.items())


def _method_fields(method: str) -> dict[str, object]:
    parts = method.split(":")
    if parts[0].startswith("EMA"):
        return {"family": parts[0], "frequency": parts[2], "policy": parts[1]}
    if parts[0] == "Calendar":
        return {"family": "Calendar", "frequency": parts[1], "policy": ""}
    return {"family": parts[0], "frequency": parts[1], "policy": ""}


def _metrics(returns: pd.Series) -> dict[str, float | int]:
    returns = returns.dropna()
    wealth = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1 / 365.25)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.0,
        "observations": int(len(returns)),
    }


def _method_returns(prices: pd.DataFrame, allocation: dict[str, float], method: str) -> pd.Series:
    fields = _method_fields(method)
    returns = prices.loc[:, list(allocation)].pct_change().dropna()
    if fields["family"] == "Calendar":
        return rebalance_returns(returns, allocation, fields["frequency"])
    if fields["family"] in ("RelativeBand", "AbsoluteBand"):
        mode = fields["family"].replace("Band", "").lower()
        band = float(method.split(":")[2])
        return _band_returns(prices, allocation, fields["frequency"], mode, band)
    return _ema_returns(prices, allocation, fields["frequency"], int(fields["family"][3:]), fields["policy"])


def _dca_metrics(returns: pd.Series) -> dict[str, float | int]:
    account = 0.0
    contributions = 0.0
    values: list[float] = []
    contributed: list[float] = []
    prior_month = None
    for timestamp, daily_return in returns.dropna().items():
        month = (timestamp.year, timestamp.month)
        contribution = MONTHLY_CONTRIBUTION if month != prior_month else 0.0
        contributions += contribution
        account = (account + contribution) * (1.0 + float(daily_return))
        values.append(account)
        contributed.append(contributions)
        prior_month = month
    series = pd.Series(values, index=returns.dropna().index)
    drawdown = series / series.cummax() - 1.0
    contribution_loss = series / pd.Series(contributed, index=series.index) - 1.0
    underwater = series < series.cummax() * (1.0 - 1e-12)
    return {
        "monthly_contribution": MONTHLY_CONTRIBUTION,
        "total_contributions": float(contributions),
        "ending_value": float(series.iloc[-1]),
        "max_account_drawdown": float(drawdown.min()),
        "worst_value_vs_contributions": float(contribution_loss.min()),
        "minimum_account_value": float(series.min()),
        "underwater_days": int(underwater.sum()),
        "underwater_months": int(underwater.resample("ME").max().sum()),
    }


def _recovery_metrics(returns: pd.Series) -> dict[str, object]:
    wealth = (1.0 + returns.dropna()).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    bottom_date = drawdown.idxmin()
    peak_series = wealth.loc[:bottom_date]
    peak_value = peak_series.max()
    peak_date = peak_series.idxmax()
    recovery = wealth.loc[bottom_date:][wealth.loc[bottom_date:] >= peak_value]
    return {
        "max_drawdown": float(drawdown.min()),
        "peak_date": peak_date.date().isoformat(),
        "bottom_date": bottom_date.date().isoformat(),
        "recovery_date": recovery.index[0].date().isoformat() if not recovery.empty else None,
        "recovery_days": int((recovery.index[0] - bottom_date).days) if not recovery.empty else None,
        "worst_daily_return": float(returns.min()),
        "worst_rolling_3m": float((1.0 + returns.dropna()).rolling(63).apply(np.prod, raw=True).min() - 1.0),
    }


def _rolling_outperformance(returns: pd.Series, benchmark: pd.Series) -> dict[str, float | int]:
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 756:
        return {"windows": 0, "beat_count": 0, "beat_rate": 0.0}
    portfolio = (1.0 + joined.iloc[:, 0]).rolling(756).apply(np.prod, raw=True) ** (252.0 / 756.0) - 1.0
    market = (1.0 + joined.iloc[:, 1]).rolling(756).apply(np.prod, raw=True) ** (252.0 / 756.0) - 1.0
    valid = pd.concat([portfolio, market], axis=1).dropna()
    return {
        "windows": int(len(valid)),
        "beat_count": int((valid.iloc[:, 0] > valid.iloc[:, 1]).sum()),
        "beat_rate": float((valid.iloc[:, 0] > valid.iloc[:, 1]).mean()),
    }


def _regime_metrics(returns: pd.Series, spy: pd.Series, vt: pd.Series) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, (start, end) in REGIMES.items():
        rows = {}
        for label, series in (("portfolio", returns), ("SPY", spy), ("VT", vt)):
            window = series.loc[start:end].dropna()
            rows[label] = _metrics(window) if len(window) > 1 else {}
        result[name] = rows
    return result


def _select_top25(matrix: list[dict[str, object]]) -> list[dict[str, object]]:
    representatives: dict[tuple[int, str], dict[str, object]] = {}
    for row in matrix:
        # Apply the same 20% test-window drawdown gate to each method, not
        # only to the allocation's original default rebalance configuration.
        if float(row["max_drawdown"]) < -0.20:
            continue
        fields = _method_fields(row["method"])
        key = (int(row["allocation_id"]), str(fields["family"]))
        current = representatives.get(key)
        if current is None or float(row["cagr"]) > float(current["cagr"]):
            representatives[key] = row
    return sorted(representatives.values(), key=lambda row: float(row["cagr"]), reverse=True)[:25]


def build_report(prices: pd.DataFrame, benchmark_prices: pd.DataFrame, matrix: list[dict[str, object]], verification: list[dict[str, object]]) -> dict[str, object]:
    top = _select_top25(matrix)
    verified = {(int(row["allocation_id"]), row["method"]): row for row in verification}
    spy = benchmark_prices["SPY"].pct_change().dropna()
    vt = benchmark_prices["VT"].pct_change().dropna()
    candidates = []
    for rank, row in enumerate(top, start=1):
        allocation = {ticker: float(weight) for ticker, weight in row["allocation"].items()}
        portfolio_returns = _method_returns(prices, allocation, row["method"])
        full_returns = portfolio_returns.loc[START:END].dropna()
        # Reinitialize each strategy at the locked test start so EMA state and
        # warm-up behavior match the matrix that selected this candidate.
        test_returns = _method_returns(prices.loc[TEST_START:END], allocation, row["method"]).dropna()
        full = _metrics(full_returns)
        test = _metrics(test_returns)
        recovery = _recovery_metrics(test_returns)
        dca = _dca_metrics(full_returns)
        testfol = verified.get((int(row["allocation_id"]), row["method"]))
        contribution = {}
        for ticker, weight in allocation.items():
            asset = prices[ticker].pct_change().loc[START:END].dropna()
            contribution[ticker] = {"weight": weight, "asset_cagr": _metrics(asset)["cagr"], "weighted_cagr": weight * _metrics(asset)["cagr"]}
        candidates.append(
            {
                "rank": rank,
                "allocation_id": row["allocation_id"],
                "allocation": allocation,
                "allocation_label": _allocation_label(allocation),
                "method": row["method"],
                **_method_fields(row["method"]),
                "local_full": full,
                "local_test": test,
                "recovery": recovery,
                "dca": dca,
                "asset_contribution": contribution,
                "rolling_3y_vs_spy": _rolling_outperformance(full_returns, spy),
                "rolling_3y_vs_vt": _rolling_outperformance(full_returns, vt),
                "regimes": _regime_metrics(full_returns, spy, vt),
                "testfol_validated": testfol is not None,
                "testfol_method": testfol["method"] if testfol else None,
                "testfol_test": testfol.get("testfol_test", {}) if testfol else {},
            }
        )
    valid = [row for row in candidates if row["testfol_validated"]]
    return {
        "selection_rule": "Top 25 by local CAGR after applying a 20% test-window max-drawdown gate and selecting one best method per allocation x method family. Absolute band thresholds are collapsed; AbsoluteBand, RelativeBand, EMA100, EMA200, and Calendar remain distinct families.",
        "period": {"start": START.date().isoformat(), "end": END.date().isoformat(), "test_start": TEST_START.date().isoformat()},
        "benchmarks": {"SPY": "SPDR S&P 500 ETF", "VT": "Vanguard Total World Stock ETF"},
        "dca_assumptions": {"monthly_contribution": MONTHLY_CONTRIBUTION, "margin": False, "taxes": False},
        "candidates": candidates,
        "summary": {
            "candidate_count": len(candidates),
            "distinct_allocations": len({row["allocation_id"] for row in candidates}),
            "testfol_exact_method_count": len(valid),
            "median_test_cagr": float(np.median([row["local_test"]["cagr"] for row in candidates])),
            "median_test_drawdown": float(np.median([row["local_test"]["max_drawdown"] for row in candidates])),
            "median_spy_rolling_3y_beat_rate": float(np.median([row["rolling_3y_vs_spy"]["beat_rate"] for row in candidates])),
            "median_vt_rolling_3y_beat_rate": float(np.median([row["rolling_3y_vs_vt"]["beat_rate"] for row in candidates])),
            "median_dca_worst_loss_vs_contributions": float(np.median([row["dca"]["worst_value_vs_contributions"] for row in candidates])),
        },
        "interpretation": {
            "confidence": "Historical evidence is useful but not a probability guarantee. The sample contains only one 2008 crisis and one 2020 shock, and Testfol exact-method validation covers only the separately verified methods.",
            "protection": "Protection comes from diversification, defensive assets, and the selected rebalance/signal rules. It does not prevent losses, gaps, prolonged drawdowns, or every asset declining together.",
            "dca": "Without margin, dollar-cost averaging cannot create a negative account balance, but it can compound losses and leave capital underwater for years. A zero outcome remains possible if the underlying assets permanently fail.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/screener-results.json")
    parser.add_argument("--universe", default="data/allocation-feasibility.json")
    parser.add_argument("--matrix", default="data/rebalance-matrix-results.jsonl")
    parser.add_argument("--verification", default="data/rebalance-testfol-verification.json")
    parser.add_argument("--output", default="data/rebalance-research-report.json")
    parser.add_argument("--csv", default="data/rebalance-research-top25.csv")
    args = parser.parse_args()
    accepted = json.loads(Path(args.input).read_text(encoding="utf-8"))["accepted"]
    allocations = [row["local"]["allocation"] for row in accepted]
    universe = json.loads(Path(args.universe).read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for allocation in allocations for ticker in allocation})
    prices = prices.dropna(subset=required)
    benchmark_prices = yf.download(["SPY", "VT"], start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"].dropna()
    matrix = [json.loads(line) for line in Path(args.matrix).read_text(encoding="utf-8").splitlines() if line.strip()]
    verification = json.loads(Path(args.verification).read_text(encoding="utf-8"))
    report = build_report(prices, benchmark_prices, matrix, verification)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for row in report["candidates"]:
        rows.append({
            "rank": row["rank"], "allocation_id": row["allocation_id"], "allocation": row["allocation_label"],
            "method": row["method"], "family": row["family"], "local_test_cagr": row["local_test"]["cagr"],
            "local_test_max_drawdown": row["local_test"]["max_drawdown"], "sharpe": row["local_test"]["sharpe"],
            "max_recovery_days": row["recovery"]["recovery_days"], "dca_worst_vs_contributions": row["dca"]["worst_value_vs_contributions"],
            "rolling_3y_spy_beat_rate": row["rolling_3y_vs_spy"]["beat_rate"], "rolling_3y_vt_beat_rate": row["rolling_3y_vs_vt"]["beat_rate"],
            "testfol_exact_method_validated": row["testfol_validated"],
        })
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
