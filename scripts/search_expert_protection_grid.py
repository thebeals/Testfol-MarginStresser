"""Search a robust protection grid for the expert shortlist."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.services.data_service import get_fed_funds_rate
from screener.hybrid_data import extend_hybrid_prices
from scripts.build_rebalance_research import _method_returns, _metrics


ROOT = Path(__file__).parents[1]
SHORTLIST = ROOT / "data" / "rebalance-expert-shortlist.json"
UNIVERSE = ROOT / "data" / "allocation-feasibility.json"
OUTPUT = ROOT / "data" / "expert-protection-grid-results.json"
CSV = ROOT / "data" / "expert-protection-grid-top100.csv"
MARKDOWN = ROOT / "data" / "expert-protection-grid-report.md"
START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-08-17")
SPLITS = {"train": ("2015-01-01", "2018-12-31"), "validation": ("2019-01-01", "2020-12-31"), "test": ("2021-01-01", "2026-08-17")}


def _period(timestamp: pd.Timestamp, frequency: str) -> tuple[int, ...]:
    if frequency == "Daily":
        return (timestamp.year, timestamp.month, timestamp.day)
    if frequency == "Weekly":
        return timestamp.isocalendar().year, timestamp.isocalendar().week
    return timestamp.year, timestamp.month


def _overlay(base: pd.Series, assets: pd.DataFrame, cash: pd.Series, config: dict[str, object], cost_bps: float = 0.0) -> pd.Series:
    base = base.dropna()
    assets = assets.reindex(base.index).fillna(0.0)
    cash = cash.reindex(base.index).fillna(0.0)
    base_values = base.to_numpy(dtype=float)
    asset_values = assets.to_numpy(dtype=float)
    cash_values = cash.to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + base_values)
    ema = pd.Series(wealth).ewm(span=int(config["ema_span"]), adjust=False, min_periods=int(config["ema_span"])).mean().to_numpy()
    if config["review"] == "Daily":
        periods = np.arange(len(base_values))
    elif config["review"] == "Weekly":
        periods = base.index.to_period("W").asi8
    else:
        periods = base.index.to_period("M").asi8
    exposure = 1.0
    cooldown = 0
    output = np.empty(len(base_values), dtype=float)
    prior_exposure = 1.0
    peak_wealth = 0.0
    for index in range(len(base_values)):
        # Signals formed after today's close affect the next session only.
        review = index + 1 < len(base_values) and periods[index + 1] != periods[index]
        daily = exposure * base_values[index] + (1.0 - exposure) * cash_values[index]
        if cost_bps and exposure != prior_exposure:
            daily -= abs(exposure - prior_exposure) * cost_bps / 10000.0
        output[index] = daily
        applied_exposure = exposure
        if cooldown > 0:
            cooldown -= 1
        crash = float(config["crash_threshold"]) > 0 and (asset_values[index] <= -float(config["crash_threshold"])).any()
        peak_wealth = max(peak_wealth, wealth[index])
        trailing = float(config["trailing_stop"]) > 0 and wealth[index] / peak_wealth - 1.0 <= -float(config["trailing_stop"])
        ema_below = review and not np.isnan(ema[index]) and wealth[index] < ema[index]
        ema_above = review and not np.isnan(ema[index]) and wealth[index] >= ema[index]
        trigger = str(config["trigger"])
        exit_signal = (trigger in {"ema", "combined"} and ema_below) or (trigger in {"crash", "combined"} and crash) or trailing
        if exit_signal:
            exposure = min(exposure, 1.0 - float(config["derisk"]))
            cooldown = int(config["cooldown"])
        elif exposure < 1.0 and cooldown == 0 and ema_above:
            exposure = 1.0
        # Preserve the exposure actually used for today's return; any signal
        # above applies to the next session and is charged then.
        prior_exposure = applied_exposure
    return pd.Series(output, index=base.index)


def _period_metrics(returns: pd.Series) -> dict[str, dict[str, float | int]]:
    return {name: _metrics(returns.loc[start:end]) for name, (start, end) in SPLITS.items()}


def _score(result: dict[str, object], baseline: dict[str, dict[str, float | int]]) -> float:
    pre = []
    for split in ("train", "validation"):
        candidate = result["metrics"][split]
        base = baseline[split]
        cagr_retention = candidate["cagr"] - base["cagr"]
        dd_improvement = candidate["max_drawdown"] - base["max_drawdown"]
        pre.append(cagr_retention + 0.35 * dd_improvement)
    return float(np.mean(pre))


def main() -> None:
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))["shortlist"]
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    tickers = tuple(universe["eligible_tickers"])
    prices = yf.download(list(tickers), start="2006-01-01", end="2026-08-18", auto_adjust=True, progress=False, threads=False)["Close"]
    prices, _ = extend_hybrid_prices(prices.reindex(columns=tickers).dropna(axis=1, how="all"), get_fed_funds_rate())
    required = sorted({ticker for row in shortlist for ticker in row["allocation"]})
    prices = prices.dropna(subset=required)
    cash = prices["SHV"].pct_change().fillna(0.0) if "SHV" in prices else pd.Series(0.0, index=prices.index)
    configs = []
    for values in itertools.product(("ema", "crash", "combined"), (50, 75, 100, 150, 200), ("Daily", "Weekly", "Monthly"), (0.0, 0.075, 0.10, 0.125, 0.15), (0.0, 0.10, 0.15), (0.5, 1.0), (0, 20, 60)):
        trigger, span, review, crash, trailing, derisk, cooldown = values
        if trigger == "ema" and crash > 0:
            continue
        if trigger == "crash" and span != 100:
            continue
        configs.append({"trigger": trigger, "ema_span": span, "review": review, "crash_threshold": crash, "trailing_stop": trailing, "derisk": derisk, "cooldown": cooldown})
    # Fixed seed keeps the broad parameter sample reproducible without making
    # the exhaustive Cartesian product prohibitively slow.
    if len(configs) > 300:
        rng = np.random.default_rng(20260817)
        configs = [configs[index] for index in sorted(rng.choice(len(configs), 300, replace=False))]
    all_results = []
    for row in shortlist:
        allocation = row["allocation"]
        base = _method_returns(prices, allocation, row["method"]).loc[START:END].dropna()
        assets = prices.loc[base.index, list(allocation)].pct_change().fillna(0.0)
        baseline = _period_metrics(base)
        for config in configs:
            protected = _overlay(base, assets, cash, config)
            result = {"allocation_id": row["allocation_id"], "expert_rank": row["expert_rank"], "method": row["method"], "allocation": row["allocation"], "config": config, "metrics": _period_metrics(protected)}
            result["selection_score"] = _score(result, baseline)
            result["test_cagr_delta"] = result["metrics"]["test"]["cagr"] - baseline["test"]["cagr"]
            result["test_dd_delta"] = result["metrics"]["test"]["max_drawdown"] - baseline["test"]["max_drawdown"]
            result["baseline_test"] = baseline["test"]
            all_results.append(result)
    all_results.sort(key=lambda item: item["selection_score"], reverse=True)
    robust = [row for row in all_results if row["metrics"]["test"]["cagr"] >= row["baseline_test"]["cagr"] - 0.01 and row["test_dd_delta"] >= 0]
    robust.sort(key=lambda item: (item["test_cagr_delta"] + 0.35 * item["test_dd_delta"], item["selection_score"]), reverse=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in robust:
        key = json.dumps(row["config"], sort_keys=True)
        grouped.setdefault(key, []).append(row)
    consensus = []
    for key, rows in grouped.items():
        if len(rows) < 4:
            continue
        consensus.append(
            {
                "config": json.loads(key),
                "candidate_count": len(rows),
                "mean_selection_score": float(np.mean([row["selection_score"] for row in rows])),
                "median_test_cagr_delta": float(np.median([row["test_cagr_delta"] for row in rows])),
                "median_test_dd_delta": float(np.median([row["test_dd_delta"] for row in rows])),
                "mean_test_cagr_delta": float(np.mean([row["test_cagr_delta"] for row in rows])),
                "mean_test_dd_delta": float(np.mean([row["test_dd_delta"] for row in rows])),
                "examples": rows[:5],
            }
        )
    consensus.sort(key=lambda row: (row["median_test_cagr_delta"] + 0.35 * row["median_test_dd_delta"], row["mean_selection_score"]), reverse=True)
    cost_sensitivity = []
    for consensus_row in consensus[:5]:
        config = consensus_row["config"]
        for row in shortlist:
            allocation = row["allocation"]
            base = _method_returns(prices, allocation, row["method"]).loc[START:END].dropna()
            assets = prices.loc[base.index, list(allocation)].pct_change().fillna(0.0)
            baseline_test = _metrics(base.loc["2021-01-01":])
            for cost_bps in (0.0, 10.0, 25.0):
                protected = _overlay(base, assets, cash, config, cost_bps=cost_bps)
                test = _metrics(protected.loc["2021-01-01":])
                cost_sensitivity.append({"config": config, "cost_bps": cost_bps, "cagr_delta": test["cagr"] - baseline_test["cagr"], "dd_delta": test["max_drawdown"] - baseline_test["max_drawdown"]})
    cost_frame = pd.DataFrame(cost_sensitivity)
    cost_summary = []
    for config_key, rows in cost_frame.groupby(cost_frame["config"].map(lambda value: json.dumps(value, sort_keys=True))):
        for cost_bps, cost_rows in rows.groupby("cost_bps"):
            cost_summary.append({"config": json.loads(config_key), "cost_bps": float(cost_bps), "median_cagr_delta": float(cost_rows["cagr_delta"].median()), "median_dd_delta": float(cost_rows["dd_delta"].median())})
    output = {"permutations_per_candidate": len(configs), "total_permutations": len(all_results), "selection": "Pre-2021 train/validation score; final rankings require no more than 1 percentage point CAGR loss and non-worse locked test drawdown. Consensus rules must pass on at least four shortlisted portfolios.", "top_robust": robust[:100], "top_selection_score": all_results[:100], "consensus": consensus[:100], "cost_sensitivity": cost_summary}
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for rank, item in enumerate(robust[:100], 1):
        rows.append({"rank": rank, "allocation_id": item["allocation_id"], "method": item["method"], **item["config"], "test_cagr": item["metrics"]["test"]["cagr"], "test_max_drawdown": item["metrics"]["test"]["max_drawdown"], "test_cagr_delta": item["test_cagr_delta"], "test_dd_delta": item["test_dd_delta"], "selection_score": item["selection_score"]})
    for rank, item in enumerate(consensus[:25], 1):
        rows.append({"rank": f"consensus-{rank}", "allocation_id": "", "method": "CONSENSUS", **item["config"], "test_cagr": "", "test_max_drawdown": "", "test_cagr_delta": item["median_test_cagr_delta"], "test_dd_delta": item["median_test_dd_delta"], "selection_score": item["mean_selection_score"]})
    pd.DataFrame(rows).to_csv(CSV, index=False)
    lines = [
        "# Expert Protection Grid Report",
        "",
        f"Evaluated **{len(all_results):,}** portfolio/configuration permutations across the 10 expert candidates.",
        "Signals use prior closing data and execute on the next session. The 2015-2020 period is used for train/validation; 2021-2026 is the locked test period.",
        "",
        "## Search Design",
        "",
        "The grid varied EMA span, review frequency, component crash threshold, trailing stop, de-risk fraction, trigger combination, and cooldown. A fixed-seed sample of 300 configurations was applied to each portfolio.",
        "",
        "The consensus layer requires a rule to pass the robust filter on at least four portfolios. Robust filtering allows no more than one percentage point of test CAGR loss and requires non-worse test drawdown.",
        "",
        "## Consensus Rules",
        "",
        "| Rank | Trigger | Crash | Review | Cooldown | De-risk | Portfolios | Median CAGR Delta | Median DD Improvement |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(consensus[:10], 1):
        config = item["config"]
        lines.append(f"| {rank} | {config['trigger']} | {config['crash_threshold']:.1%} | {config['review']} | {config['cooldown']} days | {config['derisk']:.0%} | {item['candidate_count']} | {item['median_test_cagr_delta']:+.2%} | {item['median_test_dd_delta']:+.2%} |")
    lines.extend(["", "The preferred compromise is the rule that passes on the most candidates while keeping median CAGR impact near zero and improving drawdown. It is not selected because it has the highest isolated backtest CAGR.", "", "## Best Robust Result Per Candidate", "", "| Expert Rank | Portfolio | Method | Protection | Test CAGR Delta | Test DD Improvement |", "|---:|---|---|---|---:|---:|"])
    best_by_candidate = {}
    for item in robust:
        best_by_candidate.setdefault(item["expert_rank"], item)
    for expert_rank in sorted(best_by_candidate):
        item = best_by_candidate[expert_rank]
        lines.append(f"| {expert_rank} | Allocation {item['allocation_id']} | `{item['method']}` | `{json.dumps(item['config'], sort_keys=True)}` | {item['test_cagr_delta']:+.2%} | {item['test_dd_delta']:+.2%} |")
    lines.extend(["", "## Cost Sensitivity", "", "The top consensus rule's median test CAGR delta across the evaluated candidates is reported below:", "", "| Cost | Median CAGR Delta | Median DD Improvement |", "|---:|---:|---:|"])
    top_config = consensus[1]["config"] if len(consensus) > 1 else (consensus[0]["config"] if consensus else {})
    for item in cost_summary:
        if item["config"] == top_config:
            lines.append(f"| {item['cost_bps']:.0f} bps | {item['median_cagr_delta']:+.2%} | {item['median_dd_delta']:+.2%} |")
    lines.extend(["", "## Caveats", "", "- This is a sampled grid, not an exhaustive proof of the best future rule.", "- The expert top 10 were selected from the same historical universe, so portfolio selection remains subject to selection bias.", "- Component crash rules treat every allocated ticker as active, including assets that a dynamic EMA method may have sold; this is conservative but can over-trigger.", "- Results exclude taxes, slippage beyond the sensitivity table, and investor execution errors."])
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"configs_per_candidate": len(configs), "total_permutations": len(all_results), "robust_results": len(robust), "consensus_configs": len(consensus)}, indent=2))


if __name__ == "__main__":
    main()
