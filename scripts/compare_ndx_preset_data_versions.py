#!/usr/bin/env python3
"""Compare every saved preset with the pre- and post-reconstruction NDX data.

The comparison deliberately holds the application engine, financing series,
market data, costs, and rebalance rules constant. Only the NDX-derived inputs
are switched between the preserved baseline and the rebuilt data set.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.backtest_orchestrator import run_single_backtest
from app.services import data_service
from app.services import ndx_mega_rotation as rotation


REQUEST_START = "1900-01-01"
SENSITIVE_SERIES = {"NDXMEGASIM", "NDXMEGA2SIM", "NDX30SIM", "QQUPSIM"}


@dataclass(frozen=True)
class VersionContext:
    name: str
    sim_dir: Path
    weights_path: Path
    constituents_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_ticker(ticker: str) -> str:
    return str(ticker).split("?", 1)[0].strip().upper()


def _load_presets() -> list[dict[str, Any]]:
    with (REPO_ROOT / "data" / "presets.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_preset(preset: dict[str, Any]) -> dict[str, Any]:
    allocation: dict[str, float] = {}
    maint: dict[str, float] = {}
    pm_maint: dict[str, float] = {}
    for row in preset["allocation"]:
        ticker = str(row["Ticker"]).strip()
        allocation[ticker] = allocation.get(ticker, 0.0) + float(row["Weight %"])
        base = _base_ticker(ticker)
        maint[base] = float(row.get("Maint %", 25.0) or 0.0)
        pm_value = row.get("PM Maint %", 0.0)
        pm_maint[base] = 0.0 if pd.isna(pm_value) else float(pm_value or 0.0)

    rebalance = dict(preset.get("rebalance", {}))
    month_lookup = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    rebalance["mode"] = rebalance.get("mode", "Standard")
    rebalance["freq"] = rebalance.get("freq", "Yearly")
    rebalance["month"] = month_lookup.get(rebalance.get("month_str", "Jan"), 1)
    rebalance["day"] = int(rebalance.get("day", 1))
    rebalance["threshold_pct"] = float(rebalance.get("threshold_pct", 5.0))

    original_bases = {_base_ticker(ticker) for ticker in allocation}
    dependencies: list[str] = []
    if "NDXMEGASIM" in original_bases:
        dependencies.append("NDXMEGASIM")
    if "QQUPSIM" in original_bases:
        dependencies.append("QQUPSIM<-NDXMEGAPRICESIM")
    if "NDXMEGA2SIM" in original_bases:
        dependencies.append("NDXMEGA2SIM")
    if "NDX30SIM" in original_bases:
        dependencies.append("NDX30SIM")
    if any(base.startswith("NDXMEGA_TOP") for base in original_bases):
        dependencies.append("NDXMEGA annual selection")
    if any(base.startswith("NDX_TOP") for base in original_bases):
        dependencies.append("NDX annual selection")

    return {
        "name": preset["name"],
        "category": preset.get("category", "Uncategorized"),
        "allocation": allocation,
        "maint": maint,
        "pm_maint": pm_maint,
        "rebalance": rebalance,
        "dependency": "; ".join(dependencies) if dependencies else "None",
        "ndx_dependent": bool(dependencies),
    }


def _clear_rotation_caches() -> None:
    for function_name in (
        "load_ndxmega_constituents",
        "load_ndx_quarterly_weights",
        "load_ndx_official_membership",
    ):
        function = getattr(rotation, function_name)
        if hasattr(function, "cache_clear"):
            function.cache_clear()


def _activate_version(context: VersionContext) -> None:
    global _active_sim_dir
    _active_sim_dir = context.sim_dir
    rotation._ndx_weights_path = lambda: context.weights_path
    rotation._constituents_path = lambda: context.constituents_path
    _clear_rotation_caches()


def _collect_plans(
    context: VersionContext,
    presets: list[dict[str, Any]],
    end_date: str,
) -> dict[str, Any]:
    _activate_version(context)
    plans: dict[str, Any] = {}
    for preset in presets:
        plans[preset["name"]] = rotation.build_dynamic_allocation_plan(
            preset["allocation"],
            preset["maint"],
            preset["pm_maint"],
            REQUEST_START,
            end_date,
        )
    return plans


def _universe_bases(plans: dict[str, Any]) -> set[str]:
    bases: set[str] = set()
    for plan in plans.values():
        bases.update(_base_ticker(ticker) for ticker in plan.universe_tickers)
    return bases


def _fetch_prices(bases: set[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not bases:
        return pd.DataFrame()
    frame = data_service.fetch_component_data(
        sorted(bases),
        start_date,
        end_date,
        sync_end=False,
    )
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame.sort_index()


def _version_price_frame(
    context: VersionContext,
    all_bases: set[str],
    common_prices: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    _activate_version(context)
    sensitive_bases = all_bases & SENSITIVE_SERIES
    sensitive_prices = _fetch_prices(sensitive_bases, start_date, end_date)
    # Hold the trading calendar fixed.  A version-specific simulation can have
    # a few extra/missing rows; taking the concat union changes the engine's
    # calendar even for an unrelated SPY-only preset.
    sensitive_prices = sensitive_prices.reindex(common_prices.index)
    frame = pd.concat([common_prices, sensitive_prices], axis=1).sort_index()
    frame = frame.loc[:, ~frame.columns.duplicated(keep="last")]
    missing = sorted(base for base in all_bases if base not in frame or frame[base].dropna().empty)
    for base in missing:
        if base not in frame:
            frame[base] = np.nan
    return frame.sort_index(), missing


def _force_local_engine(**_: Any):
    raise requests.exceptions.ConnectionError("Preset comparison forces a common local engine")


def _requested_start(plan: Any) -> str:
    if not plan.dynamic_schedule:
        return REQUEST_START
    return min(plan.dynamic_schedule).strftime("%Y-%m-%d")


def _run_presets(
    context: VersionContext,
    presets: list[dict[str, Any]],
    plans: dict[str, Any],
    prices: pd.DataFrame,
    end_date: str,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    _activate_version(context)
    series_by_name: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}

    for number, preset in enumerate(presets, start=1):
        name = preset["name"]
        start_date = _requested_start(plans[name])
        print(f"[{context.name} {number:02d}/{len(presets)}] {name} ({start_date}..{end_date})")
        try:
            needed_bases = sorted(
                {_base_ticker(ticker) for ticker in plans[name].universe_tickers}
            )
            preset_prices = prices.reindex(columns=needed_bases).copy(deep=True)
            result = run_single_backtest(
                allocation=copy.deepcopy(preset["allocation"]),
                maint_pcts=copy.deepcopy(preset["maint"]),
                rebalance=copy.deepcopy(preset["rebalance"]),
                start_date=start_date,
                end_date=end_date,
                start_val=10_000.0,
                cashflow_amount=0.0,
                cashflow_freq="Monthly",
                invest_div=True,
                pay_down_margin=False,
                tax_config={},
                bearer_token=None,
                name=name,
                fetch_backtest_fn=_force_local_engine,
                pm_maint_pcts=copy.deepcopy(preset["pm_maint"]),
                # The local engine derives leveraged/synthetic columns in
                # place.  Sharing this frame lets an earlier preset alter the
                # inputs seen by every later preset, which can make SPY-only
                # appear sensitive to an NDX reconstruction.
                prefetched_component_prices=preset_prices,
            )
            series = pd.to_numeric(result["port_series"], errors="coerce").dropna()
            series.index = pd.to_datetime(series.index)
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            series = series[~series.index.duplicated(keep="last")].sort_index()
            series = series.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
            if len(series) < 2:
                raise ValueError("backtest returned fewer than two observations")
            series_by_name[name] = series
        except Exception as exc:  # keep the full 28-preset matrix even if one input is unavailable
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {errors[name]}")

    return series_by_name, errors


def _metrics(series: pd.Series) -> dict[str, float]:
    normalized = series / float(series.iloc[0])
    daily = normalized.pct_change(fill_method=None).dropna()
    years = max((normalized.index[-1] - normalized.index[0]).days / 365.2425, 1 / 365.2425)
    total_return = float(normalized.iloc[-1] - 1.0)
    cagr = float(normalized.iloc[-1] ** (1.0 / years) - 1.0)
    drawdown = normalized / normalized.cummax() - 1.0
    volatility = float(daily.std() * math.sqrt(252)) if len(daily) > 1 else float("nan")
    sharpe = (
        float(daily.mean() / daily.std() * math.sqrt(252))
        if len(daily) > 1 and daily.std() > 0
        else float("nan")
    )
    return {
        "years": years,
        "terminal_10k": float(10_000.0 * normalized.iloc[-1]),
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "volatility_pct": volatility * 100.0,
        "sharpe_zero_rf": sharpe,
    }


def _compare(
    presets: list[dict[str, Any]],
    before_series: dict[str, pd.Series],
    after_series: dict[str, pd.Series],
    before_errors: dict[str, str],
    after_errors: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    curve_columns: dict[str, pd.Series] = {}

    for preset in presets:
        name = preset["name"]
        if name not in before_series or name not in after_series:
            rows.append(
                {
                    "Preset": name,
                    "Category": preset["category"],
                    "NDX Dependency": preset["dependency"],
                    "Expected NDX Impact": preset["ndx_dependent"],
                    "Status": before_errors.get(name) or after_errors.get(name) or "Missing series",
                }
            )
            continue

        paired = pd.concat(
            [before_series[name].rename("before"), after_series[name].rename("after")],
            axis=1,
            join="inner",
        ).dropna()
        if len(paired) < 2:
            rows.append(
                {
                    "Preset": name,
                    "Category": preset["category"],
                    "NDX Dependency": preset["dependency"],
                    "Expected NDX Impact": preset["ndx_dependent"],
                    "Status": "No overlapping before/after observations",
                }
            )
            continue

        paired["before"] = paired["before"] / paired["before"].iloc[0] * 10_000.0
        paired["after"] = paired["after"] / paired["after"].iloc[0] * 10_000.0
        before_metrics = _metrics(paired["before"])
        after_metrics = _metrics(paired["after"])
        daily = paired.pct_change(fill_method=None).dropna()
        return_difference = daily["after"] - daily["before"]
        max_return_difference = float(return_difference.abs().max()) if not return_difference.empty else 0.0
        unchanged = max_return_difference <= 1e-12

        curve_columns[f"Before | {name}"] = paired["before"]
        curve_columns[f"After | {name}"] = paired["after"]
        rows.append(
            {
                "Preset": name,
                "Category": preset["category"],
                "NDX Dependency": preset["dependency"],
                "Expected NDX Impact": preset["ndx_dependent"],
                "Status": "Unchanged" if unchanged else "Changed",
                "Start": paired.index[0].date().isoformat(),
                "End": paired.index[-1].date().isoformat(),
                "Observations": len(paired),
                "Years": before_metrics["years"],
                "Before Final $10k": before_metrics["terminal_10k"],
                "After Final $10k": after_metrics["terminal_10k"],
                "Final Value Delta %": (after_metrics["terminal_10k"] / before_metrics["terminal_10k"] - 1.0) * 100.0,
                "Before Total Return %": before_metrics["total_return_pct"],
                "After Total Return %": after_metrics["total_return_pct"],
                "Total Return Delta pp": after_metrics["total_return_pct"] - before_metrics["total_return_pct"],
                "Before CAGR %": before_metrics["cagr_pct"],
                "After CAGR %": after_metrics["cagr_pct"],
                "CAGR Delta pp": after_metrics["cagr_pct"] - before_metrics["cagr_pct"],
                "Before Max Drawdown %": before_metrics["max_drawdown_pct"],
                "After Max Drawdown %": after_metrics["max_drawdown_pct"],
                "Max Drawdown Delta pp": after_metrics["max_drawdown_pct"] - before_metrics["max_drawdown_pct"],
                "Before Volatility %": before_metrics["volatility_pct"],
                "After Volatility %": after_metrics["volatility_pct"],
                "Volatility Delta pp": after_metrics["volatility_pct"] - before_metrics["volatility_pct"],
                "Before Sharpe (0% RF)": before_metrics["sharpe_zero_rf"],
                "After Sharpe (0% RF)": after_metrics["sharpe_zero_rf"],
                "Daily Return Correlation": float(daily["before"].corr(daily["after"])),
                "Before/After Tracking Difference %": float(return_difference.std() * math.sqrt(252) * 100.0),
                "Max Absolute Daily Return Difference %": max_return_difference * 100.0,
            }
        )

    comparison = pd.DataFrame(rows)
    curves = pd.concat(curve_columns, axis=1).sort_index() if curve_columns else pd.DataFrame()
    return comparison, curves


def _write_chart(comparison: pd.DataFrame, path: Path) -> None:
    changed = comparison[comparison["Status"].eq("Changed")].copy()
    changed = changed.sort_values("CAGR Delta pp")
    if changed.empty:
        return

    height = max(6.0, 0.48 * len(changed))
    figure, axes = plt.subplots(1, 2, figsize=(16, height), constrained_layout=True)
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in changed["CAGR Delta pp"]]
    axes[0].barh(changed["Preset"], changed["CAGR Delta pp"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("CAGR change (after minus before)")
    axes[0].set_xlabel("Percentage points")

    final_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in changed["Final Value Delta %"]]
    axes[1].barh(changed["Preset"], changed["Final Value Delta %"], color=final_colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Terminal $10k change")
    axes[1].set_xlabel("Percent")
    figure.suptitle("Preset impact of the rebuilt Nasdaq-100 reconstruction", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_html_table(comparison: pd.DataFrame) -> str:
    selected = comparison[
        [
            "Preset",
            "Status",
            "Start",
            "End",
            "Before Final $10k",
            "After Final $10k",
            "Final Value Delta %",
            "Before CAGR %",
            "After CAGR %",
            "CAGR Delta pp",
            "Before Max Drawdown %",
            "After Max Drawdown %",
            "Max Drawdown Delta pp",
            "NDX Dependency",
        ]
    ].copy()
    money_columns = ["Before Final $10k", "After Final $10k"]
    percent_columns = [
        "Final Value Delta %",
        "Before CAGR %",
        "After CAGR %",
        "CAGR Delta pp",
        "Before Max Drawdown %",
        "After Max Drawdown %",
        "Max Drawdown Delta pp",
    ]
    for column in money_columns:
        selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"${value:,.0f}")
    for column in percent_columns:
        selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{value:+.2f}%")
    return selected.to_html(index=False, escape=True, classes="comparison")


def _write_report(
    comparison: pd.DataFrame,
    output_dir: Path,
    before: VersionContext,
    after: VersionContext,
    before_missing: list[str],
    after_missing: list[str],
) -> None:
    changed = comparison[comparison["Status"].eq("Changed")]
    unchanged = comparison[comparison["Status"].eq("Unchanged")]
    errors = comparison[~comparison["Status"].isin(["Changed", "Unchanged"])]
    table = _format_html_table(comparison)
    summary = {
        "preset_count": int(len(comparison)),
        "changed_count": int(len(changed)),
        "unchanged_count": int(len(unchanged)),
        "error_count": int(len(errors)),
        "before_weights_sha256": _sha256(before.weights_path),
        "after_weights_sha256": _sha256(after.weights_path),
        "before_missing_price_series": before_missing,
        "after_missing_price_series": after_missing,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    html_document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NDX preset reconstruction before/after</title>
  <style>
    body {{ background:#0e1117; color:#e6e9ef; font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:32px; }}
    h1,h2 {{ color:#fff; }}
    .summary {{ display:flex; gap:16px; flex-wrap:wrap; margin:20px 0; }}
    .card {{ background:#171c24; border:1px solid #2b3340; border-radius:10px; padding:14px 18px; min-width:130px; }}
    .value {{ font-size:26px; font-weight:700; }}
    img {{ max-width:100%; background:white; border-radius:8px; }}
    table {{ border-collapse:collapse; width:100%; background:#131821; }}
    th {{ position:sticky; top:0; background:#222a36; z-index:1; }}
    th,td {{ border:1px solid #303846; padding:8px; text-align:right; white-space:nowrap; }}
    th:first-child,td:first-child,th:last-child,td:last-child {{ text-align:left; }}
    tr:hover {{ background:#202733; }}
    code {{ color:#a9d6ff; }}
    .note {{ color:#b7bfca; line-height:1.5; }}
  </style>
</head>
<body>
  <h1>NDX reconstruction impact on all presets</h1>
  <p class="note">Identical current engine, financing data, expenses, leverage, and rebalance rules. Each preset uses its longest overlapping before/after history. Every equity curve is normalized to $10,000 on the first shared date.</p>
  <div class="summary">
    <div class="card"><div class="value">{len(comparison)}</div>presets tested</div>
    <div class="card"><div class="value">{len(changed)}</div>changed</div>
    <div class="card"><div class="value">{len(unchanged)}</div>unchanged</div>
    <div class="card"><div class="value">{len(errors)}</div>errors</div>
  </div>
  <h2>Material impact</h2>
  <img src="preset_impact.png" alt="CAGR and terminal value changes">
  <h2>Full 28-preset matrix</h2>
  {table}
  <h2>Provenance</h2>
  <p class="note">Before weights SHA-256: <code>{html.escape(summary['before_weights_sha256'])}</code><br>
  After weights SHA-256: <code>{html.escape(summary['after_weights_sha256'])}</code><br>
  Missing common price histories were held constant across both versions. See <code>summary.json</code> for their identities.</p>
</body>
</html>
"""
    (output_dir / "preset_comparison.html").write_text(html_document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO_ROOT / "exports" / "ndx_preset_before_after_2026-08-07" / "baseline_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "exports" / "ndx_preset_before_after_2026-08-07",
    )
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = VersionContext(
        name="before",
        sim_dir=baseline_dir,
        weights_path=baseline_dir / "nasdaq_quarterly_weights.csv",
        constituents_path=baseline_dir / "results" / "ndx_mega_constituents.csv",
    )
    after = VersionContext(
        name="after",
        sim_dir=REPO_ROOT / "data",
        weights_path=REPO_ROOT / "data" / "ndx_simulation" / "data" / "results" / "nasdaq_quarterly_weights.csv",
        constituents_path=REPO_ROOT / "data" / "ndx_simulation" / "data" / "results" / "ndx_mega_constituents.csv",
    )
    required_paths = [
        before.weights_path,
        before.constituents_path,
        before.sim_dir / "NDXMEGASIM.csv",
        before.sim_dir / "NDXMEGAPRICESIM.csv",
        after.weights_path,
        after.constituents_path,
        after.sim_dir / "NDXMEGASIM.csv",
        after.sim_dir / "NDXMEGAPRICESIM.csv",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Required comparison inputs are missing: " + ", ".join(missing_paths))

    current_ndx = pd.read_csv(after.sim_dir / "NDXMEGASIM.csv", parse_dates=["Date"])
    end_date = args.end_date or current_ndx["Date"].max().date().isoformat()
    presets = [_normalize_preset(preset) for preset in _load_presets()]
    print(f"Comparing {len(presets)} presets through {end_date}")

    # Bypass the normal disk result cache: its key intentionally does not encode
    # which reconstruction snapshot is active.
    data_service.cache_get = lambda *args, **kwargs: None
    data_service.cache_set = lambda *args, **kwargs: None

    global _active_sim_dir
    _active_sim_dir = after.sim_dir
    original_read_csv = pd.read_csv

    def versioned_read_csv(path, *read_args, **read_kwargs):
        key = os.fspath(path)
        replacements = {
            "data/NDXMEGASIM.csv": _active_sim_dir / "NDXMEGASIM.csv",
            "data/NDXMEGAPRICESIM.csv": _active_sim_dir / "NDXMEGAPRICESIM.csv",
            "data/NDXMEGA2SIM.csv": _active_sim_dir / "NDXMEGA2SIM.csv",
            "data/NDX30SIM.csv": _active_sim_dir / "NDX30SIM.csv",
        }
        return original_read_csv(replacements.get(key, path), *read_args, **read_kwargs)

    pd.read_csv = versioned_read_csv
    data_service.pd.read_csv = versioned_read_csv
    rotation.pd.read_csv = versioned_read_csv

    try:
        before_plans = _collect_plans(before, presets, end_date)
        after_plans = _collect_plans(after, presets, end_date)
        all_bases = _universe_bases(before_plans) | _universe_bases(after_plans)
        common_bases = all_bases - SENSITIVE_SERIES
        print(f"Fetching one shared market-data universe: {len(common_bases)} common bases")
        _activate_version(after)
        common_prices = _fetch_prices(common_bases, REQUEST_START, end_date)
        before_prices, before_missing = _version_price_frame(
            before, all_bases, common_prices, REQUEST_START, end_date
        )
        after_prices, after_missing = _version_price_frame(
            after, all_bases, common_prices, REQUEST_START, end_date
        )
        print(f"Price coverage: before missing={len(before_missing)}, after missing={len(after_missing)}")

        before_series, before_errors = _run_presets(
            before, presets, before_plans, before_prices, end_date
        )
        after_series, after_errors = _run_presets(
            after, presets, after_plans, after_prices, end_date
        )
        comparison, curves = _compare(
            presets,
            before_series,
            after_series,
            before_errors,
            after_errors,
        )
    finally:
        pd.read_csv = original_read_csv
        data_service.pd.read_csv = original_read_csv
        rotation.pd.read_csv = original_read_csv

    comparison.to_csv(output_dir / "preset_comparison.csv", index=False)
    curves.to_csv(output_dir / "preset_equity_curves.csv.gz", compression="gzip")
    _write_chart(comparison, output_dir / "preset_impact.png")
    _write_report(comparison, output_dir, before, after, before_missing, after_missing)

    print("\nComparison complete")
    print(comparison["Status"].value_counts(dropna=False).to_string())
    print(f"CSV: {output_dir / 'preset_comparison.csv'}")
    print(f"HTML: {output_dir / 'preset_comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
