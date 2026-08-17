"""Streamlit view for the validated allocation screener output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from screener.results import load_screened_results


RESULTS_PATH = Path(__file__).parents[2] / "data" / "screener-results.json"
BENCHMARK_PATH = Path(__file__).parents[2] / "data" / "screener-benchmark.json"


def _allocation_label(allocation: dict[str, float]) -> str:
    return " / ".join(f"{ticker} {weight:.1%}" for ticker, weight in allocation.items())


def _table_rows(results: list[dict[str, object]], benchmark: dict[str, object]) -> list[dict[str, object]]:
    spy_test_cagr = float(benchmark["periods"]["test"]["cagr"])
    rows = []
    for rank, result in enumerate(results, start=1):
        local = result["local"]
        testfol = result["testfol"]
        testfol_stats = (testfol.get("testfol_stats") or [{}])[0]
        walk_forward = local["walk_forward"]
        rows.append(
            {
                "Rank": rank,
                "Allocation": _allocation_label(local["allocation"]),
                "Rebalance": local["rebalance_freq"],
                "Factors": local["diversification"]["factor_breadth"],
                "Local Test CAGR": f"{walk_forward['test']['cagr']:.1%}",
                "Local Test DD": f"{walk_forward['test']['max_drawdown']:.1%}",
                "Testfol CAGR": f"{float(testfol_stats.get('cagr', 0.0)):.2f}%",
                "vs SPY CAGR": f"{float(testfol_stats.get('cagr', 0.0)) - spy_test_cagr * 100.0:+.2f} pp",
                "Testfol Max DD": f"{float(testfol_stats.get('max_drawdown', 0.0)):.2f}%",
                "Observations": testfol.get("testfol_observations", 0),
            }
        )
    return rows


def render_screened_results(
    path: str | Path = RESULTS_PATH,
    benchmark_path: str | Path = BENCHMARK_PATH,
) -> None:
    """Render accepted screener allocations and their validation evidence."""
    st.title("Screened Allocations")
    st.caption("Allocations that passed local diversification, walk-forward, and Testfol gates.")
    try:
        results = load_screened_results(path)
        benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.info(f"No screened results found at `{path}`. Run the research pipeline first.")
        return
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not load screened results: {error}")
        return

    if not results:
        st.info("No allocations currently pass all screener gates.")
        return

    st.metric("Accepted allocations", len(results))
    st.subheader("Locked SPY Baseline")
    st.caption(
        f"SPY benchmark from {benchmark['date_range']['start']} to {benchmark['date_range']['end']} "
        f"({benchmark['source']}). Portfolio CAGR comparisons use the test window."
    )
    baseline_columns = st.columns(3)
    for column, period in zip(baseline_columns, ("train", "validation", "test")):
        metrics = benchmark["periods"][period]
        column.metric(f"SPY {period.title()} CAGR", f"{float(metrics['cagr']):.1%}")
        column.caption(f"Max DD {float(metrics['max_drawdown']):.1%} | Sharpe {float(metrics['sharpe']):.2f}")
    st.dataframe(pd.DataFrame(_table_rows(results, benchmark)), use_container_width=True, hide_index=True)

    labels = [_allocation_label(result["local"]["allocation"]) for result in results]
    selected_label = st.selectbox("Allocation details", labels)
    selected = results[labels.index(selected_label)]
    local = selected["local"]
    testfol = selected["testfol"]
    stats = (testfol.get("testfol_stats") or [{}])[0]
    spy_test = benchmark["periods"]["test"]

    st.subheader("Selected Allocation")
    st.write(selected_label)
    metric_columns = st.columns(8)
    metric_columns[0].metric("Rebalance", local["rebalance_freq"])
    metric_columns[1].metric("Factors", local["diversification"]["factor_breadth"])
    metric_columns[2].metric("Local Test CAGR", f"{local['walk_forward']['test']['cagr']:.1%}")
    metric_columns[3].metric("Local Test DD", f"{local['walk_forward']['test']['max_drawdown']:.1%}")
    metric_columns[4].metric("Testfol CAGR", f"{float(stats.get('cagr', 0.0)):.2f}%")
    metric_columns[5].metric("Testfol Max DD", f"{float(stats.get('max_drawdown', 0.0)):.2f}%")
    metric_columns[6].metric("SPY Test CAGR", f"{float(spy_test['cagr']):.2%}")
    metric_columns[7].metric(
        "Excess vs SPY",
        f"{float(stats.get('cagr', 0.0)) / 100.0 - float(spy_test['cagr']):+.2%}",
    )

    with st.expander("Allocation weights"):
        st.dataframe(
            pd.DataFrame(
                [{"Ticker": ticker, "Weight": weight} for ticker, weight in local["allocation"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("Validation details"):
        st.json(
            {
                "factor_variance_share": local["diversification"]["factor_variance_share"],
                "stress_correlations": local["diversification"]["stress_correlations"],
                "walk_forward": local["walk_forward"],
                "testfol_observations": testfol.get("testfol_observations"),
                "testfol_start": testfol.get("testfol_start"),
                "testfol_end": testfol.get("testfol_end"),
            }
        )
