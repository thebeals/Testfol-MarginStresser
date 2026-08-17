"""Streamlit view for the validated allocation screener output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from screener.results import load_screened_results


RESULTS_PATH = Path(__file__).parents[2] / "data" / "screener-results.json"
BENCHMARK_PATH = Path(__file__).parents[2] / "data" / "screener-benchmark.json"
REBALANCE_PATH = Path(__file__).parents[2] / "data" / "rebalance-ranking.json"
MATRIX_PATH = Path(__file__).parents[2] / "data" / "rebalance-matrix-results.jsonl"


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
                "Testfol CAGR (history)": f"{float(testfol_stats.get('cagr', 0.0)):.2f}%",
                "Local Test vs SPY": f"{(walk_forward['test']['cagr'] - spy_test_cagr):+.2%}",
                "Testfol Max DD": f"{float(testfol_stats.get('max_drawdown', 0.0)):.2f}%",
                "Observations": testfol.get("testfol_observations", 0),
            }
        )
    return rows


def _load_matrix(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _matrix_rows(matrix: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "Allocation": row["allocation_id"],
            "Method": row["method"],
            "CAGR": f"{float(row['cagr']):.2%}",
            "Max DD": f"{float(row['max_drawdown']):.2%}",
            "Sharpe": f"{float(row['sharpe']):.2f}",
        }
        for row in matrix
    ]


def render_screened_results(
    path: str | Path = RESULTS_PATH,
    benchmark_path: str | Path = BENCHMARK_PATH,
    rebalance_path: str | Path = REBALANCE_PATH,
) -> None:
    """Render accepted screener allocations and their validation evidence."""
    st.title("Screened Allocations")
    st.caption("Allocations that passed local diversification, walk-forward, and Testfol gates.")
    try:
        results = load_screened_results(path)
        benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
        rebalance_ranking = json.loads(Path(rebalance_path).read_text(encoding="utf-8"))
        matrix = _load_matrix(MATRIX_PATH)
    except FileNotFoundError:
        st.info(f"No screened results found at `{path}`. Run the research pipeline first.")
        return
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not load screened results: {error}")
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
    if not results:
        st.warning("No allocations currently pass all gates, including the requirement to beat SPY in the test window.")
        return

    st.dataframe(pd.DataFrame(_table_rows(results, benchmark)), use_container_width=True, hide_index=True)

    st.subheader("Rebalance Method Ranking")
    st.caption("All 10 allocations are retained. Local best and Testfol-compatible best methods are shown separately.")
    rebalance_rows = []
    for row in rebalance_ranking.get("allocations", []):
        local_best = row["best_local"]
        testfol_best = row["best_testfol"]
        rebalance_rows.append(
            {
                "Rank": row["rank"],
                "Allocation": _allocation_label(row["allocation"]),
                "Best Local Method": local_best["method"],
                "Local CAGR": f"{float(local_best['cagr']):.2%}",
                "Best Testfol Method": testfol_best["method"],
                "Testfol Test CAGR": f"{float(testfol_best['testfol_test'].get('cagr', 0.0)):.2%}",
                "Testfol Max DD": f"{float(testfol_best['testfol_stats'].get('max_drawdown', 0.0)):.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(rebalance_rows), use_container_width=True, hide_index=True)

    st.subheader("Full Rebalance Matrix")
    st.caption(
        f"All {len(matrix)} local tests across the 10 allocations. "
        "EMA results are local dynamic-signal tests; band/calendar results are the Testfol-compatible classes."
    )
    matrix_allocation = st.selectbox(
        "Allocation matrix filter",
        ["All allocations"] + [str(index) for index in range(1, len(results) + 1)],
        key="rebalance_matrix_allocation",
    )
    visible_matrix = matrix if matrix_allocation == "All allocations" else [
        row for row in matrix if str(row["allocation_id"]) == matrix_allocation
    ]
    st.dataframe(pd.DataFrame(_matrix_rows(visible_matrix)), use_container_width=True, hide_index=True)

    with st.expander("Signal and funding rules tested"):
        st.markdown(
            """
            **Relative bands**: at the scheduled monthly, quarterly, or yearly check, rebalance when a holding deviates by 5%, 10%, or 20% of its target weight.

            **Absolute bands**: at the scheduled check, rebalance when a holding deviates by 5, 10, or 20 percentage points from target.

            **EMA100 / EMA200**: use each asset's own adjusted daily price. A prior-day close at or above the EMA is a buy/active signal; below the EMA is a sell/inactive signal. Signals are evaluated only at the configured monthly, quarterly, or yearly check.

            **cash_only**: starts with a 10% SHV reserve. Sell signals move proceeds to cash. Buy signals use available cash only; if cash is depleted, the buy waits.

            **pro_rata**: sell signals create cash. When a buy signal changes the active set, current holdings are proportionally resized to fund the active target mix without borrowing.

            **replacement**: sell signals create cash. When a buy signal needs funding, the smallest current active holding is reduced first, then available cash funds the new active mix.

            **Band behavior**: a breach is checked only on the configured schedule and rebalances to target; it does not force an immediate daily sale.
            """
        )

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
    metric_columns[4].metric("Testfol CAGR (history)", f"{float(stats.get('cagr', 0.0)):.2f}%")
    metric_columns[5].metric("Testfol Max DD", f"{float(stats.get('max_drawdown', 0.0)):.2f}%")
    metric_columns[6].metric("SPY Test CAGR", f"{float(spy_test['cagr']):.2%}")
    metric_columns[7].metric(
        "Local Excess vs SPY",
        f"{local['walk_forward']['test']['cagr'] - float(spy_test['cagr']):+.2%}",
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
