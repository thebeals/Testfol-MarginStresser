"""Streamlit view for the validated allocation screener output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener.results import load_screened_results


RESULTS_PATH = Path(__file__).parents[2] / "data" / "screener-results.json"
BENCHMARK_PATH = Path(__file__).parents[2] / "data" / "screener-benchmark.json"
REBALANCE_PATH = Path(__file__).parents[2] / "data" / "rebalance-ranking.json"
MATRIX_PATH = Path(__file__).parents[2] / "data" / "rebalance-matrix-results.jsonl"
RESEARCH_PATH = Path(__file__).parents[2] / "data" / "rebalance-research-report.json"
EXPERT_PATH = Path(__file__).parents[2] / "data" / "rebalance-expert-shortlist.json"
OVERLAY_PATH = Path(__file__).parents[2] / "data" / "expert-overlay-paths.json"


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


def _research_rows(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "Rank": row["rank"],
            "Allocation": row["allocation_label"],
            "Method": row["method"],
            "Test CAGR": f"{row['local_test']['cagr']:.2%}",
            "Test Max DD": f"{row['local_test']['max_drawdown']:.2%}",
            "Sharpe": f"{row['local_test']['sharpe']:.2f}",
            "Recovery Days": row["recovery"]["recovery_days"] or "Not recovered",
            "3Y vs SPY": f"{row['rolling_3y_vs_spy']['beat_rate']:.1%}",
            "3Y vs VT": f"{row['rolling_3y_vs_vt']['beat_rate']:.1%}",
            "DCA Worst vs Contributions": f"{row['dca']['worst_value_vs_contributions']:.1%}",
            "Exact Testfol Method": "Yes" if row["testfol_validated"] else "No",
        }
        for row in candidates
    ]


def _expert_rows(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "Rank": row["expert_rank"],
            "Allocation": row["allocation_label"],
            "Method": row["method"],
            "Score": f"{row['expert_score']:.3f}",
            "Test CAGR": f"{row['local_test']['cagr']:.2%}",
            "Max DD": f"{row['local_test']['max_drawdown']:.2%}",
            "Sharpe": f"{row['local_test']['sharpe']:.2f}",
            "3Y > SPY": f"{row['rolling_3y_vs_spy']['beat_rate']:.1%}",
            "3Y > VT": f"{row['rolling_3y_vs_vt']['beat_rate']:.1%}",
            "Testfol Exact": "Yes" if row["testfol_validated"] else "No",
        }
        for row in candidates
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
        research = json.loads(RESEARCH_PATH.read_text(encoding="utf-8"))
        expert = json.loads(EXPERT_PATH.read_text(encoding="utf-8"))
        overlay_paths = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
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

    st.subheader("Distinct Top-25 Research Set")
    st.caption(research["selection_rule"])
    summary = research["summary"]
    research_metrics = st.columns(5)
    research_metrics[0].metric("Candidates", summary["candidate_count"])
    research_metrics[1].metric("Median Test CAGR", f"{summary['median_test_cagr']:.1%}")
    research_metrics[2].metric("Median Max DD", f"{summary['median_test_drawdown']:.1%}")
    research_metrics[3].metric("3Y Windows > SPY", f"{summary['median_spy_rolling_3y_beat_rate']:.1%}")
    research_metrics[4].metric("3Y Windows > VT", f"{summary['median_vt_rolling_3y_beat_rate']:.1%}")
    st.dataframe(pd.DataFrame(_research_rows(research["candidates"])), use_container_width=True, hide_index=True)

    st.warning(
        "This is historical evidence, not a forecast. The selected set did not beat SPY in every rolling period, "
        "and no diversification rule prevents a prolonged or permanent loss."
    )
    research_labels = [f"#{row['rank']} | {row['method']} | {row['allocation_label']}" for row in research["candidates"]]
    research_choice = st.selectbox("Research candidate", research_labels, key="research_candidate")
    selected_research = research["candidates"][research_labels.index(research_choice)]
    rc = st.columns(5)
    rc[0].metric("Test CAGR", f"{selected_research['local_test']['cagr']:.1%}")
    rc[1].metric("Test Max DD", f"{selected_research['local_test']['max_drawdown']:.1%}")
    rc[2].metric("Worst 3M", f"{selected_research['recovery']['worst_rolling_3m']:.1%}")
    rc[3].metric("Recovery", f"{selected_research['recovery']['recovery_days']} days" if selected_research["recovery"]["recovery_days"] else "Not recovered")
    rc[4].metric("DCA Worst", f"{selected_research['dca']['worst_value_vs_contributions']:.1%}")
    with st.expander("Why it worked: asset contribution and market regimes"):
        contribution_rows = [
            {"Ticker": ticker, "Target Weight": values["weight"], "Asset CAGR": values["asset_cagr"], "Weighted CAGR Contribution": values["weighted_cagr"]}
            for ticker, values in selected_research["asset_contribution"].items()
        ]
        st.dataframe(pd.DataFrame(contribution_rows), use_container_width=True, hide_index=True)
        regime_rows = []
        for regime, values in selected_research["regimes"].items():
            regime_rows.append(
                {
                    "Regime": regime,
                    "Portfolio Return": values["portfolio"].get("total_return"),
                    "Portfolio Max DD": values["portfolio"].get("max_drawdown"),
                    "SPY Return": values["SPY"].get("total_return"),
                    "VT Return": values["VT"].get("total_return"),
                }
            )
        st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)
    with st.expander("Protection, failure modes, and confidence"):
        st.markdown(
            f"""
            **Protection:** {research['interpretation']['protection']}

            **Failure behavior:** The selected candidate's worst observed test drawdown was **{selected_research['local_test']['max_drawdown']:.1%}**, its worst rolling three-month result was **{selected_research['recovery']['worst_rolling_3m']:.1%}**, and its longest recovery status is **{selected_research['recovery']['recovery_days'] or 'not recovered in the sample'} days**. These are historical observations, not limits.

            **DCA risk:** {research['interpretation']['dca']} In this simulation, the worst value relative to cumulative contributions was **{selected_research['dca']['worst_value_vs_contributions']:.1%}**.

            **Confidence:** {research['interpretation']['confidence']} This candidate beat SPY in **{selected_research['rolling_3y_vs_spy']['beat_rate']:.1%}** of rolling three-year windows and VT in **{selected_research['rolling_3y_vs_vt']['beat_rate']:.1%}**.
            """
        )

    st.subheader("Expert Top-10 Shortlist")
    st.caption(expert["selection_rule"])
    st.dataframe(pd.DataFrame(_expert_rows(expert["shortlist"])), use_container_width=True, hide_index=True)
    st.info(
        "This shortlist is an investor-style judgment layer over the quantitative pool. "
        "It favors candidates that beat SPY while balancing drawdown, diversification, crisis behavior, "
        "leveraged exposure, rolling-market consistency, and Testfol validation."
    )
    expert_labels = [f"#{row['expert_rank']} | {row['method']} | {row['allocation_label']}" for row in expert["shortlist"]]
    expert_choice = st.selectbox("Expert candidate", expert_labels, key="expert_candidate")
    selected_expert = expert["shortlist"][expert_labels.index(expert_choice)]
    expert_columns = st.columns(6)
    expert_columns[0].metric("Expert Score", f"{selected_expert['expert_score']:.3f}")
    expert_columns[1].metric("Test CAGR", f"{selected_expert['local_test']['cagr']:.1%}")
    expert_columns[2].metric("Max DD", f"{selected_expert['local_test']['max_drawdown']:.1%}")
    expert_columns[3].metric("Sharpe", f"{selected_expert['local_test']['sharpe']:.2f}")
    expert_columns[4].metric("Leveraged Weight", f"{selected_expert['diversification']['leveraged_weight']:.1%}")
    expert_columns[5].metric("Testfol Exact", "Yes" if selected_expert["testfol_validated"] else "No")
    with st.expander("Why I would consider it"):
        st.markdown("**Strengths**")
        for item in selected_expert["rationale"]["why_consider"]:
            st.markdown(f"- {item}")
        st.markdown("**Weaknesses**")
        for item in selected_expert["rationale"]["what_can_go_wrong"]:
            st.markdown(f"- {item}")
        st.markdown(f"**Black-swan caveat:** {selected_expert['rationale']['black_swan_note']}")

    st.subheader("Original vs Protected Performance")
    st.caption(
        "Protection rule: sell fully to SHV after any allocated component falls 12.5% in a day; "
        "wait 20 trading sessions; re-enter at the next monthly check only when the portfolio is above EMA100."
    )
    st.caption("CAGR Change is a percentage-point change in CAGR. DD Improvement is a percentage-point reduction in drawdown; they are not the same metric.")
    overlay_options = {
        f"#{path['expert_rank']} | {path['method']} | {path['allocation_label']}": path["expert_rank"]
        for path in overlay_paths["paths"]
    }
    selected_overlay_labels = st.multiselect(
        "Portfolios visible in the chart",
        options=list(overlay_options),
        default=list(overlay_options),
        key="overlay_portfolios",
    )
    selected_overlay_ranks = {overlay_options[label] for label in selected_overlay_labels}
    comparison_rows = []
    for path in overlay_paths["paths"]:
        baseline = path["baseline_metrics"]
        protected = path["protected_metrics"]
        comparison_rows.append(
            {
                "Rank": path["expert_rank"],
                "Method": path["method"],
                "Original CAGR": f"{baseline['cagr']:.2%}",
                "Protected CAGR": f"{protected['cagr']:.2%}",
                "CAGR Change": f"{protected['cagr'] - baseline['cagr']:+.2%}",
                "Original DD": f"{baseline['max_drawdown']:.2%}",
                "Protected DD": f"{protected['max_drawdown']:.2%}",
                "DD Improvement": f"{protected['max_drawdown'] - baseline['max_drawdown']:+.2%}",
                "Protected Sharpe": f"{protected['sharpe']:.2f}",
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    st.caption(f"Preferred grid configuration: `{json.dumps(overlay_paths['preferred_config'], sort_keys=True)}`")
    chart_mode = st.radio("Chart series", ["Original", "Protected"], horizontal=True, key="expert_chart_mode")
    fig = go.Figure()
    series_key = chart_mode.lower()
    for path in overlay_paths["paths"]:
        if path["expert_rank"] not in selected_overlay_ranks:
            continue
        fig.add_trace(
            go.Scatter(
                x=path["dates"],
                y=path[series_key],
                mode="lines",
                name=f"#{path['expert_rank']} {path['method']}",
                line={"width": 1},
                opacity=0.65,
            )
        )
    first = overlay_paths["paths"][0]
    fig.add_trace(go.Scatter(x=first["dates"], y=first["SPY"], mode="lines", name="SPY", line={"width": 3, "color": "black"}))
    fig.update_layout(title=f"Expert Top 10: {chart_mode} vs SPY", yaxis={"type": "log", "title": "Growth of $100 (log scale)"}, xaxis_title="Date", hovermode="x unified", height=650)
    st.plotly_chart(fig, use_container_width=True)
    selected_overlay_rank = st.selectbox("Detailed overlay comparison", [path["expert_rank"] for path in overlay_paths["paths"]], key="overlay_detail_rank")
    detail = next(path for path in overlay_paths["paths"] if path["expert_rank"] == selected_overlay_rank)
    detail_fig = go.Figure()
    for key, label, color in (("original", "Original", "#888888"), ("protected", "Protected", "#1f77b4"), ("SPY", "SPY", "#111111")):
        detail_fig.add_trace(go.Scatter(x=detail["dates"], y=detail[key], mode="lines", name=label, line={"color": color, "width": 3 if key != "original" else 2}))
    detail_fig.update_layout(title=f"#{detail['expert_rank']} {detail['method']}: Original vs Protected vs SPY", yaxis={"type": "log", "title": "Growth of $100 (log scale)"}, xaxis_title="Date", hovermode="x unified", height=500)
    st.plotly_chart(detail_fig, use_container_width=True)

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
