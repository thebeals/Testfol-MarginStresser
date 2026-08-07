import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import chart_style
import price_manager
import changes_parser
from official_index_data import get_official_constituents
from methodology_utils import apply_ndx30_company_cap
from simulation_pricing import (
    collapse_to_pricing_weights,
    ensure_price_tickers,
    price_ticker_map,
)


def apply_ndx30_caps(w_series):
    """NDX30 two-step capping: 22.5% individual + 48% aggregate for >4.5%."""
    w = w_series.copy()
    if w.sum() == 0:
        return w
    w = w / w.sum()

    # Step 1: Hard individual cap at 22.5%
    for _ in range(config.MAX_CAP_ITERATIONS):
        over = w[w > config.NDX30_HARD_CAP]
        if over.empty:
            break
        surplus = (over - config.NDX30_HARD_CAP).sum()
        w[w > config.NDX30_HARD_CAP] = config.NDX30_HARD_CAP
        under = w[w < config.NDX30_HARD_CAP]
        if under.empty:
            break
        w[under.index] = under + surplus * under / under.sum()

    # Step 2: Aggregate constraint — sum(w > 4.5%) <= 48%
    SOFT = config.NDX30_SOFT_CAP
    AGG = config.NDX30_AGG_LIMIT
    for _ in range(50):
        above = w[w > SOFT]
        if above.empty or above.sum() <= AGG + 0.001:
            break
        # Reduce the smallest above-threshold to exactly threshold
        min_t = above.idxmin()
        excess = w[min_t] - SOFT
        w[min_t] = SOFT
        # Redistribute to below-threshold (capped at threshold)
        below = w[w < SOFT]
        if below.empty:
            break
        room = SOFT - below
        total_room = room.sum()
        if total_room <= excess:
            w[below.index] = SOFT
        else:
            share = excess * (room / total_room)
            w[below.index] = below + share

    # Validation
    if (w > config.NDX30_HARD_CAP + 0.001).any():
        print(f"Warning: NDX30 hard cap failed to converge")
    if abs(w.sum() - 1.0) > 0.001:
        print(f"Warning: NDX30 weights sum to {w.sum():.4f}, expected 1.0")

    return w


def backtest():
    print("Loading data...")
    if not os.path.exists(config.WEIGHTS_FILE):
        print(f"Error: {config.WEIGHTS_FILE} not found.")
        return

    weights_df = ensure_price_tickers(pd.read_csv(config.WEIGHTS_FILE))
    weights_df['Date'] = pd.to_datetime(weights_df['Date'])

    # Get all tickers needed
    tickers = weights_df[weights_df['IsMapped'] == True]['PriceTicker'].unique().tolist()

    # Add tickers from changes file (for mid-quarter drops)
    changes_df = changes_parser.load_changes()
    if not changes_df.empty:
        added_tickers = changes_df['Added Ticker'].dropna().unique().tolist()
        tickers.extend(added_tickers)

    tickers = list(set(tickers))

    if config.BENCHMARK_TICKER not in tickers:
        tickers.append(config.BENCHMARK_TICKER)

    print(f"Fetching prices for {len(tickers)} tickers...")
    start_date = weights_df['Date'].min().strftime('%Y-%m-%d')

    data = price_manager.get_price_data(tickers, start_date)

    if data is None or data.empty:
        print("Data fetch failed.")
        return

    data.index = pd.to_datetime(data.index)
    print(f"Data Shape: {data.shape}")

    # Simulation Variables
    dates = sorted(weights_df['Date'].unique())
    if dates and data.index.max() > dates[-1]:
        dates.append(data.index.max())
    ndx30_values = pd.Series(index=data.index, dtype=float)
    sim_start = data.index[data.index.searchsorted(dates[0])]  # First trading day on or after first weights date
    ndx30_values.loc[sim_start] = 100.0
    current_value = 100.0

    constituents_history = []
    prev_top5 = None
    prev_final_weights = None
    prev_price_tickers = None

    print("Simulating NDX30 (Nasdaq-100 Top 30) strategy...")

    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i + 1]

        # 1. Selection Phase — full reconstitution every quarter
        q_weights = weights_df[weights_df['Date'] == start_dt].copy()
        q_weights = q_weights.sort_values(by='Weight', ascending=False)

        # Data quality gate: top-5 overlap check
        current_top5 = set(q_weights.head(5)['Ticker'].tolist())

        if prev_top5 is not None:
            overlap = len(current_top5 & prev_top5)
            if overlap < 2:
                if prev_final_weights is not None:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5 with prev) — carrying forward")
                    curr_w = prev_final_weights.copy()
                    pricing_w = collapse_to_pricing_weights(
                        curr_w,
                        prev_price_tickers,
                        data.columns,
                    )
                    constituents_history.append({
                        "Date": start_dt, "Count": len(curr_w),
                        "Top": curr_w.idxmax() if not curr_w.empty else "N/A",
                        "Type": "CarryFwd",
                        "Tickers": "|".join(curr_w.index),
                        "Weights": "|".join([f"{w:.6f}" for w in curr_w]),
                        "PriceTickers": "|".join(pricing_w.index),
                    })
                    try:
                        price_slice = data.loc[start_dt:end_dt, pricing_w.index].ffill()
                        if not price_slice.empty:
                            p_start = price_slice.iloc[0]
                            valid_mask = (p_start > 0) & (p_start.notna())
                            valid_tkrs = valid_mask.index[valid_mask].tolist()
                            fw = pricing_w[valid_tkrs]
                            if fw.sum() > 0:
                                fw = fw / fw.sum()
                            shares = (fw * current_value) / p_start[valid_tkrs]
                            daily_vals = price_slice[valid_tkrs].dot(shares)
                            ndx30_values.loc[daily_vals.index] = daily_vals
                            current_value = daily_vals.iloc[-1]
                    except Exception as e:
                        print(f"  Error in carry-forward period: {e}")
                    continue
                else:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5) — no previous portfolio, skipping")
                    continue

        # NDX30: Take top 30 COMPANIES by NDX weight
        # Per methodology: dual-class shares count as one company.
        # Select 30 companies, include all their individual securities.
        official_tickers = get_official_constituents("NDX30", start_dt)
        if official_tickers:
            official_set = set(official_tickers)
            selected_tickers = (
                q_weights[q_weights["Ticker"].isin(official_set)]
                .sort_values("Weight", ascending=False)["Ticker"]
                .tolist()
            )
        else:
            dc = getattr(config, 'DUAL_CLASS_GROUPS', {})
            mapped = q_weights[q_weights['IsMapped'] == True].copy()
            mapped['Company'] = mapped['Ticker'].map(lambda t: dc.get(t, t))

            company_weights = mapped.groupby('Company')['Weight'].sum().sort_values(ascending=False)
            company_tickers = mapped.groupby('Company')['Ticker'].apply(list).to_dict()

            top_companies = company_weights.head(config.NDX30_NUM_CONSTITUENTS).index.tolist()
            selected_tickers = []
            for comp in top_companies:
                selected_tickers.extend(company_tickers.get(comp, []))

        if not selected_tickers:
            print(f"Warning: No selection for {start_dt}")
            continue

        # Filter for valid tickers in price data
        event_price_tickers = price_ticker_map(q_weights)
        valid_tickers = [
            ticker
            for ticker in selected_tickers
            if event_price_tickers.get(ticker, ticker) in data.columns
        ]
        ndx30_subset = q_weights[q_weights['Ticker'].isin(valid_tickers)].copy()

        if ndx30_subset.empty:
            continue

        # Apply methodology-aligned company-level capping, then project back to issues
        final_weights = apply_ndx30_company_cap(ndx30_subset)
        current_price_tickers = price_ticker_map(ndx30_subset)

        # Save as previous good portfolio for carry-forward
        prev_final_weights = final_weights.copy()
        prev_price_tickers = current_price_tickers.copy()
        prev_top5 = current_top5

        constituents_history.append({
            "Date": start_dt,
            "Count": len(final_weights),
            "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
            "Type": "Recon",
            "Tickers": "|".join(final_weights.index),
            "Weights": "|".join([f"{w:.6f}" for w in final_weights]),
            "PriceTickers": "|".join(
                collapse_to_pricing_weights(
                    final_weights,
                    current_price_tickers,
                    data.columns,
                ).index
            ),
        })

        # 2. Performance Simulation (Event-Driven)
        quarter_changes = changes_parser.get_changes_between(start_dt, end_dt)

        # Build timeline
        timeline = sorted(list(set([start_dt] + quarter_changes['Date'].tolist() + [end_dt])))
        timeline = [d for d in timeline if start_dt <= d <= end_dt]

        if len(timeline) < 2:
            timeline = [start_dt, end_dt]
        if timeline[0] != start_dt:
            timeline.insert(0, start_dt)
        if timeline[-1] != end_dt:
            timeline.append(end_dt)
        timeline = sorted(list(set(timeline)))

        curr_w = final_weights.copy()

        for k in range(len(timeline) - 1):
            sub_start = timeline[k]
            sub_end = timeline[k + 1]

            if sub_start >= sub_end:
                continue

            try:
                pricing_w = collapse_to_pricing_weights(
                    curr_w,
                    current_price_tickers,
                    data.columns,
                )
                if pricing_w.empty:
                    continue
                price_slice = data.loc[sub_start:sub_end, pricing_w.index]
                if price_slice.empty:
                    continue

                price_slice = price_slice.ffill()

                p_start = price_slice.iloc[0]
                valid_mask = (p_start > 0) & (p_start.notna())

                if not valid_mask.all():
                    valid_tkrs = valid_mask.index[valid_mask].tolist()
                    pricing_w = pricing_w[valid_tkrs]
                    if pricing_w.sum() > 0:
                        pricing_w = pricing_w / pricing_w.sum()
                    p_start = p_start[valid_tkrs]
                    price_slice = price_slice[valid_tkrs]

                if pricing_w.empty:
                    continue

                shares = (pricing_w * current_value) / p_start
                daily_vals = price_slice.dot(shares)

                ndx30_values.loc[daily_vals.index] = daily_vals
                current_value = daily_vals.iloc[-1]

            except Exception as e:
                print(f"Error simulating sub-period {sub_start} to {sub_end}: {e}")

            # Handle mid-quarter removals (no replacement per NDX30 methodology)
            if sub_end != end_dt:
                todays_changes = quarter_changes[quarter_changes['Date'] == sub_end]
                for _, row in todays_changes.iterrows():
                    removed = row.get('Removed Ticker')
                    if pd.notna(removed) and removed in curr_w.index:
                        print(f"  [Event {sub_end.date()}] Dropping {removed} (NDX30: No Replace)")
                        curr_w = curr_w.drop(removed)

                # Re-normalize
                if curr_w.sum() > 0:
                    curr_w = curr_w / curr_w.sum()
                else:
                    print(f"  Warning: Portfolio empty after drops on {sub_end.date()}")

    ndx30_values = ndx30_values.dropna()

    plot_year_range = f"{ndx30_values.index.min().year}-{ndx30_values.index.max().year}"

    # Chart
    if config.BENCHMARK_TICKER in data.columns:
        benchmark_data = data[config.BENCHMARK_TICKER].reindex(ndx30_values.index)

        first_valid_idx = benchmark_data.first_valid_index()
        if first_valid_idx:
            initial_val = benchmark_data.loc[first_valid_idx]
            if pd.notna(initial_val) and initial_val != 0:
                benchmark_curve = (benchmark_data / initial_val) * 100
                benchmark_curve = benchmark_curve.fillna(100)

                chart_style.apply_style()
                plt.figure(figsize=(14, 8))
                ax = plt.gca()

                plt.plot(ndx30_values.index, (ndx30_values / ndx30_values.iloc[0]) * 100,
                         label='NDX30 (Simulated)', linewidth=2.5)
                plt.plot(benchmark_curve.index, benchmark_curve,
                         label=f"{config.BENCHMARK_TICKER} (Total Return)",
                         color='black', alpha=0.6, linestyle='--')

                plt.title(f'NDX30 (Top 30) vs Nasdaq-100 ({plot_year_range})')
                plt.yscale('log')
                plt.legend()

                chart_style.format_date_axis(ax)
                chart_style.format_y_axis(ax, log=True)
                chart_style.add_watermark(ax, "NDX30")

                out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx30_backtest.png")
                plt.savefig(out_img, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {out_img}")

                tot_ret = (ndx30_values.iloc[-1] / ndx30_values.iloc[0]) - 1
                final_bench = benchmark_curve.iloc[-1]
                print(f"Total Return NDX30: {tot_ret:.2%}")
                print(f"Total Return {config.BENCHMARK_TICKER}: {final_bench - 100:.2f}%")
            else:
                print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has 0 or NaN initial value.")
        else:
            print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has NO valid data.")
    else:
        print(f"Warning: Benchmark ticker {config.BENCHMARK_TICKER} not found in price data.")

        chart_style.apply_style()
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        plt.plot(ndx30_values.index, (ndx30_values / ndx30_values.iloc[0]) * 100,
                 label='NDX30 (Simulated)', linewidth=2.5)
        plt.title(f'NDX30 (Top 30) Strategy ({plot_year_range})')
        plt.yscale('log')
        plt.legend()

        chart_style.format_date_axis(ax)
        chart_style.format_y_axis(ax, log=True)
        chart_style.add_watermark(ax, "NDX30")

        out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx30_backtest.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {out_img}")

    # Save Constituents History
    pd.DataFrame(constituents_history).to_csv(
        os.path.join(config.RESULTS_DIR, "ndx30_constituents.csv"), index=False
    )

    # Save Daily Data for Testfol
    output_path = os.path.join(config.BASE_DIR, "..", "NDX30SIM.csv")
    ndx30_values.name = "Close"
    ndx30_values.to_csv(output_path, header=True, index_label="Date")
    print(f"Saved NDX30SIM data to {output_path}")


if __name__ == "__main__":
    backtest()
