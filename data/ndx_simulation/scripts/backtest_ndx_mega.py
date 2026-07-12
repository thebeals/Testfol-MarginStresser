import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import chart_style
import price_manager
import changes_parser
from official_index_data import get_official_constituents
from methodology_utils import (
    apply_company_cap,
    build_company_views,
    canonical_company,
    expand_companies_to_tickers,
    quarterly_company_selection,
    select_companies_up_to_threshold,
)

# Configuration (Loaded from config.py)

# Iterative Capping
def apply_caps(w_series, cap):
    w = w_series.copy()
    if w.sum() == 0: return w
    
    # Normalize first
    w = w / w.sum()
    
    for _ in range(config.MAX_CAP_ITERATIONS):
        excess = w[w > cap]
        if excess.empty: break
        
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        
        others = w[w < cap]
        if others.empty: break
        
        # Redistributions
        w[w < cap] = others + (surplus * others / others.sum())
    
    # Validation Check
    if (w > cap + 0.001).any():
        print(f"Warning: Capping failed to converge for some stocks > {cap}")
    if abs(w.sum() - 1.0) > 0.001:
         print(f"Warning: Weights sum to {w.sum():.4f}, expected 1.0")
         
    return w

def backtest(*, price_return=False):
    print("Loading data...")
    weights_df = pd.read_csv(config.WEIGHTS_FILE)
    weights_df['Date'] = pd.to_datetime(weights_df['Date'])
    
    # Get all tickers needed
    tickers = weights_df[weights_df['IsMapped'] == True]['Ticker'].unique().tolist()
    
    # Add Tickers from Changes File (to ensure we have data for mid-quarter adds)
    changes_df = changes_parser.load_changes()
    if not changes_df.empty:
        added_tickers = changes_df['Added Ticker'].dropna().unique().tolist()
        tickers.extend(added_tickers)
        
    tickers = list(set(tickers))
    
    if config.BENCHMARK_TICKER not in tickers:
        tickers.append(config.BENCHMARK_TICKER)
        
    print(f"Fetching prices for {len(tickers)} tickers...")
    start_date = weights_df['Date'].min().strftime('%Y-%m-%d')
    
    # price_manager reads NDX_DATA_SOURCE and POLYGON_API_KEY from environment
    data = price_manager.get_price_data(
        tickers,
        start_date,
        adjust_prices=not price_return,
    )
    
    if data is None or data.empty:
        print("Data fetch failed.")
        return
        
    data.index = pd.to_datetime(data.index)
    print(f"Data Shape: {data.shape}")

    # Simulation Variables
    dates = sorted(weights_df['Date'].unique())
    if dates and data.index.max() > dates[-1]:
        dates.append(data.index.max())
    mega_values = pd.Series(index=data.index, dtype=float)
    sim_start = data.index[data.index.searchsorted(dates[0])]  # First trading day on or after first weights date
    mega_values.loc[sim_start] = 100.0  # Index base 100
    current_value = 100.0
    
    # Store constituents count
    constituents_history = []
    current_companies = []
    prev_top5 = None  # Track last known good top-5 for composition quality gate
    prev_final_weights = None  # Track last good portfolio for carry-forward

    print("Simulating NDX Mega strategy...")
    
    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i+1]
        
        # 1. Selection Phase (At rebalance date start_dt)
        # Get full NDX composition (Mapped AND Unmapped)
        q_weights = weights_df[weights_df['Date'] == start_dt].copy()
        
        # Sort by Weight Descending (on FULL universe)
        q_weights = q_weights.sort_values(by='Weight', ascending=False)
        q_weights['CumWeight'] = q_weights['Weight'].cumsum()

        # Data quality gate: flag quarters where top-5 changed too drastically
        # compared to last known good quarter (catches stale/distorted filing data).
        current_top5 = set(q_weights.head(5)['Ticker'].tolist())

        if prev_top5 is not None:
            overlap = len(current_top5 & prev_top5)
            if overlap < 2:
                if prev_final_weights is not None:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5 with prev) — carrying forward")
                    curr_w = prev_final_weights.copy()
                    valid_tickers = [t for t in curr_w.index if t in data.columns]
                    curr_w = curr_w[valid_tickers]
                    if curr_w.sum() > 0:
                        curr_w = curr_w / curr_w.sum()
                    constituents_history.append({
                        "Date": start_dt, "Count": len(curr_w),
                        "Top": curr_w.idxmax() if not curr_w.empty else "N/A",
                        "Type": "CarryFwd",
                        "Tickers": "|".join(curr_w.index),
                        "Weights": "|".join([f"{w:.6f}" for w in curr_w])
                    })
                    try:
                        price_slice = data.loc[start_dt:end_dt, curr_w.index].ffill()
                        if not price_slice.empty:
                            p_start = price_slice.iloc[0]
                            valid_mask = (p_start > 0) & (p_start.notna())
                            valid_tkrs = valid_mask.index[valid_mask].tolist()
                            fw = curr_w[valid_tkrs]
                            if fw.sum() > 0:
                                fw = fw / fw.sum()
                            shares = (fw * current_value) / p_start[valid_tkrs]
                            daily_vals = price_slice[valid_tkrs].dot(shares)
                            mega_values.loc[daily_vals.index] = daily_vals
                            current_value = daily_vals.iloc[-1]
                    except Exception as e:
                        print(f"  Error in carry-forward period: {e}")
                    continue
                else:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5) — no previous portfolio, skipping")
                    continue

        # Identify Annual Reconstitution (December) vs Quarterly Rebalance
        is_annual_recon = (start_dt.month == 12)

        selected_companies = []
        selected_tickers = []
        official_tickers = get_official_constituents("NDXMEGA", start_dt)
        
        _, company_weights, company_cum_weights, company_tickers = build_company_views(q_weights)

        if official_tickers:
            official_set = set(official_tickers)
            ranked = q_weights[q_weights["Ticker"].isin(official_set)].sort_values("Weight", ascending=False)
            selected_tickers = ranked["Ticker"].tolist()
            selected_companies = list(dict.fromkeys(canonical_company(t) for t in selected_tickers))
        elif is_annual_recon or not current_companies:
            # Annual Reconstitution or First Run: 47% Selection
            selected_companies = select_companies_up_to_threshold(
                company_weights,
                config.MEGA1_TARGET_THRESHOLD,
            )
            selected_tickers = expand_companies_to_tickers(selected_companies, company_tickers)

            # Fallback
            if not selected_tickers and not company_weights.empty:
                first_company = company_weights.index[0]
                selected_companies = [first_company]
                selected_tickers = expand_companies_to_tickers(selected_companies, company_tickers)
                 
        else:
            # Quarterly Rebalance: Swap/Replacement Rules
            selected_companies = quarterly_company_selection(
                company_weights,
                company_cum_weights,
                current_companies,
                config.MEGA1_BUFFER_THRESHOLD,
            )
            selected_tickers = expand_companies_to_tickers(selected_companies, company_tickers)
            
        # Update current constituents for next loop
        current_companies = selected_companies.copy()
        
        if not selected_tickers:
             print(f"Warning: No selection for {start_dt}")
             continue

        # Filter for valid tickers in our price data
        valid_tickers = [t for t in selected_tickers if t in data.columns]
        
        # Prepare subset for weighting
        mega_subset = q_weights[q_weights['Ticker'].isin(valid_tickers)].copy()
        
        if mega_subset.empty:
            continue
            
        # Final weights use company-level capping, then split proportionally across share classes.
        final_weights = apply_company_cap(mega_subset, config.MEGA1_SINGLE_STOCK_CAP, total_target=1.0)

        # Save as previous good portfolio for carry-forward
        prev_final_weights = final_weights.copy()
        prev_top5 = current_top5

        # Stats
        constituents_history.append({
            "Date": start_dt,
            "Count": len(final_weights),
            "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
            "Type": "Recon" if is_annual_recon else "Rebal",
            "Tickers": "|".join(final_weights.index),
            "Weights": "|".join([f"{w:.6f}" for w in final_weights])
        })
        
        # 3. Perf Simulation (Event-Driven)
        quarter_changes = changes_parser.get_changes_between(start_dt, end_dt)
        
        # Build timeline
        timeline = sorted(list(set([start_dt] + quarter_changes['Date'].tolist() + [end_dt])))
        timeline = [d for d in timeline if start_dt <= d <= end_dt]
        
        # Ensure we have at least start and end
        if len(timeline) < 2: timeline = [start_dt, end_dt]
        if timeline[0] != start_dt: timeline.insert(0, start_dt)
        if timeline[-1] != end_dt: timeline.append(end_dt)
        timeline = sorted(list(set(timeline))) # Specific safety re-sort

        curr_w = final_weights.copy()
        
        for k in range(len(timeline) - 1):
            sub_start = timeline[k]
            # Simulation goes up to sub_end
            # If sub_end is a change date, the change happens AFTER market close (mostly).
            # So we simulate fully UP TO sub_end.
            # Then we adjust weights for the NEXT start.
            sub_end = timeline[k+1]
            
            if sub_start >= sub_end: continue
            
            try:
                # Slice logic: [sub_start, sub_end]
                # We include sub_end because the change happens effectively after that day's close.
                price_slice = data.loc[sub_start:sub_end, curr_w.index]
                
                # If sub_end == sub_start, slice might be 1 row
                if price_slice.empty: continue
                
                # We need to drop the first row IF it overlaps with previous? 
                # No, previous loop went to its sub_end.
                # If loop 1: D1 -> D2. We simulate D1..D2. Current Value is close of D2.
                # Next loop: D2 -> D3. Start at D2?
                # Double counting D2?
                # YES.
                # Correct: 
                # Loop 1 simulates return from D1 to D2. Value updates.
                # Loop 2 simulates D2 to D3.
                # The "return" of D2 calc: (Price_D2 / Price_D1) - 1.
                # If we include D2 in Loop 2, we calculate (Price_D3 / Price_D2).
                # This is correct chaining.
                # BUT we need start price.
                # p_start = price_slice.iloc[0] (which is D2 price).
                # shares = curr_w_value / Price_D2.
                # End Value = shares * Price_D3.
                # So yes, overlaps are handled by re-basing shares at start of each sub-period.
                
                price_slice = price_slice.ffill()
                
                p_start = price_slice.iloc[0]
                valid_mask = (p_start > 0) & (p_start.notna())
                
                if not valid_mask.all():
                    valid_tkrs = valid_mask.index[valid_mask].tolist()
                    curr_w = curr_w[valid_tkrs]
                    if curr_w.sum() > 0: curr_w = curr_w / curr_w.sum()
                    p_start = p_start[valid_tkrs]
                    price_slice = price_slice[valid_tkrs]
                
                if curr_w.empty: continue
                
                # Re-calculate shares based on Current Portfolio Value
                shares = (curr_w * current_value) / p_start
                
                daily_vals = price_slice.dot(shares)
                
                # Update series
                mega_values.loc[daily_vals.index] = daily_vals
                current_value = daily_vals.iloc[-1]
                
            except Exception as e:
                print(f"Error simulating sub-period {sub_start} to {sub_end}: {e}")
            
            # --- Handle Logic at Event Date (sub_end) ---
            if sub_end != end_dt:
                # This is a change date
                todays_changes = quarter_changes[quarter_changes['Date'] == sub_end]
                for _, row in todays_changes.iterrows():
                    removed = row.get('Removed Ticker')
                    # Added ticker logic omitted for Mega 1.0 (Drop Only)
                    
                    if pd.notna(removed) and removed in curr_w.index:
                        print(f"  [Event {sub_end.date()}] Dropping {removed} (Methodology: No Replace)")
                        curr_w = curr_w.drop(removed)
                
                # Re-normalize
                if curr_w.sum() > 0:
                    curr_w = curr_w / curr_w.sum()
                else:
                    print(f"  Warning: Portfolio empty after drops on {sub_end.date()}")

    mega_values = mega_values.dropna()
    
    plot_year_range = f"{mega_values.index.min().year}-{mega_values.index.max().year}"

    # Validation against NDX
    # Validation against NDX
    plt.figure(figsize=(14, 8))
    chart_style.apply_style()
    ax = plt.gca()
    
    # Plot Portfolio
    plt.plot(mega_values, label='NDX Mega (Simulated)', linewidth=2.5)

    if config.BENCHMARK_TICKER in data.columns:
        benchmark_data = data[config.BENCHMARK_TICKER].reindex(mega_values.index)
        
        # Determine first valid index for scaling
        first_valid_idx = benchmark_data.first_valid_index()
        if first_valid_idx:
            initial_val = benchmark_data.loc[first_valid_idx]
            if pd.notna(initial_val) and initial_val != 0:
                benchmark_curve = (benchmark_data / initial_val) * 100
                
                # Fill NaNs before the first valid index with 100 (flat start)
                benchmark_curve = benchmark_curve.fillna(100)
                
                plt.plot(benchmark_curve.index, benchmark_curve, label=f"{config.BENCHMARK_TICKER} (Total Return)", color='black', alpha=0.6, linestyle='--')
                
                final_bench = benchmark_curve.iloc[-1]
                print(f"Total Return {config.BENCHMARK_TICKER}: {final_bench - 100:.2f}%")
            else:
                 print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has 0 or NaN initial value.")
        else:
            print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has NO valid data.")
    else:
        print(f"Warning: Benchmark ticker {config.BENCHMARK_TICKER} not found in price data.")
        
        plt.title(f'NDX Mega Strategy vs Nasdaq-100 ({plot_year_range})')
        plt.yscale('log')
        plt.legend()
        
        chart_style.format_date_axis(ax)
        chart_style.format_y_axis(ax, log=True)
        chart_style.add_watermark(ax, "NDX Mega 1.0")
        
        out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega_backtest.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {out_img}")
        
        # Stats
        tot_ret_mega = mega_values.iloc[-1] / mega_values.iloc[0] - 1
        tot_ret_ndx = ndx.iloc[-1] / ndx.iloc[0] - 1
        print(f"Total Return Mega: {tot_ret_mega:.2%}")
        print(f"Total Return NDX:  {tot_ret_ndx:.2%}")
        
    # Save Constituents History
    pd.DataFrame(constituents_history).to_csv(os.path.join(config.RESULTS_DIR, "ndx_mega_constituents.csv"), index=False)
    
    # Save Daily Data for Testfol
    # Target: ../NDXMEGASIM.csv (Parent directory of BASE_DIR)
    
    output_symbol = "NDXMEGAPRICESIM" if price_return else "NDXMEGASIM"
    output_path = os.path.join(config.BASE_DIR, "..", f"{output_symbol}.csv")
    mega_values.name = "Close"
    mega_values.to_csv(output_path, header=True)
    print(f"Saved {output_symbol} data to {output_path}")

if __name__ == "__main__":
    backtest(price_return=os.environ.get("NDX_PRICE_RETURN", "").lower() in ("1", "true", "yes"))
