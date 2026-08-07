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
    pick_unique_fillers,
    quarterly_company_selection,
    select_companies_up_to_threshold,
)
from simulation_pricing import (
    collapse_to_pricing_weights,
    ensure_price_tickers,
    price_ticker_map,
)

# Configuration (Loaded from config.py)

# Iterative Capping
def apply_caps(w_series, cap, total_target=1.0):
    w = w_series.copy()
    if w.sum() == 0: return w
    
    # Normalize to total_target first
    w = (w / w.sum()) * total_target
    
    for _ in range(config.MAX_CAP_ITERATIONS):
        excess = w[w > cap]
        if excess.empty: break
        
        surplus = (excess - cap).sum()
        w[w > cap] = cap
        
        others = w[w < cap]
        if others.empty: break
        
        # Redistribute surplus
        w[w < cap] = others + (surplus * others / others.sum())
    
    # Validation Check
    if (w > cap + 0.001).any():
        print(f"Warning: Capping failed to converge for some stocks > {cap}")
    if abs(w.sum() - total_target) > 0.001:
         print(f"Warning: Weights sum to {w.sum():.4f}, expected {total_target}")
         
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
    
    # Add Tickers from Changes File
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
    mega_values = pd.Series(index=data.index, dtype=float)
    sim_start = data.index[data.index.searchsorted(dates[0])]  # First trading day on or after first weights date
    mega_values.loc[sim_start] = 100.0  # Index base 100
    current_value = 100.0
    
    # Store constituents count
    constituents_history = []
    current_companies = []  # Standard selection at the company level
    
    print("Simulating NDX Mega 2.0 strategy...")

    prev_final_weights = None  # Track last good portfolio for carry-forward
    prev_price_tickers = None
    prev_top5 = None  # Track last known good top-5 for composition quality gate

    for i in range(len(dates) - 1):
        start_dt = dates[i]
        end_dt = dates[i+1]

        # 1. Selection Phase (At rebalance date start_dt)
        q_weights = weights_df[weights_df['Date'] == start_dt].copy()

        # Sort by Weight Descending
        q_weights = q_weights.sort_values(by='Weight', ascending=False)
        q_weights['CumWeight'] = q_weights['Weight'].cumsum()

        is_annual_recon = (start_dt.month == 12)

        # Data quality gate: flag quarters where top-5 changed too drastically
        # compared to last known good quarter (catches stale/distorted filing data).
        current_top5 = set(q_weights.head(5)['Ticker'].tolist())

        if prev_top5 is not None:
            overlap = len(current_top5 & prev_top5)
            if overlap < 2:
                if prev_final_weights is not None:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5 with prev) — carrying forward")
                    # Carry forward previous portfolio
                    final_weights = prev_final_weights.copy()
                    # Still need to simulate performance for this quarter
                    pricing_w = collapse_to_pricing_weights(
                        final_weights,
                        prev_price_tickers,
                        data.columns,
                    )
                    constituents_history.append({
                        "Date": start_dt, "Count": len(final_weights),
                        "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
                        "BufferRule": False, "Type": "CarryFwd",
                        "Tickers": "|".join(final_weights.index),
                        "Weights": "|".join([f"{w:.6f}" for w in final_weights]),
                        "PriceTickers": "|".join(pricing_w.index),
                    })
                    # Simulate performance with carried-forward weights
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
                            mega_values.loc[daily_vals.index] = daily_vals
                            current_value = daily_vals.iloc[-1]
                    except Exception as e:
                        print(f"  Error in carry-forward period: {e}")
                    continue
                else:
                    print(f"  Q {start_dt.date()}: DISTORTED weights (overlap={overlap}/5) — no previous portfolio, skipping")
                    continue
        
        # Standard Selection (Top 47%) - Using MEGA2 constants
        standard_tickers = []
        standard_companies = []
        official_tickers = get_official_constituents("NDXMEGA2", start_dt)
        _, company_weights, company_cum_weights, company_tickers = build_company_views(q_weights)
        
        if official_tickers:
            official_set = set(official_tickers)
            ranked = q_weights[q_weights["Ticker"].isin(official_set)].sort_values("Weight", ascending=False)
            standard_tickers = ranked["Ticker"].tolist()
            standard_companies = list(dict.fromkeys(canonical_company(t) for t in standard_tickers))
        elif is_annual_recon or not current_companies:
            # Reconstitution or First Run: 47% Selection
            standard_companies = select_companies_up_to_threshold(
                company_weights,
                config.MEGA2_TARGET_THRESHOLD,
            )
            standard_tickers = expand_companies_to_tickers(standard_companies, company_tickers)
            if not standard_tickers and not company_weights.empty:
                first_company = company_weights.index[0]
                standard_companies = [first_company]
                standard_tickers = expand_companies_to_tickers(standard_companies, company_tickers)
        else:
            # Quarterly Rebalance: Swap/Replacement Rules (Mega 2.0 Methodology)
            standard_companies = quarterly_company_selection(
                company_weights,
                company_cum_weights,
                current_companies,
                config.MEGA2_BUFFER_THRESHOLD,
            )
            standard_tickers = expand_companies_to_tickers(standard_companies, company_tickers)

        # Update current standard constituents
        current_companies = standard_companies.copy()
        
        # Minimum Security Rule: Fill to 9 if needed
        valid_mapped_all = q_weights[q_weights['IsMapped'] == True]
        
        selected_tickers = standard_tickers.copy()
        is_min_security_triggered = False
        
        if not official_tickers and len(selected_tickers) < config.MEGA2_MIN_CONSTITUENTS:
            is_min_security_triggered = True
            needed = config.MEGA2_MIN_CONSTITUENTS - len(selected_tickers)
            fillers = pick_unique_fillers(
                valid_mapped_all,
                selected_tickers,
                standard_companies,
                needed,
            )
            if fillers:
                selected_tickers.extend(fillers)

        if not selected_tickers:
             print(f"Warning: No selection for {start_dt}")
             continue

        # Filter for valid tickers in our price data
        event_price_tickers = price_ticker_map(q_weights)
        valid_tickers = [
            ticker
            for ticker in selected_tickers
            if event_price_tickers.get(ticker, ticker) in data.columns
        ]
        if is_min_security_triggered and len(valid_tickers) < config.MEGA2_MIN_CONSTITUENTS:
            needed = config.MEGA2_MIN_CONSTITUENTS - len(valid_tickers)
            live_universe = valid_mapped_all[
                valid_mapped_all["PriceTicker"].isin(data.columns)
            ]
            live_fillers = pick_unique_fillers(
                live_universe,
                valid_tickers,
                standard_companies,
                needed,
            )
            if live_fillers:
                valid_tickers.extend(live_fillers)
        mega_subset = q_weights[q_weights['Ticker'].isin(valid_tickers)].copy()
        
        if mega_subset.empty:
            continue
            
        # Stats


        if not is_min_security_triggered:
            # Normal: Re-weight by NDX company weight and 30% company cap.
            final_weights = apply_company_cap(
                mega_subset,
                config.MEGA2_SINGLE_STOCK_CAP,
                total_target=1.0,
            )
        else:
            # Minimum Security Rule Active (per methodology page 3):
            # 1. Standards get 99% of total weight (with 30% cap)
            # 2. Fillers collectively get remaining 1%, equally distributed
            # Fall back to cap-constrained split only when 99% is mathematically
            # impossible (e.g. 3 stocks × 30% cap = 90% max).
            standard_subset = mega_subset[mega_subset['Ticker'].isin(standard_tickers)]
            filler_subset = mega_subset[~mega_subset['Ticker'].isin(standard_tickers)]

            w_standard = pd.Series(dtype=float)
            w_filler = pd.Series(dtype=float)

            if not standard_subset.empty:
                standard_companies_live = standard_subset["Ticker"].map(
                    lambda t: getattr(config, "DUAL_CLASS_GROUPS", {}).get(t, t)
                ).nunique()
                max_possible = standard_companies_live * config.MEGA2_SINGLE_STOCK_CAP
                standard_target_cap = 0.99 if not filler_subset.empty else 1.0

                # Attempt 99/1 split per spec; fall back if caps prevent it
                if max_possible >= standard_target_cap:
                    standard_target = standard_target_cap
                else:
                    standard_target = max_possible

                w_standard = apply_company_cap(
                    standard_subset,
                    config.MEGA2_SINGLE_STOCK_CAP,
                    total_target=standard_target
                )

            # Fillers: Get whatever remains to reach 100%
            filler_budget = 1.0 - w_standard.sum()

            if not filler_subset.empty and filler_budget > 0:
                filler_count = len(filler_subset)
                w_filler = pd.Series(filler_budget / filler_count, index=filler_subset['Ticker'])

            # Combine
            final_weights = pd.concat([w_standard, w_filler])

        


        # Validation of Final Weights
        if abs(final_weights.sum() - 1.0) > 0.001:
             print(f"CRITICAL: Final weights for {start_dt} sum to {final_weights.sum():.4f}")
        # else:
        #      print(f"  Valid sum: {final_weights.sum():.4f}")

        # Save as previous good portfolio for carry-forward
        current_price_tickers = price_ticker_map(mega_subset)
        prev_final_weights = final_weights.copy()
        prev_price_tickers = current_price_tickers.copy()
        prev_top5 = current_top5

        # Stats Log
        constituents_history.append({
            "Date": start_dt,
            "Count": len(final_weights),
            "Top": final_weights.idxmax() if not final_weights.empty else "N/A",
            "BufferRule": is_min_security_triggered,
            "Type": "Recon" if is_annual_recon else "Rebal",
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

        # 3. Perf Simulation (Event-Driven)
        quarter_changes = changes_parser.get_changes_between(start_dt, end_dt)
        
        # Build timeline
        timeline = sorted(list(set([start_dt] + quarter_changes['Date'].tolist() + [end_dt])))
        timeline = [d for d in timeline if start_dt <= d <= end_dt]
        
        # Ensure we have at least start and end
        if len(timeline) < 2: timeline = [start_dt, end_dt]
        if timeline[0] != start_dt: timeline.insert(0, start_dt)
        if timeline[-1] != end_dt: timeline.append(end_dt)
        timeline = sorted(list(set(timeline)))

        curr_w = final_weights.copy()
        dropped_this_quarter = set()
        
        for k in range(len(timeline) - 1):
            sub_start = timeline[k]
            sub_end = timeline[k+1]
            
            if sub_start >= sub_end: continue
            
            try:
                pricing_w = collapse_to_pricing_weights(
                    curr_w,
                    current_price_tickers,
                    data.columns,
                )
                if pricing_w.empty:
                    continue
                # Slice logic [sub_start, sub_end]
                price_slice = data.loc[sub_start:sub_end, pricing_w.index]
                if price_slice.empty: continue
                
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
                
                if pricing_w.empty: continue
                
                shares = (pricing_w * current_value) / p_start
                daily_vals = price_slice.dot(shares)
                
                mega_values.loc[daily_vals.index] = daily_vals
                current_value = daily_vals.iloc[-1]
                
            except Exception as e:
                print(f"Error simulating sub-period {sub_start} to {sub_end}: {e}")

            # --- Handle Logic at Event Date (sub_end) ---
            if sub_end != end_dt:
                # Event Logic
                todays_changes = quarter_changes[quarter_changes['Date'] == sub_end]
                
                dropped_any = False
                for _, row in todays_changes.iterrows():
                     removed = row.get('Removed Ticker')
                     if pd.notna(removed) and removed in curr_w.index:
                         print(f"  [Event {sub_end.date()}] Dropping {removed}")
                         curr_w = curr_w.drop(removed)
                         dropped_any = True
                     
                     if pd.notna(removed):
                         dropped_this_quarter.add(removed)
                
                if dropped_any:
                    # Check < 9 Rule
                    current_tickers = curr_w.index.tolist()
                    needed = config.MEGA2_MIN_CONSTITUENTS - len(current_tickers)
                    
                    if needed > 0:
                        print(f"    Count fell to {len(current_tickers)}. Finding {needed} replacements...")
                        candidates_df = q_weights[
                            (~q_weights['Ticker'].isin(current_tickers)) &
                            (~q_weights['Ticker'].isin(dropped_this_quarter)) &
                            (q_weights['PriceTicker'].isin(data.columns))
                        ]
                        candidates_df = candidates_df[candidates_df['IsMapped'] == True]
                        current_standard_companies = [
                            getattr(config, "DUAL_CLASS_GROUPS", {}).get(t, t)
                            for t in standard_tickers
                        ]
                        replacements = pick_unique_fillers(
                            candidates_df,
                            current_tickers,
                            current_standard_companies,
                            needed,
                        )
                        for t in replacements:
                            print(f"      Selected replacement: {t}")
                            current_tickers.append(t)
                    
                    # Recalculate Weights (Mini-Rebalance)
                    # Classify Standards vs Fillers
                    # Update standard_tickers (remove drops)
                    standard_tickers = [t for t in standard_tickers if t in current_tickers]
                    
                    # Note: Any replacement we just added is effectively a "Filler"?
                    # PDF: "The Index Securities that were selected... are kept... The Minimum Security Rule is applied... The Minimum Security Weighting Process is applied."
                    # This implies replacements are Fillers (1% pool).
                    # 'standard_tickers' tracks the original elite set.
                    
                    # Prepare subset data
                    # We need 'Weight' data for the new set.
                    # Use q_weights for the relative weights.
                    mega_subset = q_weights[q_weights['Ticker'].isin(current_tickers)].copy()
                    current_price_tickers = price_ticker_map(mega_subset)
                    
                    # Re-Run Weighting Logic
                    is_min_security_triggered_now = (len(current_tickers) < config.MEGA2_MIN_CONSTITUENTS) # Should be false now unless we ran out of candidates
                    
                    # Logic block copied from main loop (simplified)
                    # Standards
                    standard_subset = mega_subset[mega_subset['Ticker'].isin(standard_tickers)]
                    # Fillers
                    filler_subset = mega_subset[~mega_subset['Ticker'].isin(standard_tickers)]
                    
                    w_standard = pd.Series(dtype=float)
                    w_filler = pd.Series(dtype=float)
                    
                    # Apply Logic (99/1 split per methodology)
                    if not standard_subset.empty:
                        standard_companies_live = standard_subset["Ticker"].map(
                            lambda t: getattr(config, "DUAL_CLASS_GROUPS", {}).get(t, t)
                        ).nunique()
                        max_possible = standard_companies_live * config.MEGA2_SINGLE_STOCK_CAP
                        standard_target_cap = 0.99 if not filler_subset.empty else 1.0
                        standard_target = standard_target_cap if max_possible >= standard_target_cap else max_possible

                        w_standard = apply_company_cap(
                            standard_subset,
                            config.MEGA2_SINGLE_STOCK_CAP, 
                            total_target=standard_target
                        )
                    
                    filler_budget = 1.0 - w_standard.sum()
                    if not filler_subset.empty and filler_budget > 0:
                        filler_count = len(filler_subset)
                        w_filler = pd.Series(filler_budget / filler_count, index=filler_subset['Ticker'])

                    curr_w = pd.concat([w_standard, w_filler])
                    
                    # Validation
                    if abs(curr_w.sum() - 1.0) > 0.001:
                         print(f"    Warning: Re-calc weights sum to {curr_w.sum():.4f}")

    mega_values = mega_values.dropna()
    
    plot_year_range = f"{mega_values.index.min().year}-{mega_values.index.max().year}"

    # Comparison
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
                
                # Apply Styling
                chart_style.apply_style()
                
                plt.figure(figsize=(14, 8))
                ax = plt.gca()
                
                plt.plot(mega_values.index, (mega_values / mega_values.iloc[0]) * 100, label='NDX Mega 2.0 (Simulated)', linewidth=2.5)
                plt.plot(benchmark_curve.index, benchmark_curve, label=f"{config.BENCHMARK_TICKER} (Total Return)", color='black', alpha=0.6, linestyle='--')
                
                plt.title(f'NDX Mega 2.0 Strategy vs Nasdaq-100 ({plot_year_range})')
                plt.yscale('log')
                plt.legend()
                
                chart_style.format_date_axis(ax)
                chart_style.format_y_axis(ax, log=True)
                chart_style.add_watermark(ax, "NDX Mega 2.0")
                
                out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega2_backtest.png")
                plt.savefig(out_img, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {out_img}")
                
                tot_ret_mega = (mega_values.iloc[-1] / mega_values.iloc[0]) - 1
                final_bench = benchmark_curve.iloc[-1]
                print(f"Total Return Mega 2.0: {tot_ret_mega:.2%}")
                print(f"Total Return {config.BENCHMARK_TICKER}: {final_bench - 100:.2f}%")
            else:
                 print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has 0 or NaN initial value.")
        else:
            print(f"Warning: Benchmark {config.BENCHMARK_TICKER} has NO valid data.")
    else:
        print(f"Warning: Benchmark ticker {config.BENCHMARK_TICKER} not found in price data.")
        
        # If no benchmark, still plot the mega_values
        chart_style.apply_style()
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        plt.plot(mega_values.index, (mega_values / mega_values.iloc[0]) * 100, label='NDX Mega 2.0 (Simulated)', linewidth=2.5)
        plt.title(f'NDX Mega 2.0 Strategy ({plot_year_range})')
        plt.yscale('log')
        plt.legend()
        
        chart_style.format_date_axis(ax)
        chart_style.format_y_axis(ax, log=True)
        chart_style.add_watermark(ax, "NDX Mega 2.0")
        
        out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega2_backtest.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {out_img}")
        
        tot_ret_mega = mega_values.iloc[-1] / mega_values.iloc[0] - 1
        tot_ret_ndx = ndx.iloc[-1] / ndx.iloc[0] - 1
        print(f"Total Return Mega 2.0: {tot_ret_mega:.2%}")
        print(f"Total Return NDX:      {tot_ret_ndx:.2%}")
        
    # Save Constituents History
    pd.DataFrame(constituents_history).to_csv(os.path.join(config.RESULTS_DIR, "ndx_mega2_constituents.csv"), index=False)
        
    # Save Daily Data for Testfol
    output_path = os.path.join(config.BASE_DIR, "..", "NDXMEGA2SIM.csv")
    mega_values.name = "Close"
    mega_values.to_csv(output_path, header=True, index_label="Date")
    print(f"Saved NDXMEGA2SIM data to {output_path}")

if __name__ == "__main__":
    backtest()
