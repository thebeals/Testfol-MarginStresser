import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import config
import chart_style
import price_manager
from ndx_methodology import build_validation_periods

# Configuration
INDEX_EVENT_WEIGHTS_FILE = os.path.join(
    config.RESULTS_DIR,
    "nasdaq_index_event_weights.csv",
)
WEIGHTS_FILE = (
    INDEX_EVENT_WEIGHTS_FILE
    if os.path.exists(INDEX_EVENT_WEIGHTS_FILE)
    else config.WEIGHTS_FILE
)
PROXY_TICKER = "QQQ"
OFFICIAL_NDX_FRED_SERIES = "NASDAQXNDX"


def get_fred_series(series_id):
    """Fetch an official daily index series from FRED."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        frame = pd.read_csv(
            io.StringIO(response.text),
            parse_dates=["observation_date"],
            index_col="observation_date",
        )
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"  Failed to fetch {series_id} from FRED: {exc}")
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[series_id], errors="coerce").dropna()


def validate():
    print("Loading weights...")
    weights = pd.read_csv(WEIGHTS_FILE)
    weights['Date'] = pd.to_datetime(weights['Date'])
    
    # Get unique tickers
    tickers = weights[weights['IsMapped'].astype(bool)]['Ticker'].unique().tolist()
    print(f"Fetching prices for {len(tickers)} mapped tickers + {PROXY_TICKER}...")
    
    # Fetch all prices including benchmark
    tickers_to_fetch = tickers + [PROXY_TICKER]
    
    # Optimize: Chunking not needed for 200 items usually, but good practice
    start_date = weights['Date'].min().strftime('%Y-%m-%d')
    
    try:
        data = price_manager.get_price_data(tickers_to_fetch, start_date)
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Calculate Synthetic Index
    # We need to simulate the portfolio value over time.
    # Start Value = 100 on first date.
    
    # Align dates
    dates = sorted(weights['Date'].unique())
    
    current_value = 100.0
    
    # Ensure data index is datetime
    data.index = pd.to_datetime(data.index)
    
    print("Simulating portfolio performance...")
    print(f"Data Columns: {data.columns[:5]}")
    print(f"Weights Sample: {weights.head()}")
    
    # Create a full daily date range for the simulation
    full_dates = data.index
    sim_values = pd.Series(index=full_dates, dtype=float)
    quarter_dates = sorted(weights['Date'].unique())
    sim_start = full_dates[full_dates.searchsorted(quarter_dates[0])]  # First trading day on or after first weights date
    sim_values.loc[sim_start] = 100.0
    
    # Actually, simpler approach:
    # Iterate through quarters.
    # For each quarter, we have a portfolio of tickers and weights.
    # Hold that portfolio until next quarter.
    # Calculate return of that portfolio.
    
    prev_top5 = None  # Track last known good top-5 for composition quality gate

    periods = build_validation_periods(dates, full_dates[-1])
    for i, (start_dt, end_dt) in enumerate(periods):

        # Get Weights at Start Date
        w_df = weights[weights['Date'] == start_dt]

        # Data quality gate: skip quarters with too few positions (catches garbage data)
        if len(w_df) < 15:
            print(f"  Q {start_dt.date()}: SKIPPED — only {len(w_df)} positions (< 15)")
            continue

        # Data quality gate: flag quarters where top-5 changed too drastically
        # compared to last known good quarter (catches stale/distorted filing data).
        current_top5 = set(w_df.sort_values('Weight', ascending=False).head(5)['Ticker'].tolist())
        if prev_top5 is not None:
            overlap = len(current_top5 & prev_top5)
            if overlap < 2:
                print(f"  Q {start_dt.date()}: WARNING — top-5 overlap={overlap}/5 with prev event")
        prev_top5 = current_top5

        # PriceTicker keeps the constituent identity intact while explicitly
        # assigning QQQ only to holdings whose own historical series is absent.
        # This prevents survivor renormalization from silently deleting losers.
        pricing_w = w_df.copy()
        pricing_w["PricingTicker"] = pricing_w.get(
            "PriceTicker", pricing_w["Ticker"]
        ).fillna(pricing_w["Ticker"])
        def has_period_prices(ticker):
            if ticker not in data.columns:
                return False
            series = data.loc[start_dt:end_dt, ticker].ffill()
            return bool(
                not series.empty
                and pd.notna(series.iloc[0])
                and series.iloc[0] > 0
                and pd.notna(series.iloc[-1])
            )

        has_prices = pricing_w["PricingTicker"].map(has_period_prices)
        proxy_weight = float(pricing_w.loc[~has_prices, "Weight"].sum())
        pricing_w.loc[~has_prices, "PricingTicker"] = PROXY_TICKER
        proxy_weight = float(
            pricing_w.loc[pricing_w["PricingTicker"] == PROXY_TICKER, "Weight"].sum()
        )
        pricing_w = pricing_w.groupby("PricingTicker", as_index=False)["Weight"].sum()
        valid_w = pricing_w[pricing_w['PricingTicker'].isin(data.columns)]

        if i == 0:
            print(f"Date: {start_dt}")
            print(f"Total Weights in CSV: {len(w_df)}")
            print(f"Valid Tickers in Prices: {len(valid_w)}")
            if len(valid_w) == 0:
                print(f"Sample Tickers in CSV: {w_df['Ticker'].iloc[:5].tolist()}")
                print(f"Sample Columns in Data: {data.columns[:5].tolist()}")

        if len(valid_w) == 0:
            continue
            
        raw_sum = valid_w['Weight'].sum()
        if raw_sum <= 0:
            continue

        port_tickers = valid_w['PricingTicker'].values
        abs_weights = valid_w['Weight'].values / raw_sum  # Normalized to 1.0

        # Get prices and calculate return
        try:
            price_slice = data.loc[start_dt:end_dt, port_tickers].ffill()
            p_start = price_slice.iloc[0]
            p_end = price_slice.iloc[-1]

            valid_mask = (p_start > 0) & (p_start.notna()) & (p_end.notna())
            if not valid_mask.any():
                print(f"Period {start_dt}: No valid pricing.")
                continue

            valid_tickers = valid_mask.index[valid_mask].tolist()
            p_s = p_start[valid_tickers]
            p_e = p_end[valid_tickers]
            w_s = pd.Series(abs_weights, index=port_tickers)[valid_tickers]

            # Re-normalize for tickers with valid prices
            used_weight = w_s.sum()
            if used_weight > 0:
                w_s = w_s / used_weight

            effective_coverage = raw_sum * used_weight
            survivor_ret = (w_s * (p_e / p_s)).sum()

            # Benchmark return for the period
            bm_ret = 1.0
            if PROXY_TICKER in data.columns:
                bm_slice = data.loc[start_dt:end_dt, PROXY_TICKER].ffill()
                if not bm_slice.empty and pd.notna(bm_slice.iloc[0]) and bm_slice.iloc[0] > 0:
                    bm_ret = bm_slice.iloc[-1] / bm_slice.iloc[0]

            # Daily series using survivor prices (gives good daily shape)
            price_valid = price_slice[valid_tickers]
            shares = w_s / p_s
            daily_survivor = price_valid.dot(shares)  # Relative to 1.0

            minimum_coverage = float(
                os.environ.get("NDX_MIN_VALIDATION_COVERAGE", "0.99")
            )
            if effective_coverage < minimum_coverage:
                raise RuntimeError(
                    f"{start_dt.date()} priced weight coverage "
                    f"{effective_coverage:.2%} is below {minimum_coverage:.2%}"
                )

            period_factor = daily_survivor.iloc[-1]

            # Coverage report
            print(f"  Q {start_dt.date()}: coverage={effective_coverage:.1%}, "
                  f"proxy-weight={proxy_weight:.1%}, portfolio={period_factor:.4f}, "
                  f"QQQ={bm_ret:.4f}")

            daily_vals_scaled = daily_survivor * current_value
            sim_values.loc[daily_vals_scaled.index] = daily_vals_scaled
            current_value = daily_vals_scaled.iloc[-1]

        except RuntimeError:
            raise
        except Exception as e:
            print(f"Error in period {start_dt}: {e}")
            pass

    sim_values = sim_values.dropna()
    
    if sim_values.empty:
        print("Simulation yielded no values.")
        return

    # QQQ remains only a survivorship-bias proxy above. Validate the finished
    # reconstruction against Nasdaq's independent total-return index instead.
    ndx = get_fred_series(OFFICIAL_NDX_FRED_SERIES)
    if ndx.empty:
        print(f"Official benchmark {OFFICIAL_NDX_FRED_SERIES} is unavailable.")
        return

    ndx = ndx.reindex(sim_values.index).dropna()
    common_idx = sim_values.index.intersection(ndx.index)
    
    if len(common_idx) < 10:
        print("Not enough overlapping data between Simulation and Benchmark.")
        return
        
    sim_values = sim_values.loc[common_idx]
    ndx = ndx.loc[common_idx]

    ndx = ndx / ndx.iloc[0] * sim_values.iloc[0]
    
    # Calculate Stats
    correlation = sim_values.corr(ndx)
    
    r_sim = sim_values.pct_change().dropna()
    r_bm = ndx.pct_change().dropna()
    daily_correlation = r_sim.corr(r_bm)
    te = (r_sim - r_bm).std() * np.sqrt(252)
    sim_total_return = sim_values.iloc[-1] / sim_values.iloc[0] - 1
    official_total_return = ndx.iloc[-1] / ndx.iloc[0] - 1
    
    print("\n--- Validation Results ---")
    print(f"Benchmark: {OFFICIAL_NDX_FRED_SERIES} (Nasdaq-100 Total Return)")
    print(f"Level Correlation: {correlation:.4f}")
    print(f"Daily Return Correlation: {daily_correlation:.4f}")
    print(f"Tracking Error (Annualized): {te:.2%}")
    print(f"Total Return Sim: {sim_total_return:.2%}")
    print(f"Total Return NDX: {official_total_return:.2%}")
    
    # Plot
    chart_style.apply_style()
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    plt.plot(sim_values, label='Reconstructed (From Filings)', linewidth=2.0)
    plt.plot(
        ndx,
        label=f'Nasdaq-100 Total Return ({OFFICIAL_NDX_FRED_SERIES})',
        linestyle='--',
        alpha=0.8,
        color='#555555',
    )
    
    plt.yscale('log')
    chart_style.format_date_axis(ax)
    chart_style.format_y_axis(ax, log=True)
    plt.title(
        "Reconstructed Nasdaq-100 vs Official Total Return Index\n"
        f"Daily Corr: {daily_correlation:.4f}, TE: {te:.2%}"
    )
    plt.legend()
    
    out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_validation.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to {out_img}")

    return {
        "benchmark": OFFICIAL_NDX_FRED_SERIES,
        "start": common_idx[0],
        "end": common_idx[-1],
        "observations": len(common_idx),
        "level_correlation": float(correlation),
        "daily_return_correlation": float(daily_correlation),
        "tracking_error": float(te),
        "sim_total_return": float(sim_total_return),
        "official_total_return": float(official_total_return),
    }

def validate_qbig():
    print("\n\n=== Validating NDXMEGA2SIM vs QBIG ===")
    
    # 1. Load Simulation
    sim_path = os.path.join(config.BASE_DIR, "..", "NDXMEGA2SIM.csv")
    
    if not os.path.exists(sim_path):
        print(f"Skipping QBIG check: {sim_path} not found.")
        return

    print(f"Loading simulation from {sim_path}...")
    sim_df = pd.read_csv(sim_path, parse_dates=[0], index_col=0).sort_index()
    sim_df.index.name = "Date"
    
    # Rename for clarity
    if 'Close' in sim_df.columns:
        sim_series = sim_df['Close']
    else:
        sim_series = sim_df.iloc[:, 0]
    
    sim_series.name = "NDXMEGA2SIM"

    # 2. Fetch Real (QBIG)
    ticker = "QBIG"
    print(f"Fetching real data for {ticker}...")
    try:
        real_df = price_manager.get_price_data([ticker], "2000-01-01")
        
        if isinstance(real_df, pd.Series):
             real_series = real_df
        else:
             if ticker in real_df.columns:
                 real_series = real_df[ticker]
             else:
                 available = ', '.join(map(str, real_df.columns[:10]))
                 raise ValueError(f"{ticker} was not returned by the price provider. Available columns: {available}")
                 
        real_series.name = "Real (QBIG)"
        real_series.index = pd.to_datetime(real_series.index)
        real_series = real_series.dropna() 
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return

    # 3. Align and Compare
    common_idx = sim_series.index.intersection(real_series.index)
    
    if len(common_idx) < 10:
        print("Insufficient overlap between Simulation and Real QBIG.")
        return

    print(f"Comparing overlap: {common_idx.min().date()} to {common_idx.max().date()} ({len(common_idx)} days)")
    
    sim_slice = sim_series.loc[common_idx]
    real_slice = real_series.loc[common_idx]

    # Normalize to 100
    sim_norm = sim_slice / sim_slice.iloc[0] * 100
    real_norm = real_slice / real_slice.iloc[0] * 100

    # Calculate Metrics
    tr_sim = (sim_slice.iloc[-1] / sim_slice.iloc[0]) - 1
    tr_real = (real_slice.iloc[-1] / real_slice.iloc[0]) - 1
    
    corr = sim_slice.corr(real_slice)
    
    ret_sim = sim_slice.pct_change().dropna()
    ret_real = real_slice.pct_change().dropna()
    diff = ret_sim - ret_real
    te = diff.std() * (252 ** 0.5)

    print(f"--> Correlation:       {corr:.4f}")
    print(f"--> Tracking Error:    {te:.2%}")
    print(f"--> Sim Period Return: {tr_sim:.2%}")
    print(f"--> QBIG Period Return:{tr_real:.2%}")
    print(f"--> Difference:        {tr_sim - tr_real:.2%}")

    # Plot
    chart_style.apply_style()
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    plt.plot(sim_norm, label=f'NDX Mega 2.0 Index (Simulated Underlying) ({tr_sim:.1%})', linewidth=2.5)
    plt.plot(real_norm, label=f'QBIG ETF (Real) ({tr_real:.1%})', linestyle='--', linewidth=2.0, color='#C44E52')
    
    chart_style.format_date_axis(ax)
    chart_style.format_y_axis(ax, log=False)
    chart_style.add_watermark(ax, "QBIG Comparison")
    
    plt.title(f"Validation: NDX Mega 2.0 Index vs QBIG ETF\nCorr: {corr:.4f}, TE: {te:.2%}")
    plt.legend()
    
    out_img = os.path.join(config.RESULTS_DIR, "charts", "validation_qbig.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_img}")

def compare_strategies():
    print("\n\n=== Comparing NDX Mega 1.0 vs 2.0 vs NDX30 ===")

    # Paths
    path1 = os.path.join(config.BASE_DIR, "..", "NDXMEGASIM.csv")
    path2 = os.path.join(config.BASE_DIR, "..", "NDXMEGA2SIM.csv")
    path3 = os.path.join(config.BASE_DIR, "..", "NDX30SIM.csv")

    if not (os.path.exists(path1) and os.path.exists(path2)):
        print("Missing one or both Mega simulation files.")
        return

    # Load
    s1 = pd.read_csv(path1, parse_dates=[0], index_col=0).sort_index().iloc[:, 0]
    s1.name = "Mega 1.0"

    s2 = pd.read_csv(path2, parse_dates=[0], index_col=0).sort_index().iloc[:, 0]
    s2.name = "Mega 2.0"

    s3 = None
    if os.path.exists(path3):
        s3 = pd.read_csv(path3, parse_dates=[0], index_col=0).sort_index().iloc[:, 0]
        s3.name = "NDX30"

    # Align
    idx = s1.index.intersection(s2.index)
    if s3 is not None:
        idx = idx.intersection(s3.index)
    if idx.empty:
        print("No overlap.")
        return

    slice1 = s1.loc[idx]
    slice2 = s2.loc[idx]

    # Stats
    ret1 = slice1.iloc[-1] / slice1.iloc[0] - 1
    ret2 = slice2.iloc[-1] / slice2.iloc[0] - 1

    # Plot
    norm1 = slice1 / slice1.iloc[0] * 100
    norm2 = slice2 / slice2.iloc[0] * 100

    chart_style.apply_style()
    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    plt.plot(norm1, label=f'NDX Mega 1.0 ({ret1:.0%})', linewidth=2.0)
    plt.plot(norm2, label=f'NDX Mega 2.0 ({ret2:.0%})', linewidth=2.0)

    if s3 is not None:
        slice3 = s3.loc[idx]
        ret3 = slice3.iloc[-1] / slice3.iloc[0] - 1
        norm3 = slice3 / slice3.iloc[0] * 100
        plt.plot(norm3, label=f'NDX30 ({ret3:.0%})', linewidth=2.0)

    chart_style.format_date_axis(ax)
    chart_style.format_y_axis(ax, log=True)
    chart_style.add_watermark(ax, "Strategy Comparison")

    plt.title("Strategy Comparison: NDX Mega 1.0 vs 2.0 vs NDX30")
    plt.legend()

    out_img = os.path.join(config.RESULTS_DIR, "charts", "ndx_mega_comparison.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_img}")

def validate_against_real_indexes():
    """Compare simulations against real Nasdaq index data from FRED (price + total return)."""
    print("\n\n=== Validating Simulations vs Real Nasdaq Indexes (FRED) ===")

    def compare_pair(label, sim_series, fred_id, tag=""):
        """Compare sim vs a single FRED series. Returns (corr, te, gap) or None."""
        real = get_fred_series(fred_id)
        if real.empty:
            return None

        common = sim_series.index.intersection(real.index)
        if len(common) < 10:
            print(f"  {label} vs {fred_id}: only {len(common)} overlapping days — skipping")
            return None

        s = sim_series.loc[common]
        r = real.loc[common]

        s_norm = s / s.iloc[0] * 100
        r_norm = r / r.iloc[0] * 100

        corr = s_norm.corr(r_norm)
        ret_s = s.iloc[-1] / s.iloc[0] - 1
        ret_r = r.iloc[-1] / r.iloc[0] - 1

        rs = s.pct_change().dropna()
        rr = r.pct_change().dropna()
        common_ret = rs.index.intersection(rr.index)
        te = (rs.loc[common_ret] - rr.loc[common_ret]).std() * np.sqrt(252)

        suffix = f" ({tag})" if tag else ""
        print(f"  vs {fred_id}{suffix}:")
        print(f"    Overlap:        {common[0].date()} to {common[-1].date()} ({len(common)} days)")
        print(f"    Correlation:    {corr:.4f}")
        print(f"    Tracking Error: {te:.2%}")
        print(f"    Sim Return:     {ret_s:.2%}")
        print(f"    Real Return:    {ret_r:.2%}")
        print(f"    Gap:            {(ret_s - ret_r)*100:+.2f}pp")

        return {"corr": corr, "te": te, "ret_s": ret_s, "ret_r": ret_r,
                "s_norm": s_norm, "r_norm": r_norm, "s": s, "r": r,
                "common": common, "fred_id": fred_id, "tag": tag}

    # Each entry: (label, sim_file, price_return_fred, total_return_fred)
    # NOTE: Our simulation uses yfinance auto_adjust=True (dividend-adjusted prices)
    # so it computes TOTAL RETURN. The primary comparison should be vs the
    # Total Return index; Price Return comparison will show a systematic gap
    # equal to the dividend yield.
    comparisons = [
        ("NDX Mega 1.0", "NDXMEGASIM", "NASDAQNDXMEGA", "NASDAQNDXMEGAT"),
        ("NDX Mega 2.0", "NDXMEGA2SIM", "NASDAQNDXMEGA2", "NASDAQNDXMEGA2T"),
        ("NDX30", "NDX30SIM", "NASDAQNDX30", "NASDAQNDX30T"),
    ]

    for label, sim_name, fred_price, fred_total in comparisons:
        sim_path = os.path.join(config.BASE_DIR, "..", f"{sim_name}.csv")
        if not os.path.exists(sim_path):
            print(f"  Skipping {label}: {sim_path} not found")
            continue

        sim_df = pd.read_csv(sim_path, parse_dates=[0], index_col=0)
        sim_series = sim_df.iloc[:, 0]

        print(f"\n--- {label} (sim = Total Return via adjusted prices) ---")

        # Compare against TOTAL RETURN first (primary — matches our simulation type)
        result_total = compare_pair(label, sim_series, fred_total, "Total Return")

        # Compare against PRICE RETURN (secondary — expect systematic positive gap from dividends)
        result_price = compare_pair(label, sim_series, fred_price, "Price Return")

        # Chart for each available comparison
        for result in [result_price, result_total]:
            if result is None:
                continue

            chart_style.apply_style()
            plt.figure(figsize=(14, 8))
            ax = plt.gca()

            gap_pp = (result["ret_s"] - result["ret_r"]) * 100
            plt.plot(result["s_norm"], label=f'{label} Sim ({result["ret_s"]:.1%})', linewidth=2.5)
            plt.plot(result["r_norm"], label=f'{result["fred_id"]} ({result["tag"]}) ({result["ret_r"]:.1%})',
                     linestyle='--', linewidth=2.0, color='#C44E52')

            chart_style.format_date_axis(ax)
            chart_style.format_y_axis(ax, log=False)
            chart_style.add_watermark(ax, f"{label} vs Real")

            plt.title(f"{label}: Sim vs {result['fred_id']} ({result['tag']})\n"
                       f"Corr: {result['corr']:.4f}, TE: {result['te']:.2%}, Gap: {gap_pp:+.1f}pp")
            plt.legend()

            chart_name = f"{sim_name.lower().replace('sim', '_vs_real')}_{result['tag'].lower().replace(' ', '_')}"
            out_img = os.path.join(config.RESULTS_DIR, "charts", f"{chart_name}.png")
            plt.savefig(out_img, dpi=300, bbox_inches='tight')
            print(f"  Chart saved to {out_img}")

        # --- Per-Quarter Attribution Diagnostic ---
        # Use total return comparison (matches our simulation type) if available
        result = result_total or result_price
        if result is None:
            continue

        s = result["s"]
        r = result["r"]
        # Monthly return gap
        s_monthly = s.resample('ME').last().dropna()
        r_monthly = r.resample('ME').last().dropna()
        common_monthly = s_monthly.index.intersection(r_monthly.index)
        if len(common_monthly) < 2:
            continue

        s_m = s_monthly.loc[common_monthly]
        r_m = r_monthly.loc[common_monthly]
        ret_s_m = s_m.pct_change().dropna()
        ret_r_m = r_m.pct_change().dropna()
        gap_m = ret_s_m - ret_r_m

        # Quarterly attribution table
        s_quarterly = s.resample('QE').last().dropna()
        r_quarterly = r.resample('QE').last().dropna()
        common_q = s_quarterly.index.intersection(r_quarterly.index)

        if len(common_q) >= 2:
            s_q = s_quarterly.loc[common_q]
            r_q = r_quarterly.loc[common_q]
            ret_s_q = s_q.pct_change().dropna()
            ret_r_q = r_q.pct_change().dropna()
            gap_q = ret_s_q - ret_r_q

            print(f"\n  Per-Quarter Attribution ({result['fred_id']}):")
            print(f"  {'Quarter':<12} {'Sim':>8} {'Real':>8} {'Gap':>8}")
            print(f"  {'-'*36}")

            # Sort by absolute gap descending to show biggest mismatches first
            for dt in gap_q.abs().sort_values(ascending=False).index:
                qstr = f"{dt.year}Q{(dt.month-1)//3+1}"
                print(f"  {qstr:<12} {ret_s_q.loc[dt]:>8.2%} {ret_r_q.loc[dt]:>8.2%} {gap_q.loc[dt]*100:>+7.2f}pp")

        # Monthly gap bar chart
        if len(gap_m) >= 2:
            chart_style.apply_style()
            fig, ax = plt.subplots(figsize=(14, 6))

            colors = ['#2ca02c' if g >= 0 else '#d62728' for g in gap_m.values]
            ax.bar(gap_m.index, gap_m.values * 100, width=20, color=colors, alpha=0.8)
            ax.axhline(0, color='black', linewidth=0.5)

            ax.set_ylabel('Monthly Return Gap (pp)')
            ax.set_title(f'{label}: Monthly Return Gap vs {result["fred_id"]}\n'
                         f'Green = sim outperforms, Red = sim underperforms')
            chart_style.format_date_axis(ax)

            chart_name = f"{sim_name.lower().replace('sim', '_attribution')}"
            out_img = os.path.join(config.RESULTS_DIR, "charts", f"{chart_name}.png")
            plt.savefig(out_img, dpi=300, bbox_inches='tight')
            print(f"  Attribution chart saved to {out_img}")



if __name__ == "__main__":
    validate()
    validate_qbig()
    validate_against_real_indexes()
    compare_strategies()
