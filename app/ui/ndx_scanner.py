"""
NDX-100 Moving Average Scanner

Scans all current Nasdaq 100 components and identifies which are trading
below their 200-Day Moving Average or 200-Week Moving Average (Munger).
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import os
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core import calculations
from app.services.ndx_components import load_current_ndx_components
from app.services import testfol_api as api

# Rate limiting for Testfol API
TESTFOL_RATE_LIMIT = 5  # requests per second
TESTFOL_MAX_WORKERS = 3  # concurrent threads


# Paths to data files
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/ndx_simulation/data")
NDX_COMPONENTS_FILE = os.path.join(DATA_DIR, "assets/nasdaq_components.csv")
NAME_MAPPING_FILE = os.path.join(DATA_DIR, "assets/name_mapping.json")


@st.cache_data(ttl=86400, show_spinner=False)  # Daily membership refresh
def get_current_ndx_components():
    """
    Get current Nasdaq-100 membership from Nasdaq, verified against Wikipedia.

    Falls back to Wikipedia, then the local SEC-derived snapshot when live
    sources are unavailable.
    """
    try:
        snapshot = load_current_ndx_components(
            NDX_COMPONENTS_FILE,
            NAME_MAPPING_FILE,
        )
        return (
            snapshot.components,
            snapshot.as_of,
            snapshot.source,
            snapshot.weight_basis,
            snapshot.warning,
        )
    except Exception as e:
        st.error(f"Failed to load NDX components: {e}")
        return pd.DataFrame(), None, "Unavailable", "Unavailable", None


@st.cache_data(ttl=900, show_spinner=False)  # 15-minute cache
def fetch_ma_data(tickers: list, tolerance_days: int = 0, min_days_filter: int = 0):
    """
    Fetch FULL historical price data and calculate 200-day SMA for all tickers.
    Returns DataFrame with: Ticker, Price, SMA_200, Distance_Pct, Status, Max Depth, Duration, Depth Rank
    """
    if not tickers:
        return pd.DataFrame()

    results = []

    # Batch download - get FULL history for depth ranking
    try:
        data = yf.download(
            tickers,
            period="max",  # Full history for accurate depth ranking
            auto_adjust=True,
            threads=True,
            progress=False,
            timeout=30,
        )['Close']

        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])

        for ticker in tickers:
            try:
                if ticker not in data.columns:
                    continue

                series = data[ticker].dropna()

                if len(series) < 200:
                    continue

                # Use shared analysis logic (handles Merge Tolerance)
                dma_series, events_df = calculations.analyze_ma(series, window=200, tolerance_days=tolerance_days)

                if dma_series is None or dma_series.empty:
                    continue

                current_price = series.iloc[-1]
                sma_200 = dma_series.iloc[-1]

                distance_pct = ((current_price - sma_200) / sma_200) * 100

                status = "🟢 Above"
                duration = 0
                max_deviation = 0.0
                depth_rank = None  # Numeric for sorting
                total_breaches = 0

                # Determine Status from Events
                if not events_df.empty:
                    # Apply min_days_filter to get filtered events for ranking
                    filtered_events = events_df[events_df["Duration (Days)"] >= min_days_filter] if min_days_filter > 0 else events_df

                    # Get all historical max depths for ranking
                    all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                    total_breaches = len(all_depths)

                    last_event = events_df.iloc[-1]

                    if last_event["Status"] == "Ongoing":
                        status = "🔴 Below"
                        duration = last_event["Duration (Days)"]

                        # Calculate current max depth
                        start_date = last_event["Start Date"]
                        event_prices = series[start_date:]

                        if not event_prices.empty:
                            start_price = event_prices.iloc[0]
                            min_price = event_prices.min()
                            max_deviation = ((min_price - start_price) / start_price) * 100

                        # Calculate depth rank (how current depth ranks among all historical)
                        if total_breaches > 0 and max_deviation < 0:
                            # Sort depths from deepest (most negative) to shallowest
                            sorted_depths = sorted(all_depths)  # Most negative first
                            # Find rank (1 = deepest)
                            rank = 1
                            for d in sorted_depths:
                                if max_deviation <= d:
                                    break
                                rank += 1
                            depth_rank = rank  # Numeric for proper sorting
                    else:
                        # Currently Above
                        last_end = last_event["End Date"]
                        if pd.notna(last_end):
                            duration = (series.index[-1] - last_end).days

                            valid_range = series.index > last_end
                            above_prices = series[valid_range]
                            above_sma = dma_series[valid_range]

                            if not above_prices.empty:
                                deviations = ((above_prices - above_sma) / above_sma) * 100
                                max_deviation = deviations.max()
                        else:
                            duration = 0
                else:
                    # Never below SMA
                    duration = len(series)
                    deviations = ((series - dma_series) / dma_series) * 100
                    max_deviation = deviations.max()

                results.append({
                    'Ticker': ticker,
                    'Price': current_price,
                    'SMA_200': sma_200,
                    'Distance %': distance_pct,
                    'Max Depth / Peak': max_deviation,
                    'Depth Rank': depth_rank,
                    'Total Breaches': total_breaches,
                    'Status': status,
                    'Duration': duration
                })

            except Exception:
                continue

    except Exception as e:
        st.error(f"Failed to fetch price data: {e}")
        return pd.DataFrame()

    return pd.DataFrame(results)


@st.cache_data(ttl=900, show_spinner=False)  # 15-minute cache
def fetch_wma_data(tickers: list, tolerance_weeks: int = 0, min_weeks_filter: int = 0):
    """
    Fetch FULL historical price data and calculate 200-week WMA for all tickers.
    Returns DataFrame with: Ticker, Price, WMA_200, Distance_Pct, Status, Max Depth, Duration (weeks), Depth Rank
    """
    if not tickers:
        return pd.DataFrame()

    results = []

    try:
        # Get FULL history for depth ranking
        data = yf.download(
            tickers,
            period="max",  # Full history for accurate depth ranking
            auto_adjust=True,
            threads=True,
            progress=False,
            timeout=30,
        )['Close']

        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0])

        for ticker in tickers:
            try:
                if ticker not in data.columns:
                    continue

                series = data[ticker].dropna()

                if len(series) < 200 * 5:  # Need ~200 weeks of daily data
                    continue

                # Use the weekly MA analysis
                weekly_series, wma_series, events_df = calculations.analyze_wma(
                    series, window=200, tolerance_weeks=tolerance_weeks
                )

                if wma_series is None or wma_series.empty:
                    continue

                current_price = weekly_series.iloc[-1]
                wma_200 = wma_series.iloc[-1]

                distance_pct = ((current_price - wma_200) / wma_200) * 100

                status = "🟢 Above"
                duration = 0
                max_deviation = 0.0
                depth_rank = None  # Numeric for sorting
                total_breaches = 0

                # Determine Status from Events
                if not events_df.empty:
                    # Apply min_weeks_filter to get filtered events for ranking
                    filtered_events = events_df[events_df["Duration (Weeks)"] >= min_weeks_filter] if min_weeks_filter > 0 else events_df

                    # Get all historical max depths for ranking
                    all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                    total_breaches = len(all_depths)

                    last_event = events_df.iloc[-1]

                    if last_event["Status"] == "Ongoing":
                        status = "🔴 Below"
                        duration = last_event["Duration (Weeks)"]

                        # Calculate Max Depth for this event
                        start_date = last_event["Start Date"]
                        event_prices = weekly_series[start_date:]

                        if not event_prices.empty:
                            start_price = event_prices.iloc[0]
                            min_price = event_prices.min()
                            max_deviation = ((min_price - start_price) / start_price) * 100

                        # Calculate depth rank (how current depth ranks among all historical)
                        if total_breaches > 0 and max_deviation < 0:
                            # Sort depths from deepest (most negative) to shallowest
                            sorted_depths = sorted(all_depths)  # Most negative first
                            # Find rank (1 = deepest)
                            rank = 1
                            for d in sorted_depths:
                                if max_deviation <= d:
                                    break
                                rank += 1
                            depth_rank = rank  # Numeric for proper sorting
                    else:
                        # Last event recovered - currently Above
                        last_end = last_event["End Date"]
                        if pd.notna(last_end):
                            # Count weeks since recovery
                            duration = len(weekly_series[weekly_series.index > last_end])

                            # Calculate Max Peak for this 'Above' period
                            valid_range = weekly_series.index > last_end
                            above_prices = weekly_series[valid_range]
                            above_wma = wma_series[valid_range]

                            if not above_prices.empty and not above_wma.empty:
                                deviations = ((above_prices - above_wma) / above_wma) * 100
                                max_deviation = deviations.max()
                        else:
                            duration = 0
                else:
                    # Never below WMA in this period
                    duration = len(weekly_series)
                    deviations = ((weekly_series - wma_series) / wma_series) * 100
                    max_deviation = deviations.dropna().max() if not deviations.dropna().empty else 0.0

                results.append({
                    'Ticker': ticker,
                    'Price': current_price,
                    'WMA_200': wma_200,
                    'Distance %': distance_pct,
                    'Max Depth / Peak': max_deviation,
                    'Depth Rank': depth_rank,
                    'Total Breaches': total_breaches,
                    'Status': status,
                    'Duration': duration
                })

            except Exception:
                continue

    except Exception as e:
        st.error(f"Failed to fetch price data: {e}")
        return pd.DataFrame()

    return pd.DataFrame(results)


def _fetch_single_ticker_testfol(ticker: str, start_date: str, end_date: str, bearer_token: str = None) -> tuple:
    """
    Fetch a single ticker from Testfol API.
    Returns (ticker, series) or (ticker, None) on failure.
    """
    try:
        result = api.fetch_backtest(
            start_date=start_date,
            end_date=end_date,
            start_val=10000,
            cashflow=0,
            cashfreq="Never",
            rolling=False,
            invest_div=True,
            rebalance="Never",
            allocation={ticker: 100},
            bearer_token=bearer_token
        )

        if result and 'series' in result:
            series = result['series']
            if isinstance(series, pd.DataFrame) and ticker in series.columns:
                return (ticker, series[ticker].dropna())
            elif isinstance(series, pd.Series):
                return (ticker, series.dropna())
        return (ticker, None)
    except Exception:
        return (ticker, None)


@st.cache_data(ttl=900, show_spinner=False)  # 15-minute cache
def fetch_ma_data_testfol(tickers: list, tolerance_days: int = 0, min_days_filter: int = 0, bearer_token: str = None):
    """
    Fetch historical price data from Testfol API and calculate 200-day SMA.
    Uses rate limiting and parallel requests.
    """
    if not tickers:
        return pd.DataFrame()

    results = []
    ticker_series = {}

    # Set date range for full history
    start_date = "1884-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    # Rate-limited parallel fetching
    completed = 0
    last_request_time = 0

    with ThreadPoolExecutor(max_workers=TESTFOL_MAX_WORKERS) as executor:
        futures = {}

        for ticker in tickers:
            # Rate limiting
            elapsed = time.time() - last_request_time
            min_interval = 1.0 / TESTFOL_RATE_LIMIT
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            future = executor.submit(_fetch_single_ticker_testfol, ticker, start_date, end_date, bearer_token)
            futures[future] = ticker
            last_request_time = time.time()

        for future in as_completed(futures):
            ticker, series = future.result()
            completed += 1
            if series is not None and len(series) >= 200:
                ticker_series[ticker] = series

    # Process each ticker's data
    for ticker, series in ticker_series.items():
        try:
            # Use shared analysis logic
            dma_series, events_df = calculations.analyze_ma(series, window=200, tolerance_days=tolerance_days)

            if dma_series is None or dma_series.empty:
                continue

            current_price = series.iloc[-1]
            sma_200 = dma_series.iloc[-1]
            distance_pct = ((current_price - sma_200) / sma_200) * 100

            status = "🟢 Above"
            duration = 0
            max_deviation = 0.0
            depth_rank = None
            total_breaches = 0

            if not events_df.empty:
                filtered_events = events_df[events_df["Duration (Days)"] >= min_days_filter] if min_days_filter > 0 else events_df
                all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                total_breaches = len(all_depths)

                last_event = events_df.iloc[-1]

                if last_event["Status"] == "Ongoing":
                    status = "🔴 Below"
                    duration = last_event["Duration (Days)"]

                    start_date_event = last_event["Start Date"]
                    event_prices = series[start_date_event:]

                    if not event_prices.empty:
                        start_price = event_prices.iloc[0]
                        min_price = event_prices.min()
                        max_deviation = ((min_price - start_price) / start_price) * 100

                    if total_breaches > 0 and max_deviation < 0:
                        sorted_depths = sorted(all_depths)
                        rank = 1
                        for d in sorted_depths:
                            if max_deviation <= d:
                                break
                            rank += 1
                        depth_rank = rank
                else:
                    last_end = last_event["End Date"]
                    if pd.notna(last_end):
                        duration = (series.index[-1] - last_end).days
                        valid_range = series.index > last_end
                        above_prices = series[valid_range]
                        above_sma = dma_series[valid_range]

                        if not above_prices.empty:
                            deviations = ((above_prices - above_sma) / above_sma) * 100
                            max_deviation = deviations.max()
            else:
                duration = len(series)
                deviations = ((series - dma_series) / dma_series) * 100
                max_deviation = deviations.max()

            results.append({
                'Ticker': ticker,
                'Price': current_price,
                'SMA_200': sma_200,
                'Distance %': distance_pct,
                'Max Depth / Peak': max_deviation,
                'Depth Rank': depth_rank,
                'Total Breaches': total_breaches,
                'Status': status,
                'Duration': duration
            })

        except Exception:
            continue

    return pd.DataFrame(results)


@st.cache_data(ttl=900, show_spinner=False)  # 15-minute cache
def fetch_wma_data_testfol(tickers: list, tolerance_weeks: int = 0, min_weeks_filter: int = 0, bearer_token: str = None):
    """
    Fetch historical price data from Testfol API and calculate 200-week WMA.
    Uses rate limiting and parallel requests.
    """
    if not tickers:
        return pd.DataFrame()

    results = []
    ticker_series = {}

    # Set date range for full history
    start_date = "1884-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    # Rate-limited parallel fetching
    completed = 0
    last_request_time = 0

    with ThreadPoolExecutor(max_workers=TESTFOL_MAX_WORKERS) as executor:
        futures = {}

        for ticker in tickers:
            # Rate limiting
            elapsed = time.time() - last_request_time
            min_interval = 1.0 / TESTFOL_RATE_LIMIT
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            future = executor.submit(_fetch_single_ticker_testfol, ticker, start_date, end_date, bearer_token)
            futures[future] = ticker
            last_request_time = time.time()

        for future in as_completed(futures):
            ticker, series = future.result()
            completed += 1
            if series is not None and len(series) >= 200:
                ticker_series[ticker] = series

    # Process each ticker's data
    for ticker, series in ticker_series.items():
        try:
            # Use shared WMA analysis logic
            weekly_series, wma_series, events_df = calculations.analyze_wma(series, window=200, tolerance_weeks=tolerance_weeks)

            if wma_series is None or wma_series.empty:
                continue

            current_price = weekly_series.iloc[-1]
            wma_200 = wma_series.iloc[-1]
            distance_pct = ((current_price - wma_200) / wma_200) * 100

            status = "🟢 Above"
            duration = 0
            max_deviation = 0.0
            depth_rank = None
            total_breaches = 0

            if not events_df.empty:
                filtered_events = events_df[events_df["Duration (Weeks)"] >= min_weeks_filter] if min_weeks_filter > 0 else events_df
                all_depths = filtered_events["Max Depth (%)"].dropna().tolist()
                total_breaches = len(all_depths)

                last_event = events_df.iloc[-1]

                if last_event["Status"] == "Ongoing":
                    status = "🔴 Below"
                    duration = last_event["Duration (Weeks)"]

                    start_date_event = last_event["Start Date"]
                    event_prices = weekly_series[start_date_event:]

                    if not event_prices.empty:
                        start_price = event_prices.iloc[0]
                        min_price = event_prices.min()
                        max_deviation = ((min_price - start_price) / start_price) * 100

                    if total_breaches > 0 and max_deviation < 0:
                        sorted_depths = sorted(all_depths)
                        rank = 1
                        for d in sorted_depths:
                            if max_deviation <= d:
                                break
                            rank += 1
                        depth_rank = rank
                else:
                    last_end = last_event["End Date"]
                    if pd.notna(last_end):
                        duration = len(weekly_series[weekly_series.index > last_end])
                        valid_range = weekly_series.index > last_end
                        above_prices = weekly_series[valid_range]
                        above_wma = wma_series[valid_range]

                        if not above_prices.empty and not above_wma.empty:
                            deviations = ((above_prices - above_wma) / above_wma) * 100
                            max_deviation = deviations.max()
            else:
                duration = len(weekly_series)
                deviations = ((weekly_series - wma_series) / wma_series) * 100
                max_deviation = deviations.dropna().max() if not deviations.dropna().empty else 0.0

            results.append({
                'Ticker': ticker,
                'Price': current_price,
                'WMA_200': wma_200,
                'Distance %': distance_pct,
                'Max Depth / Peak': max_deviation,
                'Depth Rank': depth_rank,
                'Total Breaches': total_breaches,
                'Status': status,
                'Duration': duration
            })

        except Exception:
            continue

    return pd.DataFrame(results)


@st.fragment
def render_ndx_scanner():
    """
    Main renderer for the NDX-100 Moving Average Scanner.
    """
    st.header("📊 NDX-100 Moving Average Scanner")

    # MA Type selector
    ma_type = st.radio(
        "Moving Average Type",
        ["200 SMA (Daily)", "200 WMA (Weekly - Munger)"],
        horizontal=True,
        help="**200 SMA**: 200-day moving average (short-term trends)\n\n**200 WMA**: 200-week moving average (~4 years, secular trends)"
    )

    is_weekly = ma_type == "200 WMA (Weekly - Munger)"
    ma_label = "200 WMA" if is_weekly else "200 SMA"

    if is_weekly:
        st.caption("Identify Nasdaq 100 components trading below their 200-Week Moving Average (Munger Indicator)")
    else:
        st.caption("Identify Nasdaq 100 components trading below their 200-Day Moving Average")

    # Load components
    (
        components_df,
        latest_date,
        component_source,
        weight_basis,
        component_warning,
    ) = get_current_ndx_components()

    # Assign Rank by Weight (it's already sorted by Weight)
    if not components_df.empty:
        components_df['Rank'] = range(1, len(components_df) + 1)

    if components_df.empty:
        st.warning("No NDX component data available. Run the NDX rebuild script first.")
        return

    st.info(
        f"📅 **Membership as of:** {latest_date.strftime('%Y-%m-%d')} | "
        f"**Securities:** {len(components_df)} | **Source:** {component_source}"
    )
    st.caption(f"Weight/rank basis: {weight_basis}")
    if component_warning:
        st.warning(component_warning)

    # Check if bearer token is available (session state or env var)
    has_bearer_token = bool(
        st.session_state.get("_bearer_token") or
        os.environ.get("TESTFOL_API_KEY")
    )

    # Filter controls - different defaults for weekly vs daily
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if has_bearer_token:
            use_testfol = st.checkbox(
                "Use Testfol Data",
                value=False,
                help="Use Testfol API instead of yfinance. Slower (~20-30s) but consistent with backtest data and deeper history."
            )
        else:
            use_testfol = False
            st.checkbox(
                "Use Testfol Data",
                value=False,
                disabled=True,
                help="⚠️ Requires Bearer Token in sidebar API Settings (or TESTFOL_API_KEY env var)"
            )
    with c2:
        st.write("")  # Spacer
    with c3:
        if is_weekly:
            use_filters = st.checkbox(
                "Apply Noise Filter (2w)",
                value=True,
                help="Merges events with <2 week recovery gaps and ignores drops <4 weeks duration"
            )
            tol_param = 2 if use_filters else 0
            min_filter = 4 if use_filters else 0
        else:
            use_filters = st.checkbox(
                "Apply Noise Filter (14d)",
                value=True,
                help="Merges events with <14d recovery gaps and ignores drops <14d duration"
            )
            tol_param = 14 if use_filters else 0
            min_filter = 14 if use_filters else 0

    # Get tickers list
    tickers = components_df['Ticker'].tolist()

    # Get bearer token for Testfol API
    bearer_token = st.session_state.get("_bearer_token") or os.environ.get("TESTFOL_API_KEY")

    # Fetch MA data with progress
    if use_testfol:
        data_source = "Testfol API"
        if is_weekly:
            with st.spinner(f"Fetching weekly data from Testfol API for {len(tickers)} tickers (rate limited, ~20-30s)..."):
                ma_data = fetch_wma_data_testfol(tickers, tolerance_weeks=tol_param, min_weeks_filter=min_filter, bearer_token=bearer_token)
            ma_col = 'WMA_200'
        else:
            with st.spinner(f"Fetching daily data from Testfol API for {len(tickers)} tickers (rate limited, ~20-30s)..."):
                ma_data = fetch_ma_data_testfol(tickers, tolerance_days=tol_param, min_days_filter=min_filter, bearer_token=bearer_token)
            ma_col = 'SMA_200'
    else:
        data_source = "yfinance"
        if is_weekly:
            with st.spinner(f"Fetching weekly price data for {len(tickers)} tickers (this may take longer)..."):
                ma_data = fetch_wma_data(tickers, tolerance_weeks=tol_param, min_weeks_filter=min_filter)
            ma_col = 'WMA_200'
        else:
            with st.spinner(f"Fetching price data for {len(tickers)} tickers..."):
                ma_data = fetch_ma_data(tickers, tolerance_days=tol_param, min_days_filter=min_filter)
            ma_col = 'SMA_200'

    if ma_data.empty:
        st.error("Failed to fetch moving average data.")
        return

    # Show data source
    st.caption(f"📊 Data source: **{data_source}** | Tickers loaded: {len(ma_data)}/{len(tickers)}")

    # Merge with component names
    result_df = ma_data.merge(
        components_df[['Ticker', 'Name', 'Weight', 'Rank']],
        on='Ticker',
        how='left'
    )

    # Reorder columns
    result_df = result_df[['Rank', 'Ticker', 'Name', 'Weight', 'Price', ma_col, 'Distance %', 'Max Depth / Peak', 'Depth Rank', 'Total Breaches', 'Duration', 'Status']]

    # Rename MA column for display
    result_df = result_df.rename(columns={ma_col: ma_label})

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        filter_option = st.selectbox(
            "Filter",
            ["All", f"🔴 Below {ma_label}", f"🟢 Above {ma_label}"],
            index=0
        )

    with col2:
        sort_option = st.selectbox(
            "Sort By",
            ["Distance % (Ascending)", "Distance % (Descending)", "Depth Rank (Deepest First)", "Depth Rank (Shallowest First)", "Max Depth / Peak (Ascending)", "Max Depth / Peak (Descending)", "Weight (Descending)", "Ticker (A-Z)"],
            index=0
        )

    # Apply filter
    filtered_df = result_df.copy()

    if "Below" in filter_option:
        filtered_df = filtered_df[filtered_df['Distance %'] < 0]
        if use_filters:
            filtered_df = filtered_df[filtered_df['Duration'] >= min_filter]

    elif "Above" in filter_option:
        filtered_df = filtered_df[filtered_df['Distance %'] >= 0]

    # Apply sort
    if sort_option == "Distance % (Ascending)":
        filtered_df = filtered_df.sort_values('Distance %', ascending=True)
    elif sort_option == "Distance % (Descending)":
        filtered_df = filtered_df.sort_values('Distance %', ascending=False)
    elif sort_option == "Depth Rank (Deepest First)":
        filtered_df = filtered_df.sort_values('Depth Rank', ascending=True, na_position='last')
    elif sort_option == "Depth Rank (Shallowest First)":
        filtered_df = filtered_df.sort_values('Depth Rank', ascending=False, na_position='last')
    elif sort_option == "Max Depth / Peak (Ascending)":
        filtered_df = filtered_df.sort_values('Max Depth / Peak', ascending=True)
    elif sort_option == "Max Depth / Peak (Descending)":
        filtered_df = filtered_df.sort_values('Max Depth / Peak', ascending=False)
    elif sort_option == "Weight (Descending)":
        filtered_df = filtered_df.sort_values('Weight', ascending=False)
    elif sort_option == "Ticker (A-Z)":
        filtered_df = filtered_df.sort_values('Ticker', ascending=True)

    # Summary stats
    below_count = len(result_df[result_df['Distance %'] < 0])
    above_count = len(result_df[result_df['Distance %'] >= 0])

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric(f"🔴 Below {ma_label}", below_count)
    col_s2.metric(f"🟢 Above {ma_label}", above_count)
    col_s3.metric("Total Scanned", len(result_df))
    
    # --- Quick Analyze Action ---
    st.markdown("##### 🚀 Quick Analyze")
    qa_col1, qa_col2 = st.columns([3, 1])
    
    with qa_col1:
        # Get list of tickers from the current filtered view
        available_tickers = filtered_df['Ticker'].tolist()
        selected_ticker_analyze = st.selectbox(
            "Select Ticker to Analyze", 
            available_tickers if available_tickers else ["No tickers available"],
            key="scanner_ticker_select"
        )
        
    with qa_col2:
        # Align button with the selectbox input (compensate for label height)
        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        if st.button("Load Stock", type="primary", use_container_width=True, disabled=not available_tickers):
            if "portfolios" in st.session_state:
                import uuid

                # Set wide date range - Testfol API will return whatever's available
                from datetime import date
                st.session_state._set_start_date = date(1884, 1, 1)
                st.session_state._set_end_date = date.today()

                # Flag to auto-run backtest on next page load
                st.session_state._auto_run_backtest = True

                new_id = f"p_scan_{uuid.uuid4().hex[:8]}"
                new_name = f"Analysis: {selected_ticker_analyze}"

                new_port = {
                    "id": new_id,
                    "name": new_name,
                    "alloc_df": pd.DataFrame([
                        {"Ticker": selected_ticker_analyze, "Weight %": 100.0, "Maint %": 25.0}
                    ]),
                    "rebalance": {
                        "mode": "Standard",
                        "freq": "Yearly",
                        "month": 1,
                        "day": 1,
                        "compare_std": False
                    },
                    "cashflow": {
                        "start_val": 10000.0,
                        "amount": 0.0,
                        "freq": "Monthly",
                        "invest_div": True,
                        "pay_down_margin": False
                    },
                    "dca": {"mode": "Proportional", "target_ticker": ""}
                }

                # Clear all portfolios and set only this one
                st.session_state.portfolios = [new_port]
                st.session_state.active_tab_idx = 0

                if "portfolio_selector" in st.session_state:
                    del st.session_state.portfolio_selector

                # Pop stable per-portfolio widget keys and set p_name explicitly
                # (fragment widget cache may not pick up new defaults from pop alone)
                for _k in ["p_rmode", "p_rfreq", "p_rmon", "p_rday",
                           "p_cmp", "p_rthresh", "p_rfreq_tc", "p_rthresh_tc",
                           "p_rfreq_std", "p_editor"]:
                    st.session_state.pop(_k, None)
                st.session_state["p_name"] = new_name

                st.toast(f"✅ Loaded '{selected_ticker_analyze}' with full history! Switch to Portfolio tab.", icon="🚀")

                # Rerun full app (we're inside @st.fragment)
                time.sleep(0.3)
                st.rerun(scope="app")
            else:
                st.error("Portfolio state not initialized.")
    
    # Display table with column configs for tooltips
    duration_unit = "weeks" if is_weekly else "days"
    ma_full_name = "200-Week Moving Average" if is_weekly else "200-Day Moving Average"

    column_config = {
        "Rank": st.column_config.NumberColumn(
            "Rank",
            help=f"Nasdaq-100 component ranking based on {weight_basis}",
            format="%d"
        ),
        "Ticker": st.column_config.TextColumn(
            "Ticker",
            help="Stock ticker symbol"
        ),
        "Name": st.column_config.TextColumn(
            "Name",
            help="Company name"
        ),
        "Weight": st.column_config.NumberColumn(
            "Market-cap proxy %" if component_source.startswith("Nasdaq") else "Weight %",
            help=f"Component weight basis: {weight_basis}",
            format="%.2f%%"
        ),
        "Price": st.column_config.NumberColumn(
            "Price",
            help="Current stock price",
            format="$%.2f"
        ),
        ma_label: st.column_config.NumberColumn(
            ma_label,
            help=f"{ma_full_name} value",
            format="$%.2f"
        ),
        "Distance %": st.column_config.NumberColumn(
            "Distance %",
            help=f"Percentage distance from {ma_label}. Negative = below MA, Positive = above MA",
            format="%+.2f%%"
        ),
        "Max Depth / Peak": st.column_config.NumberColumn(
            "Max Depth / Peak",
            help=f"For stocks BELOW {ma_label}: Maximum drawdown from when price crossed below MA. For stocks ABOVE: Maximum peak above MA since recovery.",
            format="%+.2f%%"
        ),
        "Depth Rank": st.column_config.NumberColumn(
            "Depth Rank",
            help=f"Ranks current drawdown among ALL historical {ma_label} breaches. 1 = deepest ever. Empty = currently above MA.",
            format="%d"
        ),
        "Total Breaches": st.column_config.NumberColumn(
            "Total Breaches",
            help=f"Total number of {ma_label} breach events in the stock's history (with filtering applied).",
            format="%d"
        ),
        "Duration": st.column_config.NumberColumn(
            "Duration",
            help=f"For stocks BELOW {ma_label}: {duration_unit.capitalize()} since price dropped below MA. For stocks ABOVE: {duration_unit.capitalize()} since price recovered above MA.",
            format="%d"
        ),
        "Status": st.column_config.TextColumn(
            "Status",
            help=f"Current position relative to {ma_label}"
        )
    }

    # Apply styling for status colors
    styled_df = filtered_df.style.map(
        lambda x: 'color: #ff4b4b' if '🔴' in str(x) else 'color: #21c354',
        subset=['Status']
    )

    st.dataframe(
        styled_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    st.caption(f"💡 Data cached for 15 minutes. Duration is in {duration_unit}. Refresh the page to update.")
