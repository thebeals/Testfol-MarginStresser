import yfinance as yf
import pandas as pd
import os
import config
from datetime import datetime, timedelta

# Global data source setting (can be overridden by environment variable)
DATA_SOURCE = os.environ.get('NDX_DATA_SOURCE', 'yfinance')
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', None)

def get_price_data(
    tickers,
    start_date='2000-01-01',
    force_refresh=False,
    data_source=None,
    polygon_api_key=None,
    adjust_prices=True,
):
    """
    Fetches price data for the specified tickers.
    Uses a local pickle cache to avoid redundant downloads.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        force_refresh: If True, bypass cache and download fresh
        data_source: 'yfinance' (default), 'polygon', or 'stooq'
        polygon_api_key: API key for Polygon.io (required if data_source='polygon')
        adjust_prices: Use dividend-adjusted closes when True; raw closes when False
    """
    # Determine data source
    source = data_source or DATA_SOURCE
    api_key = polygon_api_key or POLYGON_API_KEY
    if not adjust_prices and source != 'yfinance':
        print("Price-return reconstruction requires raw Yahoo closes; using yfinance.")
        source = 'yfinance'
    
    # Use different cache files for different sources and return conventions.
    # The ordinary simulations use dividend-adjusted prices (total return). The
    # QQUP backfill needs raw closes because QQUP targets the price index.
    if source == 'polygon':
        cache_file = config.PRICE_CACHE_FILE.replace('.pkl', '_polygon.pkl')
    elif source == 'stooq':
        cache_file = config.PRICE_CACHE_FILE.replace('.pkl', '_stooq.pkl')
    else:
        cache_file = config.PRICE_CACHE_FILE
    if not adjust_prices:
        cache_file = cache_file.replace('.pkl', '_price_return.pkl')
    
    # 1. Try Loading Cache
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading price data from cache: {cache_file} ...")
        try:
            df = pd.read_pickle(cache_file)
            
            missing = [t for t in tickers if t not in df.columns]
            present = [t for t in tickers if t in df.columns and df[t].notna().any()]
            pct_missing = len(missing) / len(tickers) if tickers else 0

            # For small ad-hoc requests, a missing ticker usually means we are
            # adding a live benchmark/proxy such as QBIG. Fetch and merge it
            # instead of returning an unrelated cached universe.
            if missing and (len(tickers) <= 5 or not present):
                print(f"Cache missing requested ticker(s): {', '.join(missing)}. Fetching and merging...")
                return download_and_cache(
                    missing, start_date, cache_file, source, api_key, adjust_prices
                )
            
            if pct_missing > 0.25:
                print(f"Cache missing {len(missing)} tickers ({pct_missing:.1%}) — likely historical/delisted, using cache as-is.")
            elif pct_missing > 0.10:
                print(f"Cache missing {len(missing)} tickers ({pct_missing:.1%}) — likely delisted, using cache as-is.")
            
            if pd.to_datetime(start_date) < df.index[0]:
                start_gap = df.index[0] - pd.to_datetime(start_date)
                if start_gap > pd.Timedelta(days=7):
                    print(f"Cache starts {df.index[0].date()}, need {start_date}. Refreshing...")
                    return download_and_cache(
                        tickers, start_date, cache_file, source, api_key, adjust_prices
                    )
                print(f"Cache starts {df.index[0].date()}, close enough to {start_date}. Using cache.")
                 
            return df
            
        except Exception as e:
            print(f"Error loading cache: {e}. Downloading fresh...")
            
    # 2. Download Fresh
    return download_and_cache(tickers, start_date, cache_file, source, api_key, adjust_prices)

def download_and_cache(
    tickers,
    start_date,
    cache_file,
    data_source='yfinance',
    polygon_api_key=None,
    adjust_prices=True,
):
    """Download price data from specified source and cache it."""
    unique_tickers = list(set(tickers))
    
    if data_source == 'polygon':
        if not polygon_api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY env var or pass polygon_api_key parameter.")
        new_df = download_from_polygon(unique_tickers, start_date, polygon_api_key)
    elif data_source == 'stooq':
        new_df = download_from_stooq(unique_tickers, start_date)
    else:
        # Default: Yahoo Finance with Fallback
        new_df = download_from_yfinance(unique_tickers, start_date, adjust_prices=adjust_prices)
        
        # --- Fallback Logic ---
        # Identify missing tickers (requested but not returned OR returned as all NaNs)
        downloaded_tickers = new_df.columns.tolist() if not new_df.empty else []
        
        # Check 1: Tickers not in columns
        missing_tickers = [t for t in unique_tickers if t not in downloaded_tickers]
        
        # Check 2: truly empty tickers. Do not drop partial-history names:
        # recent NDX additions/IPOs can be mostly NaN when the request starts in 1999.
        if not new_df.empty:
            for t in list(downloaded_tickers):
                valid_count = int(new_df[t].notna().sum())
                if valid_count < 2:
                    missing_tickers.append(t)
                    new_df = new_df.drop(columns=[t])
        
        missing_tickers = list(set(missing_tickers))

        if missing_tickers:
            print(f"  yfinance missed {len(missing_tickers)} tickers (likely delisted — wayback merge will fill these)")

            # Optional: Stooq fallback (slow, only if explicitly requested)
            use_stooq = os.environ.get('NDX_USE_STOOQ_FALLBACK', '').lower() in ('1', 'true', 'yes')
            if use_stooq:
                print(f"  [Stooq fallback enabled] Trying {len(missing_tickers)} tickers...")
                stooq_df = download_from_stooq(missing_tickers, start_date)
                if not stooq_df.empty:
                    print(f"  Stooq recovered {len(stooq_df.columns)} tickers")
                    new_df = new_df.join(stooq_df, how='outer').sort_index() if not new_df.empty else stooq_df

    if new_df is None or new_df.empty:
        print("ERROR: No data downloaded!")
        return pd.DataFrame()
        
    # Merge with existing cache to preserve other tickers
    if os.path.exists(cache_file):
        try:
            old_df = pd.read_pickle(cache_file)
            print(f"Merging with existing cache ({len(old_df.columns)} cols)...")
            
            cols_to_drop = [c for c in new_df.columns if c in old_df.columns]
            if cols_to_drop:
                old_df = old_df.drop(columns=cols_to_drop)
                
            df = old_df.join(new_df, how='outer').sort_index()
        except Exception as e:
            print(f"Merge failed ({e}), using fresh data only.")
            df = new_df
    else:
        df = new_df
        
    # Apply successor ticker fallback for acquired companies
    df = apply_successor_fallback(df, adjust_prices=adjust_prices)

    # Merge Wayback Machine recovered prices for delisted tickers
    df = merge_wayback_prices(df, adjust_prices=adjust_prices)

    df.to_pickle(cache_file)
    print(f"Saved {len(df.columns)} tickers to cache: {cache_file}")
    return df


def merge_wayback_prices(df, adjust_prices=True):
    """Merge Wayback Machine scraped prices for delisted tickers.

    Source: data/assets/wayback_prices/{TICKER}.csv
    These were recovered from archived Yahoo Finance pages via the
    WayBackMachineStockScraper (see data/assets/wayback_prices/manifest.json).

    Coverage summary (42 tickers scraped, 38 fill yfinance gaps):
      - 38 delisted NDX tickers recovered that yfinance/stooq cannot provide
      - 4 redundant (ESRX, MEDI, SPLS, SUNW — yfinance has these, wayback is backup)
      - 49 tickers still have NO source (neither yfinance nor wayback)

    Strategy: gap-fill only for existing tickers, add column for new ones.
    Uses "Adj Close" when available, falls back to "Close".
    """
    wayback_dir = os.path.join(config.ASSETS_DIR, "wayback_prices")
    if not os.path.isdir(wayback_dir):
        return df

    csv_files = sorted(
        f for f in os.listdir(wayback_dir)
        if f.endswith('.csv') and f != 'manifest.json'
    )
    if not csv_files:
        return df

    # Track what yfinance is missing so we can report coverage
    all_tickers_needed = set(df.columns)
    missing_before = {t for t in all_tickers_needed if df[t].isna().all()}

    added, gap_filled, skipped = [], [], []

    for fname in csv_files:
        ticker = fname.replace('.csv', '').upper()
        fpath = os.path.join(wayback_dir, fname)
        try:
            wb = pd.read_csv(fpath, parse_dates=["Date"], index_col="Date")
        except Exception:
            continue

        if adjust_prices and "Adj Close" in wb.columns:
            col = "Adj Close"
        else:
            col = "Close"
        series = wb[col].dropna()
        if series.empty:
            continue

        if ticker in df.columns:
            # Gap-fill only — don't overwrite existing data
            na_mask = df[ticker].isna()
            overlap = na_mask.index.intersection(series.index)
            fillable = na_mask.loc[overlap] & series.reindex(overlap).notna()
            if fillable.any():
                df.loc[fillable[fillable].index, ticker] = series.reindex(fillable[fillable].index)
                gap_filled.append((ticker, int(fillable.sum())))
            else:
                skipped.append(ticker)
        else:
            # New ticker — not in yfinance download at all
            df[ticker] = series.reindex(df.index)
            non_null = int(series.reindex(df.index).notna().sum())
            added.append((ticker, non_null))

    missing_after = {t for t in all_tickers_needed if df[t].isna().all()}
    recovered = missing_before - missing_after

    total = len(added) + len(gap_filled)
    if total:
        print(f"\n  [Wayback Machine] Merged {total} tickers from {wayback_dir}")
        if added:
            print(f"    Added {len(added)} new: {', '.join(t for t, _ in added)}")
        if gap_filled:
            print(f"    Gap-filled {len(gap_filled)}: {', '.join(f'{t}({n})' for t, n in gap_filled)}")
        if skipped:
            print(f"    Skipped {len(skipped)} (already complete): {', '.join(skipped)}")
        if recovered:
            print(f"    Recovered from all-NaN: {', '.join(sorted(recovered))}")

    return df

def apply_successor_fallback(df, adjust_prices=True):
    """For delisted tickers with no data, try to fill with successor ticker data.
    Downloads successor tickers that aren't already in the cache."""
    try:
        from mapper import SUCCESSOR_TICKERS
    except ImportError:
        return df

    def parse_successor_spec(spec):
        if isinstance(spec, dict):
            successor = spec.get('successor')
            effective_date = spec.get('effective_date')
        else:
            successor = spec
            effective_date = None

        if effective_date:
            effective_date = pd.Timestamp(effective_date)
        return successor, effective_date

    successors_needed = {}
    for t, spec in SUCCESSOR_TICKERS.items():
        succ, effective_date = parse_successor_spec(spec)
        if succ and succ != t:
            # Fill if ticker is missing OR has >50% NaN (matching yfinance threshold)
            if t not in df.columns or df[t].isna().mean() > 0.50:
                successors_needed[t] = {
                    'successor': succ,
                    'effective_date': effective_date,
                }

    if not successors_needed:
        return df

    # Download successor tickers that aren't in the dataframe
    missing_succs = list(set(
        spec['successor'] for spec in successors_needed.values()
        if spec['successor'] not in df.columns or df[spec['successor']].isna().all()
    ))
    if missing_succs:
        import time
        print(f"  Downloading {len(missing_succs)} successor tickers: {missing_succs}")
        for succ in missing_succs:
            try:
                single = yf.download(
                    succ,
                    start='2000-01-01',
                    auto_adjust=adjust_prices,
                    progress=False,
                )
                if not single.empty:
                    close = single['Close'] if 'Close' in single.columns else single.iloc[:, 0]
                    if hasattr(close, 'columns'):
                        close = close.iloc[:, 0]
                    if close.notna().sum() > 50:
                        close.index = pd.to_datetime(close.index)
                        df[succ] = close.reindex(df.index)
                        print(f"    Downloaded {succ}: {close.notna().sum()} days")
                time.sleep(0.1)
            except Exception as e:
                print(f"    Failed to download {succ}: {e}")

    filled = 0
    for orig, spec in successors_needed.items():
        succ = spec['successor']
        effective_date = spec['effective_date']

        if succ not in df.columns or df[succ].isna().all():
            continue

        if orig not in df.columns:
            df[orig] = pd.Series(index=df.index, dtype=float)

        existing = df[orig].copy()
        if existing.notna().any():
            start_date = existing.dropna().index.max() + pd.offsets.BDay(1)
        else:
            start_date = df.index.min()

        if effective_date is not None:
            start_date = max(start_date, effective_date)

        fill_mask = (
            (df.index >= start_date)
            & df[orig].isna()
            & df[succ].notna()
        )

        if fill_mask.any():
            df.loc[fill_mask, orig] = df.loc[fill_mask, succ]
            filled += 1

    if filled:
        print(f"  Successor fallback: filled {filled} delisted tickers with date-aware acquirer data")

    return df

def download_from_yfinance(tickers, start_date, adjust_prices=True):
    """Download price data from yfinance."""
    print(f"[yfinance] Downloading prices for {len(tickers)} tickers from {start_date}...")
    
    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=adjust_prices,
        progress=True,
    )

    if data.empty:
        return pd.DataFrame()

    ticker_list = list(tickers)
    single_ticker = ticker_list[0] if len(ticker_list) == 1 else None

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            close = data['Close']
        else:
            close = data
    elif 'Close' in data.columns:
        close = data['Close']
    else:
        close = data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=single_ticker or close.name or 'Close')
    elif single_ticker and len(close.columns) == 1:
        close = close.rename(columns={close.columns[0]: single_ticker})

    return close

def download_from_stooq(tickers, start_date):
    """Download price data directly from Stooq CSV endpoint (no pandas-datareader)."""
    import time
    import io
    import requests

    print(f"[Stooq Direct] Downloading prices for {len(tickers)} tickers from {start_date}...")

    all_data = {}
    failed_tickers = []

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
    })

    start_dt = pd.to_datetime(start_date)

    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} tickers...")

        # Stooq URL format
        stooq_ticker = f"{ticker}.US" if not ticker.startswith('^') else ticker
        url = (f"https://stooq.com/q/d/l/"
               f"?s={stooq_ticker}"
               f"&d1={start_dt.strftime('%Y%m%d')}"
               f"&d2={pd.Timestamp.now().strftime('%Y%m%d')}"
               f"&i=d")

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and 'Date' in resp.text[:50]:
                df = pd.read_csv(io.StringIO(resp.text), parse_dates=['Date'], index_col='Date')
                if not df.empty and 'Close' in df.columns:
                    series = df['Close'].sort_index()
                    if len(series) > 10:
                        all_data[ticker] = series
                    else:
                        failed_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

        time.sleep(0.5)  # Rate limit

    if failed_tickers:
        print(f"  Failed: {len(failed_tickers)} tickers")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"  Downloaded {len(df.columns)} tickers, {len(df)} trading days")
    return df

def download_from_polygon(tickers, start_date, api_key):
    """
    Download price data from Polygon.io.
    
    Polygon.io provides more complete historical data than yfinance,
    including data for delisted tickers.
    """
    try:
        from polygon import RESTClient
    except ImportError:
        print("ERROR: polygon-api-client not installed!")
        print("Install with: pip install polygon-api-client")
        raise ImportError("Please install polygon-api-client: pip install polygon-api-client")
    
    print(f"[Polygon.io] Downloading prices for {len(tickers)} tickers from {start_date}...")
    
    client = RESTClient(api_key)
    
    # Convert dates
    start_dt = pd.to_datetime(start_date)
    end_dt = datetime.now()
    
    all_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)} tickers...")
            
        try:
            # Get daily aggregates
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_dt.strftime('%Y-%m-%d'),
                to=end_dt.strftime('%Y-%m-%d'),
                adjusted=True,
                sort="asc",
                limit=50000  # Max allowed
            )
            
            if aggs:
                # Convert to DataFrame
                dates = [pd.to_datetime(a.timestamp, unit='ms') for a in aggs]
                closes = [a.close for a in aggs]
                all_data[ticker] = pd.Series(closes, index=dates, name=ticker)
            else:
                print(f"  Warning: No data for {ticker}")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"  Failed tickers: {len(failed_tickers)} ({', '.join(failed_tickers[:10])}{'...' if len(failed_tickers) > 10 else ''})")
    
    if not all_data:
        print("ERROR: No data downloaded from Polygon!")
        return pd.DataFrame()
    
    # Combine into DataFrame
    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"  Downloaded {len(df)} trading days for {len(df.columns)} tickers")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return df
