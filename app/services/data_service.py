from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import warnings
from functools import lru_cache

import pandas as pd

from app.common.cache import cache_get, cache_key, cache_set
from app.common.constants import Tickers
from app.common.special_tickers import (
    is_testfol_preset_ticker,
    provider_fallback_ticker,
    zero_return_series,
)
from app.services import testfol_api as api
from app.services.price_providers import get_price_provider

logger = logging.getLogger(__name__)

COMPONENT_DATA_CACHE_TTL = 86400
COMPONENT_DATA_CACHE_PREFIX = "component_prices"
NDX_MEGA_FRED_SERIES = {
    Tickers.NDXMEGASIM: "NASDAQNDXMEGAT",
    Tickers.NDXMEGA2SIM: "NASDAQNDXMEGA2T",
}


def _splice_official_total_return(
    local_series: pd.Series,
    official_series: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Keep local history through a common anchor, then preserve official returns."""
    local = pd.Series(local_series, copy=True).dropna().sort_index()
    official = pd.Series(official_series, copy=True).dropna().sort_index()
    for series in (local, official):
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
    local = local[~local.index.duplicated(keep="last")]
    official = official[~official.index.duplicated(keep="last")]

    if local.empty:
        return official.rename(name)
    if official.empty:
        return local.rename(name)

    common_dates = local.index.intersection(official.index)
    if common_dates.empty:
        warnings.warn(
            f"Official {name} history has no common anchor with the local simulation; "
            "using local history only."
        )
        return local.rename(name)

    anchor = pd.Timestamp(common_dates[0])
    official_anchor = float(official.loc[anchor])
    if official_anchor == 0:
        warnings.warn(f"Official {name} anchor is zero; using local history only.")
        return local.rename(name)

    scaled_official = official * (float(local.loc[anchor]) / official_anchor)
    combined = pd.concat([local.loc[:anchor], scaled_official.loc[scaled_official.index > anchor]])
    return combined[~combined.index.duplicated(keep="last")].sort_index().rename(name)


def _normalize_date_str(value) -> str:
    """Return a stable YYYY-MM-DD-style string for cache/provider keys."""
    return value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)


def clip_component_data_to_synced_end(
    prices: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """Drop trailing dates until every requested component has real data.

    This prevents mixed-date portfolios, e.g. live single names updating to a
    new trading day while SIM/proxy sleeves are still one close behind.
    """
    if prices.empty or not required_columns:
        return prices

    result = prices.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()

    columns = [col for col in dict.fromkeys(required_columns) if col in result.columns]
    if len(columns) < 2:
        return result

    last_valid_by_col: dict[str, pd.Timestamp] = {}
    for col in columns:
        series = result[col].dropna()
        if not series.empty:
            last_valid_by_col[col] = pd.Timestamp(series.index[-1])

    if len(last_valid_by_col) < 2:
        return result

    synced_end = min(last_valid_by_col.values())
    latest_end = max(last_valid_by_col.values())
    if synced_end >= latest_end:
        return result

    stale_cols = sorted(
        col for col, last_date in last_valid_by_col.items()
        if last_date == synced_end
    )
    fresh_cols = sorted(
        col for col, last_date in last_valid_by_col.items()
        if last_date == latest_end
    )
    logger.warning(
        "Clipping component data to %s because not all sleeves are synced "
        "(stale: %s; latest: %s at %s)",
        synced_end.date(),
        ", ".join(stale_cols),
        ", ".join(fresh_cols),
        latest_end.date(),
    )
    return result.loc[result.index <= synced_end]


def _component_cache_is_complete(
    prices: pd.DataFrame,
    required_columns: list[str],
    start_date: str,
    end_date: str,
) -> bool:
    if prices.empty:
        return False

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    window = prices.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    if window.empty:
        return False

    for col in dict.fromkeys(required_columns):
        if col not in window.columns or window[col].dropna().empty:
            logger.info("Ignoring component cache with missing/empty column: %s", col)
            return False

    return True


def _is_special_component_request(ticker: str) -> bool:
    base = str(ticker).split("?")[0].strip().upper()
    return is_testfol_preset_ticker(base) or base.endswith("SIM")


def _xnys_sessions(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    """Return actual NYSE sessions, including exceptional full-day closures."""
    try:
        import exchange_calendars as xcals

        # Pad construction bounds because either requested endpoint may itself be
        # a weekend or holiday, outside a calendar built to those exact dates.
        calendar = xcals.get_calendar(
            "XNYS",
            start=start_date - pd.Timedelta(days=7),
            end=end_date + pd.Timedelta(days=7),
        )
        sessions = calendar.sessions_in_range(start_date, end_date)
        if sessions.tz is not None:
            sessions = sessions.tz_localize(None)
        return pd.DatetimeIndex(sessions)
    except Exception as exc:
        # Keep a usable fallback for partially installed environments. Production
        # installs include exchange-calendars, which is required for exact parity.
        logger.warning("XNYS calendar unavailable; falling back to weekdays: %s", exc)
        return pd.bdate_range(start_date, end_date)


def _build_total_return_from_annual_rates(
    annual_rates: pd.Series | None,
    start_date: str,
    end_date: str,
    name: str,
) -> pd.Series | None:
    if annual_rates is None or annual_rates.empty:
        return None

    rates = annual_rates.copy()
    if not isinstance(rates.index, pd.DatetimeIndex):
        rates.index = pd.to_datetime(rates.index)
    if rates.index.tz is not None:
        rates.index = rates.index.tz_localize(None)
    rates = pd.to_numeric(rates, errors="coerce").sort_index().dropna()
    if rates.empty:
        return None

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < rates.index[0] or start_ts > end_ts:
        return None

    index = _xnys_sessions(max(start_ts, rates.index[0]), end_ts)
    if index.empty:
        return None

    aligned = rates.reindex(index, method="ffill").dropna()
    if aligned.empty:
        return None

    # Testfol's EFFRX/CASHX pseudo-tickers apply the quoted annual rate once
    # per trading observation using simple /252 (weekends are not multiplied).
    daily_returns = aligned / 100.0 / 252.0
    return (100.0 * (1.0 + daily_returns).cumprod()).rename(name)


def _local_preset_fallback(ticker: str, start_date: str, end_date: str) -> pd.Series | None:
    base = str(ticker).split("?")[0].strip().upper()
    if base == "ZEROX":
        return zero_return_series(start_date, end_date, name=ticker)
    if base == "EFFRX":
        return _build_total_return_from_annual_rates(
            get_fed_funds_rate(),
            start_date,
            end_date,
            ticker,
        )
    if base in {"TBILL", "CASHX"}:
        return _build_total_return_from_annual_rates(
            get_tbill_rate(),
            start_date,
            end_date,
            ticker,
        )
    if base == "INFLATION":
        series = get_inflation_index()
        if series is None or series.empty:
            return None
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        series = series.loc[start_ts:end_ts]
        return series.rename(ticker) if not series.empty else None
    return None


@lru_cache(maxsize=1)
def _load_ndx_price_cache() -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    cache_path = os.path.join(data_dir, "ndx_simulation", "data", "cache", "prices_cache.pkl")
    if not os.path.exists(cache_path):
        return pd.DataFrame()

    try:
        prices = pd.read_pickle(cache_path)
    except Exception as exc:
        logger.warning("Failed to load local NDX simulation price cache: %s", exc)
        return pd.DataFrame()

    if prices.empty:
        return pd.DataFrame()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()
    prices.columns = [str(col).strip().upper() for col in prices.columns]
    return prices


def _local_ndx_price_cache_fallback(ticker: str, start_date: str, end_date: str) -> pd.Series | None:
    base = str(ticker).split("?")[0].strip().upper()
    prices = _load_ndx_price_cache()
    if prices.empty or base not in prices.columns:
        return None

    series = prices[base].dropna().sort_index()
    if series.empty:
        return None

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    series = series.loc[start_ts:end_ts]
    return series.rename(base) if not series.empty else None


def fetch_component_data(tickers: list[str], start_date, end_date, *, sync_end: bool = True) -> pd.DataFrame:
    """
    Fetches historical data for each ticker individually via Testfol API (or local).
    Handles composite tickers like NDXMEGASIM by splicing local CSVs with live data.
    """
    if not tickers:
        return pd.DataFrame()

    unique_tickers = list(dict.fromkeys(tickers))
    unique_bases = list(dict.fromkeys(ticker.split("?")[0] for ticker in unique_tickers))
    sd_str = _normalize_date_str(start_date)
    ed_str = _normalize_date_str(end_date)

    cache_payload = json.dumps(
        {
            "tickers": unique_tickers,
            "start_date": sd_str,
            "end_date": ed_str,
            "sync_end": sync_end,
            "sync_policy": "requested-window-v6-testfol-sessions-modifiers",
        },
        sort_keys=True,
    )
    ck = cache_key(cache_payload)
    cached = cache_get(ck, prefix=COMPONENT_DATA_CACHE_PREFIX, ttl=COMPONENT_DATA_CACHE_TTL)
    if cached is not None and _component_cache_is_complete(cached, unique_bases, sd_str, ed_str):
        return cached

    combined_prices: dict[str, pd.Series] = {}
    provider = get_price_provider()
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    proxy_tickers: list[str] = []
    if Tickers.NDX30SIM in unique_bases:
        proxy_tickers.append("QTOP")
    proxy_prices = pd.DataFrame()
    if proxy_tickers:
        try:
            proxy_prices = provider.fetch_prices(sorted(set(proxy_tickers)), "2000-01-01", today_str)
        except Exception as e:
            warnings.warn(f"Failed to fetch proxy data for synthetic tickers: {e}")
            proxy_prices = pd.DataFrame()

    from app.core.shadow_backtest import parse_ticker

    mapped_provider_tickers: dict[str, str] = {}
    for base in unique_bases:
        is_special_request = _is_special_component_request(base)
        if not is_special_request:
            mapped_ticker, _ = parse_ticker(base)
            mapped_provider_tickers[base] = mapped_ticker

    batched_provider_prices = pd.DataFrame()
    if mapped_provider_tickers:
        try:
            batched_provider_prices = provider.fetch_prices(
                sorted(set(mapped_provider_tickers.values())),
                sd_str,
                ed_str,
            )
        except Exception as e:
            warnings.warn(f"Failed batched provider fetch: {e}")
            batched_provider_prices = pd.DataFrame()

    for base in unique_bases:
        try:
            # SPECIAL: NDX30SIM — load simulation CSV + splice with QTOP
            if base == Tickers.NDX30SIM:
                try:
                    csv_path = f"data/{base}.csv"
                    df_sim = pd.Series(dtype=float)
                    if os.path.exists(csv_path):
                        _df = pd.read_csv(csv_path)
                        if "Date" in _df.columns:
                            _df["Date"] = pd.to_datetime(_df["Date"])
                            _df = _df.set_index("Date")
                        if "Close" in _df.columns:
                            df_sim = _df["Close"].sort_index()
                    else:
                        warnings.warn(f"{base} requested but {csv_path} not found.")

                    qtop_series = pd.Series(dtype=float)
                    if not proxy_prices.empty and "QTOP" in proxy_prices.columns:
                        qtop_series = proxy_prices["QTOP"].dropna().sort_index()

                    if not qtop_series.empty and not df_sim.empty:
                        splice_date = qtop_series.index[0]
                        sim_part = df_sim[df_sim.index < splice_date]
                        if not sim_part.empty:
                            sim_end_val = sim_part.iloc[-1]
                            qtop_start_val = qtop_series.iloc[0]
                            scale_factor = qtop_start_val / sim_end_val if sim_end_val != 0 else 1.0
                            combined_prices[base] = pd.concat([sim_part * scale_factor, qtop_series])
                        else:
                            combined_prices[base] = qtop_series
                    elif not df_sim.empty:
                        combined_prices[base] = df_sim
                    elif not qtop_series.empty:
                        combined_prices[base] = qtop_series

                    continue
                except Exception as e:
                    raise RuntimeError(f"Failed to load/splice NDX30SIM: {e}")

            # SPECIAL: local pre-launch NDX Mega simulation + official total-return index.
            if base in [Tickers.NDXMEGASIM, Tickers.NDXMEGA2SIM]:
                try:
                    csv_path = f"data/{base}.csv"
                    df_sim = pd.DataFrame()
                    if os.path.exists(csv_path):
                        try:
                            df_sim = pd.read_csv(csv_path)
                        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
                            warnings.warn(f"Corruption detected in {base}.csv ({e}). Attempting auto-rebuild...")
                            rebuild_script = os.path.join("data", "ndx_simulation", "scripts", "rebuild_all.py")
                            if os.path.exists(rebuild_script):
                                try:
                                    subprocess.run([sys.executable, rebuild_script], check=True, timeout=1800)
                                    logger.info("Rebuild complete. Reloading data.")
                                    df_sim = pd.read_csv(csv_path)
                                except Exception as rebuild_err:
                                    raise RuntimeError(f"Rebuild failed: {rebuild_err}")
                            else:
                                raise FileNotFoundError(f"Cannot rebuild: Script not found at {rebuild_script}")

                        if "Date" in df_sim.columns:
                            df_sim["Date"] = pd.to_datetime(df_sim["Date"])
                            df_sim = df_sim.set_index("Date")
                        if "Close" not in df_sim.columns:
                            warnings.warn(f"{base}.csv missing 'Close' column")
                            df_sim = pd.DataFrame()
                        else:
                            df_sim = df_sim["Close"].sort_index()
                    else:
                        warnings.warn(f"{base} requested but {csv_path} not found.")

                    fred_id = NDX_MEGA_FRED_SERIES[base]
                    official_series = _get_fred_series(
                        fred_id,
                        f"{fred_id}.csv",
                        max_age_days=1,
                    )
                    if official_series is None or official_series.empty:
                        warnings.warn(
                            f"Official FRED series {fred_id} is unavailable; "
                            f"using local {base} history only."
                        )
                        combined_prices[base] = df_sim
                    else:
                        combined_prices[base] = _splice_official_total_return(
                            df_sim,
                            official_series,
                            name=base,
                        )

                    continue
                except Exception as e:
                    raise RuntimeError(f"Failed to load/splice local {base}: {e}")

            is_special_request = _is_special_component_request(base)
            mapped_ticker = mapped_provider_tickers.get(base)

            if base.upper() == "ZEROX":
                combined_prices[base] = zero_return_series(sd_str, ed_str, name=base)
                continue

            if mapped_ticker and not batched_provider_prices.empty:
                if mapped_ticker in batched_provider_prices.columns and not batched_provider_prices[mapped_ticker].dropna().empty:
                    combined_prices[base] = batched_provider_prices[mapped_ticker]
                    continue

            ndx_cache_fallback = _local_ndx_price_cache_fallback(base, sd_str, ed_str)
            if ndx_cache_fallback is not None and not ndx_cache_fallback.dropna().empty:
                combined_prices[base] = ndx_cache_fallback
                continue

            # Testfol API fallback (preferred for preset tickers and provider misses)
            try:
                series, _, _ = api.fetch_backtest(
                    start_date="1900-01-01",
                    end_date=today_str,
                    start_val=10000,
                    cashflow=0,
                    cashfreq="Monthly",
                    rolling=1,
                    invest_div=True,
                    rebalance="Yearly",
                    allocation={base: 100.0},
                )
                combined_prices[base] = series
            except Exception as api_err:
                local_fallback = _local_preset_fallback(base, sd_str, ed_str)
                if local_fallback is not None and not local_fallback.dropna().empty:
                    combined_prices[base] = local_fallback
                    continue

                # Last resort: try provider chain for Testfol/synthetic tickers.
                if is_special_request:
                    try:
                        mapped_ticker = provider_fallback_ticker(base)
                        prices = provider.fetch_prices([mapped_ticker], sd_str, ed_str)
                        if not prices.empty and mapped_ticker in prices.columns:
                            combined_prices[base] = prices[mapped_ticker]
                            continue
                    except Exception:
                        pass
                raise api_err

        except Exception as e:
            warnings.warn(f"Failed to fetch data for {base}: {e}")

    result = pd.DataFrame(combined_prices)
    if not result.empty:
        result = result.sort_index()
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        result = result.loc[pd.Timestamp(sd_str):pd.Timestamp(ed_str)]
        if sync_end:
            result = clip_component_data_to_synced_end(result, unique_bases)
    cache_set(ck, result, prefix=COMPONENT_DATA_CACHE_PREFIX)
    return result

import time

@lru_cache(maxsize=1)
def get_fed_funds_rate() -> pd.Series | None:
    """Return FRED's daily Effective Federal Funds Rate (``DFF``), in percent."""
    values = _get_fred_series("DFF", "DFF.csv", max_age_days=1)
    if values is None or values.empty:
        return None
    full_idx = pd.date_range(
        start=values.index.min(),
        end=max(pd.Timestamp.today().normalize(), values.index.max()),
        freq="D",
    )
    return values.reindex(full_idx).ffill().rename("DFF")


def _get_fred_series(series_id: str, filename: str, *, max_age_days: int = 30) -> pd.Series | None:
    """Fetch or read a cached FRED time series as numeric values."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    file_path = os.path.join(data_dir, filename)

    should_download = True
    if os.path.exists(file_path):
        mtime = os.path.getmtime(file_path)
        if (time.time() - mtime) < (max_age_days * 86400):
            should_download = False

    if should_download:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            import requests
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(r.content)
            logger.info("Downloaded fresh %s", filename)
        except Exception as e:
            logger.warning("Failed to download %s: %s. Using cached if available.", series_id, e)

    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=["observation_date"], index_col="observation_date")
        values = pd.to_numeric(df[series_id].replace(".", pd.NA), errors="coerce").dropna()
        return values.sort_index()
    except Exception as e:
        raise RuntimeError(f"Error reading {filename}: {e}")


def get_tbill_rate() -> pd.Series | None:
    """Return the 3-month Treasury Bill annualized rate from FRED."""
    return _get_fred_series("DTB3", "DTB3.csv")


def get_fred_yield_rate(series_id: str) -> pd.Series | None:
    """Return a FRED DGS yield series accepted by Testfol's ``FR`` modifier."""
    normalized = str(series_id).strip().upper()
    if not normalized.startswith("DGS") or not normalized[3:].isalnum():
        raise ValueError(f"Unsupported FRED financing series: {series_id}")
    return _get_fred_series(normalized, f"{normalized}.csv", max_age_days=1)


def get_inflation_index() -> pd.Series | None:
    """Return a daily-forward-filled CPI index for the INFLATION pseudo-ticker."""
    cpi = _get_fred_series("CPIAUCSL", "CPIAUCSL.csv")
    if cpi is None or cpi.empty:
        return None

    full_idx = pd.bdate_range(start=cpi.index.min(), end=pd.Timestamp.today())
    return cpi.reindex(full_idx).ffill().rename("INFLATION")
