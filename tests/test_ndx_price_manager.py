from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd


NDX_SRC = Path(__file__).parents[1] / "data" / "ndx_simulation" / "src"
sys.path.insert(0, str(NDX_SRC))

import price_manager  # noqa: E402


def _write_cache(path, age_days):
    cache_date = pd.Timestamp(datetime.now().date() - timedelta(days=age_days))
    pd.DataFrame(
        {"AAPL": [50.0, 100.0]},
        index=[pd.Timestamp("2000-01-03"), cache_date],
    ).to_pickle(path)


def test_get_price_data_refreshes_stale_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "prices.pkl"
    _write_cache(cache_file, age_days=10)
    refreshed = pd.DataFrame({"AAPL": [101.0]}, index=[pd.Timestamp.today()])
    calls = []

    monkeypatch.setattr(
        price_manager.config,
        "PRICE_CACHE_FILE",
        str(cache_file),
        raising=False,
    )
    monkeypatch.setenv("NDX_PRICE_CACHE_MAX_AGE_DAYS", "3")

    def fake_download(*args):
        calls.append(args)
        return refreshed

    monkeypatch.setattr(price_manager, "download_and_cache", fake_download)

    result = price_manager.get_price_data(["AAPL"])

    assert result is refreshed
    assert len(calls) == 1


def test_get_price_data_reuses_fresh_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "prices.pkl"
    _write_cache(cache_file, age_days=1)

    monkeypatch.setattr(
        price_manager.config,
        "PRICE_CACHE_FILE",
        str(cache_file),
        raising=False,
    )

    def unexpected_download(*args):
        raise AssertionError("fresh cache should not be downloaded again")

    monkeypatch.setattr(price_manager, "download_and_cache", unexpected_download)

    result = price_manager.get_price_data(["AAPL"], max_cache_age_days=3)

    assert result.iloc[-1]["AAPL"] == 100.0
