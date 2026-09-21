import pandas as pd
import pytest

from screener.hybrid_data import extend_hybrid_prices, extend_validated_letfs, load_testfol_simulated_prices


def test_letf_backfill_uses_simulation_before_observed_history() -> None:
    index = pd.date_range("2006-01-03", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0],
            "UPRO": [None, None, 100.0, 101.0, 102.0],
        },
        index=index,
    )

    result, simulated = extend_validated_letfs(prices, pd.Series(4.0, index=index))

    assert "UPRO" in simulated
    assert pd.notna(result["UPRO"].loc[index[1]])
    assert result["UPRO"].pct_change().loc[index[3]] == pytest.approx(0.01)


def test_cached_testfol_returns_are_loaded_as_prices(tmp_path) -> None:
    cache = tmp_path / "testfol.json"
    cache.write_text(
        '{"series": {"DBMF": {"daily_returns_pct": [["2006-01-03", 1.0, 101000.0], ["2006-01-04", -0.5, 100495.0]]}}}',
        encoding="utf-8",
    )

    prices = load_testfol_simulated_prices(cache)

    assert prices["DBMF"].iloc[0] == pytest.approx(101.0)
    assert prices["DBMF"].iloc[1] == pytest.approx(100.495)


def test_hybrid_loader_reports_effective_source(tmp_path) -> None:
    cache = tmp_path / "testfol.json"
    cache.write_text(
        '{"series": {"DBMF": {"daily_returns_pct": [["2006-01-03", 1.0, 101000.0]]}}}',
        encoding="utf-8",
    )
    prices = pd.DataFrame({"DBMF": [100.0]}, index=pd.DatetimeIndex(["2006-01-03"]))

    result, provenance = extend_hybrid_prices(prices, None, testfol_cache_path=cache)

    assert result["DBMF"].iloc[0] == pytest.approx(101.0)
    assert provenance["source_by_ticker"]["DBMF"] == "testfol_simulated"
