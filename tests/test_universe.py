"""Tests for the ticker universe and risk-factor mapping."""

from screener.universe import (
    FACTOR_MAP,
    Factor,
    WRAPPER_FACTORS,
    all_tickers,
    factors_for,
    factor_tickers,
    wrapper_tickers,
)


def test_total_universe_size_is_62():
    assert len(all_tickers()) == 62


def test_factor_and_wrapper_counts():
    assert len(factor_tickers()) == 50
    assert len(wrapper_tickers()) == 12
    assert set(factor_tickers()) & set(wrapper_tickers()) == set()


def test_all_tickers_union_is_complete():
    assert set(all_tickers()) == set(factor_tickers()) | set(wrapper_tickers())


def test_eight_factors_present():
    assert set(Factor) == {
        Factor.EQUITY,
        Factor.NOMINAL_DURATION,
        Factor.REAL_DURATION,
        Factor.CASH,
        Factor.GOLD,
        Factor.COMMODITIES,
        Factor.MANAGED_FUTURES,
        Factor.REAL_ESTATE,
    }


def test_every_factor_has_at_least_one_ticker():
    covered = set(FACTOR_MAP.values())
    assert covered == set(Factor)


def test_leveraged_equity_maps_to_equity():
    for t in ("UPRO", "SSO", "TQQQ"):
        assert FACTOR_MAP[t] == Factor.EQUITY


def test_sector_etfs_map_to_equity():
    for t in ("XLE", "XLI", "XLP", "XLU", "XLV", "VPU"):
        assert FACTOR_MAP[t] == Factor.EQUITY


def test_leveraged_duration_maps_to_nominal_duration():
    for t in ("TMF", "TYD", "UBT"):
        assert FACTOR_MAP[t] == Factor.NOMINAL_DURATION


def test_spot_checks_across_factors():
    assert FACTOR_MAP["TLT"] == Factor.NOMINAL_DURATION
    assert FACTOR_MAP["TIP"] == Factor.REAL_DURATION
    assert FACTOR_MAP["VTIP"] == Factor.REAL_DURATION
    assert FACTOR_MAP["BIL"] == Factor.CASH
    assert FACTOR_MAP["GLD"] == Factor.GOLD
    assert FACTOR_MAP["DBC"] == Factor.COMMODITIES
    assert FACTOR_MAP["KMLM"] == Factor.MANAGED_FUTURES
    assert FACTOR_MAP["VNQ"] == Factor.REAL_ESTATE


def test_factors_for_single_factor_ticker():
    assert factors_for("SPY") == (Factor.EQUITY,)
    assert factors_for("VNQ") == (Factor.REAL_ESTATE,)


def test_factors_for_wrapper_ticker():
    assert factors_for("NTSX") == (Factor.EQUITY, Factor.NOMINAL_DURATION)
    assert factors_for("RSST") == (Factor.EQUITY, Factor.MANAGED_FUTURES)


def test_factors_for_unknown_ticker_raises():
    try:
        factors_for("NOT_A_TICKER")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown ticker")


def test_rssx_is_present_and_flagged():
    # RSSX carries bitcoin exposure; kept in the universe, unresolved (Jonathan's call).
    assert "RSSX" in WRAPPER_FACTORS
    assert factors_for("RSSX") == (Factor.EQUITY,)


def test_no_ticker_maps_to_two_entries():
    assert not (set(FACTOR_MAP.keys()) & set(WRAPPER_FACTORS.keys()))


def test_wrappers_stack_at_least_one_known_factor():
    known = set(Factor)
    for wrapper, factors in WRAPPER_FACTORS.items():
        assert wrapper == wrapper.upper()
        assert factors, f"wrapper {wrapper} maps to no factor"
        assert all(f in known for f in factors)
