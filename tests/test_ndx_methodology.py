from pathlib import Path
import sys

import pandas as pd
import pytest


NDX_SRC = Path(__file__).parents[1] / "data" / "ndx_simulation" / "src"
sys.path.insert(0, str(NDX_SRC))

import ndx_methodology  # noqa: E402
from ndx_methodology import (  # noqa: E402
    apply_2026_ndx_capping,
    build_validation_periods,
    interpolate_new_entry_values,
    modified_market_cap_factor,
    uses_2026_methodology,
)


def _cap(weights, *, annual=False):
    group = pd.DataFrame(
        {
            "Ticker": list(weights),
            "Weight": list(weights.values()),
        }
    )
    return apply_2026_ndx_capping(group, is_annual=annual).set_index(
        "Ticker"
    )["CappedWeight"]


def test_methodology_cutover_and_modified_market_cap_factor():
    assert not uses_2026_methodology("2026-04-30")
    assert uses_2026_methodology("2026-05-01")
    assert modified_market_cap_factor(100, 10, "2026-04-30") == 1.0
    assert modified_market_cap_factor(100, 10, "2026-05-01") == pytest.approx(
        0.3
    )
    assert modified_market_cap_factor(100, 60, "2026-05-01") == 1.0
    assert modified_market_cap_factor(None, None, "2026-05-01") == 1.0


def test_new_entry_values_are_linearly_interpolated_by_market_cap_rank():
    values = pd.Series({"A": 100.0, "N1": 90.0, "N2": 80.0, "B": 50.0})

    adjusted, interpolated = interpolate_new_entry_values(values, {"N1", "N2"})

    assert interpolated == {"N1", "N2"}
    assert adjusted["N1"] == pytest.approx(100.0 - (50.0 / 3.0))
    assert adjusted["N2"] == pytest.approx(100.0 - (100.0 / 3.0))
    assert adjusted[["A", "B"]].to_dict() == {"A": 100.0, "B": 50.0}


def test_interpolation_leaves_unbounded_edge_entry_unchanged():
    values = pd.Series({"NEW": 120.0, "A": 100.0, "B": 50.0})

    adjusted, interpolated = interpolate_new_entry_values(values, {"NEW"})

    assert adjusted.equals(values)
    assert interpolated == set()


def test_company_cap_combines_share_classes_and_preserves_their_ratio(
    monkeypatch,
):
    monkeypatch.setattr(
        ndx_methodology.config,
        "DUAL_CLASS_GROUPS",
        {"GOOG": "GOOGL", "GOOGL": "GOOGL"},
        raising=False,
    )
    weights = {"GOOG": 0.16, "GOOGL": 0.13, "BIG": 0.12}
    weights.update({f"T{i:03}": 0.59 / 97 for i in range(97)})

    capped = _cap(weights)

    alphabet_weight = capped["GOOG"] + capped["GOOGL"]
    assert capped.sum() == pytest.approx(1.0)
    assert alphabet_weight == pytest.approx(0.20)
    assert capped["GOOG"] / capped["GOOGL"] == pytest.approx(0.16 / 0.13)


def test_large_company_cohort_is_reduced_to_40_percent_without_rank_flip():
    weights = {f"BIG{i}": 0.10 for i in range(6)}
    weights.update({f"SMALL{i:03}": 0.40 / 94 for i in range(94)})

    capped = _cap(weights)
    big = capped.loc[[f"BIG{i}" for i in range(6)]]
    small = capped.loc[[f"SMALL{i:03}" for i in range(94)]]

    assert capped.sum() == pytest.approx(1.0)
    assert big.sum() == pytest.approx(0.40)
    assert big.min() >= small.max()
    assert capped[capped > 0.045].sum() <= 0.48 + 1e-12


def test_annual_security_stage_targets_top_five_and_outside_cap():
    weights = {
        f"T{i:03}": weight
        for i, weight in enumerate(
            [0.18, 0.12, 0.09, 0.07, 0.06] + [0.48 / 95] * 95
        )
    }

    capped = _cap(weights, annual=True).sort_values(ascending=False)
    outside_cap = min(0.044, capped.iloc[4])

    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.14 + 1e-12
    assert capped.iloc[:5].sum() == pytest.approx(0.385)
    assert capped.iloc[5:].max() <= outside_cap + 1e-12


def test_validation_carries_latest_weights_to_end_of_price_data():
    periods = build_validation_periods(
        ["2026-03-23", "2026-06-22"],
        "2026-08-06",
    )

    assert periods == [
        (pd.Timestamp("2026-03-23"), pd.Timestamp("2026-06-22")),
        (pd.Timestamp("2026-06-22"), pd.Timestamp("2026-08-06")),
    ]
