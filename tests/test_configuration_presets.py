import json

from app.ui.configuration import (
    DEFAULT_GLOBAL_CASHFLOW,
    PRESET_CATEGORY_ORDER,
    _preset_category,
    _preset_display_label,
    _sort_preset_names,
)


def test_global_cashflow_defaults_do_not_margin_fund_dca():
    assert DEFAULT_GLOBAL_CASHFLOW["fund_dca_margin"] is False


def test_sort_preset_names_keeps_research_presets_at_bottom():
    presets = [
        {"name": "Research - Risk Parity Split"},
        {"name": "SPY Only"},
        {"name": "NDX Top 2 Research - Safe"},
        {"name": "NDXMEGASPLIT"},
        {"name": "A Test"},
        {"name": "NDX Mega 1.0 (Sim)"},
    ]

    assert _sort_preset_names(presets) == [
        "NDX Mega 1.0 (Sim)",
        "NDXMEGASPLIT",
        "A Test",
        "SPY Only",
        "NDX Top 2 Research - Safe",
        "Research - Risk Parity Split",
    ]


def test_all_presets_are_categorized_and_sorted_by_category():
    with open("data/presets.json", encoding="utf-8") as handle:
        presets = json.load(handle)

    assert len(presets) == 28
    assert len({preset["name"] for preset in presets}) == len(presets)
    assert all(preset.get("category") in PRESET_CATEGORY_ORDER for preset in presets)

    by_name = {preset["name"]: preset for preset in presets}
    sorted_names = _sort_preset_names(presets)
    category_rank = {category: idx for idx, category in enumerate(PRESET_CATEGORY_ORDER)}
    ranks = [category_rank[_preset_category(by_name[name])] for name in sorted_names]
    assert ranks == sorted(ranks)
    assert sorted_names[:5] == [
        "NDXMEGASPLIT (w/ ERs)",
        "NDXMEGASPLIT (w/ ERs) - NDXMEGASIM 2x",
        "NDXMEGASPLIT",
        "QQUP - 2x NDXMEGA Proxy (No Rebalance)",
        "NDX Mega 1.0 (Sim)",
    ]


def test_preset_display_label_adds_category_without_renaming():
    presets = [{"name": "SPY Only", "category": "Single Asset & Proxies"}]
    assert _preset_display_label("SPY Only", presets) == (
        "[Single Asset & Proxies] SPY Only"
    )
    assert _preset_display_label("Missing", presets) == "Missing"


def test_qqup_presets_use_realized_tracker_simulation():
    with open("data/presets.json", encoding="utf-8") as handle:
        presets = {preset["name"]: preset for preset in json.load(handle)}

    standalone = presets["QQUP - 2x NDXMEGA Proxy (No Rebalance)"]
    er_aware = presets["NDXMEGASPLIT (w/ ERs)"]
    total_return_er_aware = presets["NDXMEGASPLIT (w/ ERs) - NDXMEGASIM 2x"]
    assert standalone["allocation"][0]["Ticker"] == "QQUPSIM"
    assert er_aware["allocation"][0]["Ticker"] == "QQUPSIM"
    assert total_return_er_aware["allocation"][0]["Ticker"] == (
        "NDXMEGASIM?L=2&E=0.95"
    )
    assert total_return_er_aware["allocation"][1:] == er_aware["allocation"][1:]
    assert total_return_er_aware["rebalance"] == er_aware["rebalance"]

    # The no-expense/theoretical preset intentionally remains index based.
    theoretical = presets["NDXMEGASPLIT"]
    assert theoretical["allocation"][0]["Ticker"] == "NDXMEGASIM?L=2"
