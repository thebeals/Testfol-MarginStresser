from app.ui.configuration import DEFAULT_GLOBAL_CASHFLOW, _sort_preset_names


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
