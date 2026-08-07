from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import pandas as pd


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_ndx_preset_data_versions.py"
SPEC = importlib.util.spec_from_file_location("compare_ndx_presets", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def test_each_preset_receives_an_unmodified_price_frame(monkeypatch, tmp_path):
    observed = []

    def mutating_engine(**kwargs):
        frame = kwargs["prefetched_component_prices"]
        observed.append(float(frame.iloc[0, 0]))
        frame.iloc[:, :] = 0.0
        return {
            "port_series": pd.Series(
                [10000.0, 10001.0],
                index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
            )
        }

    monkeypatch.setattr(comparison, "run_single_backtest", mutating_engine)
    monkeypatch.setattr(comparison, "_activate_version", lambda context: None)
    context = comparison.VersionContext("test", tmp_path, tmp_path / "w", tmp_path / "c")
    preset = {
        "category": "test",
        "allocation": {"SPY": 100.0},
        "maint": {"SPY": 25.0},
        "pm_maint": {"SPY": 0.0},
        "rebalance": {},
    }
    presets = [dict(preset, name="one"), dict(preset, name="two")]
    plans = {
        "one": SimpleNamespace(dynamic_schedule={}, universe_tickers={"SPY"}),
        "two": SimpleNamespace(dynamic_schedule={}, universe_tickers={"SPY"}),
    }
    prices = pd.DataFrame(
        {"SPY": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"])
    )

    _, errors = comparison._run_presets(
        context, presets, plans, prices, "2020-01-02"
    )

    assert not errors
    assert observed == [1.0, 1.0]
    assert prices.iloc[0, 0] == 1.0
