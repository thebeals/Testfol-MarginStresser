from pathlib import Path
import importlib.util


MODULE_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "ndx_simulation"
    / "scripts"
    / "rebuild_all.py"
)
SPEC = importlib.util.spec_from_file_location("ndx_rebuild_all", MODULE_PATH)
rebuild_all = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rebuild_all)


def test_ndxmega_rebuild_generates_qqup_price_return_then_total_return(monkeypatch):
    calls = []

    def capture(script_path, desc, env=None, script_args=None):
        calls.append((script_path, desc, env, script_args))

    monkeypatch.setattr(rebuild_all, "run_script", capture)
    base_env = {"NDX_DATA_SOURCE": "stooq"}

    rebuild_all.run_ndxmega_backtests("backtest_ndx_mega.py", base_env)

    assert base_env == {"NDX_DATA_SOURCE": "stooq"}
    assert [call[1] for call in calls] == [
        "Backtest NDX Mega 1.0 Price Return (QQUPSIM)",
        "Backtest NDX Mega 1.0 Total Return",
    ]
    assert calls[0][2] == {
        "NDX_DATA_SOURCE": "stooq",
        "NDX_PRICE_RETURN": "1",
    }
    assert calls[1][2] == {
        "NDX_DATA_SOURCE": "stooq",
        "NDX_PRICE_RETURN": "0",
    }


def test_deep_clean_includes_qqup_price_return_output():
    assert "NDXMEGAPRICESIM.csv" in rebuild_all.TOP_LEVEL_SIMULATION_OUTPUTS
