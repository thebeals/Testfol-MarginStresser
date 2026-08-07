from pathlib import Path
import importlib.util

import pandas as pd


MODULE_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "ndx_simulation"
    / "src"
    / "simulation_pricing.py"
)
SPEC = importlib.util.spec_from_file_location("ndx_simulation_pricing", MODULE_PATH)
simulation_pricing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simulation_pricing)


def test_pricing_weights_use_aliases_and_preserve_missing_company_weight():
    rows = pd.DataFrame(
        {
            "Ticker": ["FB", "META", "OLD"],
            "PriceTicker": ["META", "META", "QQQ"],
        }
    )
    security_weights = pd.Series({"FB": 0.30, "META": 0.20, "OLD": 0.50})

    result = simulation_pricing.collapse_to_pricing_weights(
        security_weights,
        simulation_pricing.price_ticker_map(rows),
        available_tickers=["META", "QQQ"],
    )

    assert result.to_dict() == {"META": 0.5, "QQQ": 0.5}
