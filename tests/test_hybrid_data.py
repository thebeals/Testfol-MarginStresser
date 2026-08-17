import pandas as pd

from screener.hybrid_data import extend_validated_letfs


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
    assert result["UPRO"].loc[index[1]].notna()
    assert result["UPRO"].pct_change().loc[index[3]] == 0.01
