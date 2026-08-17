import numpy as np
import pandas as pd

from screener.prescreen import _factor_signature, prescreen_subsets


def test_prescreen_keeps_factor_signature_coverage() -> None:
    returns = pd.DataFrame(
        np.array(
            [
                [0.01, 0.001, 0.002, 0.004],
                [0.02, 0.001, -0.001, 0.003],
                [-0.01, 0.001, 0.003, -0.002],
                [0.015, 0.001, 0.001, 0.005],
            ]
        ),
        columns=["SPY", "SGOV", "GLD", "TLT"],
    )

    subsets = prescreen_subsets(
        returns,
        min_tickers=2,
        max_tickers=2,
        max_candidates=4,
        min_factors=2,
    )

    signatures = {_factor_signature(subset.tickers) for subset in subsets}
    assert len(signatures) == len(subsets)
