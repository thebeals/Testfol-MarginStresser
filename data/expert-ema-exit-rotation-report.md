# EMA Exit and Rotation Study

Daily signals use prior-day information and execute on the next session. The Pine-style strategy uses `SMA(close, 2)` crossing EMA +/- ATR. The crossover strategy uses 9/20 EMA crosses with 100/200 EMA filters.

Tested **390** configurations across the expert top 10.

| Rank | Strategy | Parameters | Rotation | CAGR Change | DD Improvement |
|---:|---|---|---|---:|---:|
| 5 | atr_band | 100 / ATR 0.5 | last_buy | +3.75% | +0.51% |
| 1 | atr_band | 100 / ATR 0.5 | last_buy | +3.63% | +0.16% |
| 5 | atr_band | 100 / ATR 1.0 | last_buy | +0.39% | +1.68% |
| 1 | atr_band | 100 / ATR 1.0 | last_buy | +0.27% | +1.33% |

No fees, taxes, slippage, or market impact are modeled. Rotation is deterministic: `winner` sends proceeds to the held asset with the strongest trailing 63-session return; `last_buy` sends proceeds to the most recent asset with a buy signal, otherwise cash.
