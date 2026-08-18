# 20 Downside Ideas x 10 Variants

Every simulation uses prior-day signals, next-session execution, and a portfolio catastrophe brake: force cash at 20% drawdown or a 15% five-session loss, then require ten sessions and recovery to within 3% of the high before re-entry.

Tested **2000** configurations across the 10 expert portfolios.

## Ideas
1. **EMA trend exit:** Exit an asset below its EMA and re-enter above it.
2. **EMA cross exit:** Exit on a fast/slow EMA bearish cross; re-enter on a bullish cross.
3. **EMA ATR buffer:** Exit only when price falls below EMA minus an ATR buffer; re-enter above EMA.
4. **Donchian exit:** Exit below the prior rolling low and re-enter above the rolling high.
5. **Trailing peak stop:** Exit after a fixed drawdown from each asset's rolling peak; re-enter after recovery.
6. **Asset drawdown brake:** Exit when an asset's drawdown from its all-time observed high exceeds the threshold.
7. **Consecutive down days:** Exit after consecutive negative days; re-enter after consecutive positive days.
8. **RSI regime:** Exit in weak RSI regimes and re-enter when RSI recovers above a fixed level.
9. **MACD regime:** Exit while MACD is below its signal line; re-enter on a bullish cross.
10. **Momentum lookback:** Exit negative trailing momentum and re-enter when trailing momentum turns positive.
11. **Volatility shock:** Exit an asset when volatility spikes and its return is negative; re-enter after volatility normalizes.
12. **Mean-reversion guard:** Exit deep weakness below an SMA; re-enter only after price recovers above the SMA.
13. **EMA slope:** Exit when the long EMA slopes down; re-enter when its slope turns positive.
14. **Two-factor vote:** Exit when at least two of EMA, momentum, and channel signals are bearish.
15. **Breadth gate:** Move the portfolio to cash when too many allocated assets are below their EMA.
16. **Relative strength:** Exit the weakest momentum assets and concentrate proceeds in the strongest eligible asset.
17. **Volatility weighting:** Reduce or exit high-volatility assets and redistribute to lower-volatility holdings.
18. **Scale-out trend:** Sell half on a soft trend break and fully exit on a harder break; re-enter above trend.
19. **Cooldown re-entry:** Use an EMA exit, then require a fixed number of sessions before re-entry.
20. **Portfolio catastrophe brake:** Force all positions to cash after a portfolio drawdown or rolling-loss event, then wait for recovery.

## Best Balanced Results

| Idea | Variant | Portfolio | CAGR Change | DD Improvement | Protected CAGR | Protected DD |
|---|---:|---:|---:|---:|---:|---:|
| Volatility shock | 8 | 3 | +4.84% | +3.45% | 22.51% | -11.94% |
| Volatility shock | 8 | 10 | +5.10% | +2.67% | 22.51% | -11.94% |
| Mean-reversion guard | 3 | 8 | +3.96% | +3.23% | 23.45% | -15.97% |
| EMA cross exit | 2 | 5 | +4.51% | +1.65% | 24.36% | -17.96% |
| Trailing peak stop | 2 | 5 | +1.03% | +8.43% | 20.88% | -11.18% |
| EMA cross exit | 1 | 8 | +1.56% | +7.19% | 21.05% | -12.01% |
| EMA ATR buffer | 5 | 8 | +4.14% | +1.84% | 23.63% | -17.37% |
| EMA cross exit | 2 | 1 | +4.39% | +1.31% | 24.36% | -17.96% |
| Trailing peak stop | 2 | 1 | +0.91% | +8.09% | 20.88% | -11.18% |
| Volatility shock | 8 | 9 | +4.78% | +0.22% | 22.51% | -11.94% |
| RSI regime | 6 | 8 | +0.83% | +7.78% | 20.32% | -11.43% |
| EMA slope | 4 | 8 | +4.34% | +0.69% | 23.83% | -18.52% |
| Two-factor vote | 2 | 8 | +2.67% | +3.28% | 22.17% | -15.93% |
| EMA cross exit | 3 | 5 | +2.97% | +2.63% | 22.82% | -16.98% |
| EMA cross exit | 3 | 8 | +1.89% | +4.70% | 21.39% | -14.51% |
| EMA cross exit | 2 | 8 | +0.76% | +6.72% | 20.26% | -12.49% |
| Trailing peak stop | 2 | 8 | +0.27% | +7.67% | 19.77% | -11.54% |
| Trailing peak stop | 2 | 3 | +1.47% | +5.26% | 19.15% | -10.13% |
| EMA cross exit | 3 | 1 | +2.85% | +2.29% | 22.82% | -16.98% |
| Trailing peak stop | 2 | 10 | +1.73% | +4.48% | 19.15% | -10.13% |
| Mean-reversion guard | 3 | 5 | +1.86% | +3.25% | 21.71% | -16.36% |
| Donchian exit | 1 | 8 | +0.64% | +5.34% | 20.13% | -13.87% |
| Mean-reversion guard | 3 | 1 | +1.74% | +2.90% | 21.71% | -16.36% |
| EMA trend exit | 4 | 8 | +2.81% | +0.69% | 22.31% | -18.52% |
| Cooldown re-entry | 7 | 8 | +2.81% | +0.69% | 22.31% | -18.52% |
| Asset drawdown brake | 1 | 5 | +1.09% | +3.92% | 20.94% | -15.69% |
| EMA ATR buffer | 7 | 8 | +2.11% | +1.84% | 21.61% | -17.37% |
| Asset drawdown brake | 2 | 5 | +2.06% | +1.75% | 21.91% | -17.86% |
| Asset drawdown brake | 1 | 1 | +0.97% | +3.58% | 20.94% | -15.69% |
| Two-factor vote | 5 | 8 | +2.68% | +0.00% | 22.18% | -19.21% |
