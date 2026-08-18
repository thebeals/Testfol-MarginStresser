# Expert Top-10 Investment Memo

This memo reviews the expert shortlist through an investor-risk lens and then tests two explicit cash overlays. Results are historical simulations, not guarantees or personalized advice.

## Shortlist

The shortlist candidates all beat the locked SPY test CAGR and pass the method-level 20% drawdown gate. The selection balances return, drawdown, diversification, leverage, crisis behavior, rolling consistency, and Testfol validation.

## Protection Rules Tested

- **EMA100 monthly cash:** At the monthly check, move the whole portfolio to SHV when the portfolio wealth index is below its EMA100; re-enter when it is above EMA100.
- **10% component-crash cash:** Move the whole portfolio to SHV on the next session after any allocated component loses at least 10% in one day; re-enter at a monthly check when portfolio wealth is above EMA100.
- Cash uses SHV total-return data where available.
- No margin, tax, slippage, or leverage was added by these overlays.

### #1: DBC 6.7% / DBMF 15.8% / GLD 31.0% / TQQQ 16.6% / VNQ 29.9%

**Method:** `AbsoluteBand:Yearly:0.2`  
**Asset classes:** US equity, commodities, gold, managed futures, real estate  
**Leveraged ETF weight:** 16.6%  
**Exact Testfol method:** Yes

**Why it is good**

- Test CAGR of 20.0%, +4.2% versus SPY.
- Observed max drawdown of -19.3% in the locked test window.
- Spans 5 asset classes: US equity, commodities, gold, managed futures, real estate.
- The exact rebalance method has a zero-error Testfol verification record.

**Downside risks**

- Explicit leveraged ETF exposure is 16.6%; losses can compound sharply.
- Worst observed rolling three-month return was -13.6%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 20.0% | -19.3% | 1.18 | 231 | 0.0% |
| EMA100 monthly cash | 13.0% | -17.9% | 0.89 | 97 | 23.1% |
| 10% component crash cash | 14.5% | -15.1% | 1.05 | 94 | 27.8% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #2: DBC 34.2% / GLD 44.6% / TIP 4.9% / TQQQ 10.9% / VNQ 2.1% / VPU 3.2%

**Method:** `AbsoluteBand:Yearly:0.2`  
**Asset classes:** US equity, bonds, commodities, gold, real estate  
**Leveraged ETF weight:** 10.9%  
**Exact Testfol method:** Yes

**Why it is good**

- Test CAGR of 20.7%, +5.0% versus SPY.
- Observed max drawdown of -19.6% in the locked test window.
- Spans 5 asset classes: US equity, bonds, commodities, gold, real estate.
- The exact rebalance method has a zero-error Testfol verification record.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- Worst observed rolling three-month return was -13.6%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 20.7% | -19.6% | 1.29 | 262 | 0.0% |
| EMA100 monthly cash | 17.3% | -12.8% | 1.19 | Not recovered | 14.5% |
| 10% component crash cash | 17.3% | -11.4% | 1.36 | 43 | 26.0% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #3: DBC 25.4% / DBMF 11.6% / GLD 37.4% / SHV 4.3% / UPRO 7.1% / VTV 14.2%

**Method:** `AbsoluteBand:Yearly:0.05`  
**Asset classes:** US equity, cash, commodities, gold, managed futures  
**Leveraged ETF weight:** 7.1%  
**Exact Testfol method:** Yes

**Why it is good**

- Test CAGR of 17.7%, +1.9% versus SPY.
- Observed max drawdown of -15.4% in the locked test window.
- Spans 5 asset classes: US equity, cash, commodities, gold, managed futures.
- The exact rebalance method has a zero-error Testfol verification record.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- It did not beat SPY in most rolling three-year windows.
- Worst observed rolling three-month return was -11.4%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 17.7% | -15.4% | 1.38 | 521 | 0.0% |
| EMA100 monthly cash | 15.2% | -11.1% | 1.29 | 300 | 15.3% |
| 10% component crash cash | 17.3% | -11.1% | 1.53 | Not recovered | 12.5% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #4: DBC 37.9% / DBMF 33.5% / TQQQ 11.7% / VNQ 10.8% / XLP 6.1%

**Method:** `Calendar:Yearly`  
**Asset classes:** US equity, commodities, managed futures, real estate  
**Leveraged ETF weight:** 11.7%  
**Exact Testfol method:** Yes

**Why it is good**

- Test CAGR of 18.5%, +2.8% versus SPY.
- Observed max drawdown of -15.9% in the locked test window.
- Spans 4 asset classes: US equity, commodities, managed futures, real estate.
- The exact rebalance method has a zero-error Testfol verification record.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- It did not beat SPY in most rolling three-year windows.
- Worst observed rolling three-month return was -12.0%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 18.5% | -15.9% | 1.26 | 155 | 0.0% |
| EMA100 monthly cash | 17.1% | -12.1% | 1.32 | 427 | 22.3% |
| 10% component crash cash | 16.2% | -12.6% | 1.36 | 536 | 29.8% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #5: DBC 6.7% / DBMF 15.8% / GLD 31.0% / TQQQ 16.6% / VNQ 29.9%

**Method:** `Calendar:Yearly`  
**Asset classes:** US equity, commodities, gold, managed futures, real estate  
**Leveraged ETF weight:** 16.6%  
**Exact Testfol method:** No

**Why it is good**

- Test CAGR of 19.8%, +4.1% versus SPY.
- Observed max drawdown of -19.6% in the locked test window.
- Spans 5 asset classes: US equity, commodities, gold, managed futures, real estate.

**Downside risks**

- Explicit leveraged ETF exposure is 16.6%; losses can compound sharply.
- The exact dynamic/local method is not directly Testfol-validated.
- Worst observed rolling three-month return was -13.7%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 19.8% | -19.6% | 1.18 | 241 | 0.0% |
| EMA100 monthly cash | 13.0% | -18.3% | 0.89 | 124 | 23.1% |
| 10% component crash cash | 14.5% | -15.6% | 1.06 | 94 | 27.8% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #6: DBC 27.7% / DBMF 39.1% / TLT 2.8% / TQQQ 11.6% / TYD 2.8% / VNQ 16.0%

**Method:** `Calendar:Yearly`  
**Asset classes:** US equity, bonds, commodities, managed futures, real estate  
**Leveraged ETF weight:** 14.4%  
**Exact Testfol method:** Yes

**Why it is good**

- Test CAGR of 16.7%, +1.0% versus SPY.
- Observed max drawdown of -15.7% in the locked test window.
- Spans 5 asset classes: US equity, bonds, commodities, managed futures, real estate.
- The exact rebalance method has a zero-error Testfol verification record.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- It did not beat SPY in most rolling three-year windows.
- Worst observed rolling three-month return was -11.7%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 16.7% | -15.7% | 1.21 | 154 | 0.0% |
| EMA100 monthly cash | 15.9% | -11.6% | 1.30 | 427 | 23.8% |
| 10% component crash cash | 14.1% | -12.3% | 1.24 | 540 | 29.8% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #7: DBC 34.2% / GLD 44.6% / TIP 4.9% / TQQQ 10.9% / VNQ 2.1% / VPU 3.2%

**Method:** `Calendar:Yearly`  
**Asset classes:** US equity, bonds, commodities, gold, real estate  
**Leveraged ETF weight:** 10.9%  
**Exact Testfol method:** No

**Why it is good**

- Test CAGR of 20.7%, +5.0% versus SPY.
- Observed max drawdown of -19.6% in the locked test window.
- Spans 5 asset classes: US equity, bonds, commodities, gold, real estate.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- The exact dynamic/local method is not directly Testfol-validated.
- Worst observed rolling three-month return was -13.7%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 20.7% | -19.6% | 1.29 | 262 | 0.0% |
| EMA100 monthly cash | 17.3% | -12.8% | 1.19 | Not recovered | 14.5% |
| 10% component crash cash | 17.3% | -11.4% | 1.37 | 43 | 26.0% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #8: DBMF 29.3% / GLD 29.4% / QQQ 28.2% / TQQQ 0.9% / VNQ 12.3%

**Method:** `EMA100:replacement:Monthly`  
**Asset classes:** US equity, gold, managed futures, real estate  
**Leveraged ETF weight:** 0.9%  
**Exact Testfol method:** No

**Why it is good**

- Test CAGR of 19.5%, +3.8% versus SPY.
- Observed max drawdown of -19.2% in the locked test window.
- Spans 4 asset classes: US equity, gold, managed futures, real estate.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- QQQ/TQQQ overlap is 29.0%, so the equity sleeve is less diversified than the ticker count suggests.
- The exact dynamic/local method is not directly Testfol-validated.
- Worst observed rolling three-month return was -15.6%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 19.5% | -19.2% | 1.16 | Not recovered | 0.0% |
| EMA100 monthly cash | 12.9% | -19.2% | 0.86 | Not recovered | 17.1% |
| 10% component crash cash | 13.9% | -25.5% | 1.02 | Not recovered | 23.0% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #9: DBC 25.4% / DBMF 11.6% / GLD 37.4% / SHV 4.3% / UPRO 7.1% / VTV 14.2%

**Method:** `EMA100:pro_rata:Monthly`  
**Asset classes:** US equity, cash, commodities, gold, managed futures  
**Leveraged ETF weight:** 7.1%  
**Exact Testfol method:** No

**Why it is good**

- Test CAGR of 17.7%, +2.0% versus SPY.
- Observed max drawdown of -12.2% in the locked test window.
- Spans 5 asset classes: US equity, cash, commodities, gold, managed futures.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- It did not beat SPY in most rolling three-year windows.
- The exact dynamic/local method is not directly Testfol-validated.
- Worst observed rolling three-month return was -9.6%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 17.7% | -12.2% | 1.22 | 252 | 0.0% |
| EMA100 monthly cash | 10.6% | -14.7% | 0.91 | 356 | 17.4% |
| 10% component crash cash | 17.7% | -12.2% | 1.30 | Not recovered | 9.5% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

### #10: DBC 25.4% / DBMF 11.6% / GLD 37.4% / SHV 4.3% / UPRO 7.1% / VTV 14.2%

**Method:** `Calendar:Yearly`  
**Asset classes:** US equity, cash, commodities, gold, managed futures  
**Leveraged ETF weight:** 7.1%  
**Exact Testfol method:** No

**Why it is good**

- Test CAGR of 17.4%, +1.7% versus SPY.
- Observed max drawdown of -14.6% in the locked test window.
- Spans 5 asset classes: US equity, cash, commodities, gold, managed futures.
- Keeps explicit leveraged sleeves relatively contained.

**Downside risks**

- It did not beat SPY in most rolling three-year windows.
- The exact dynamic/local method is not directly Testfol-validated.
- Worst observed rolling three-month return was -10.5%.

**Protection I would use**

- Keep the leveraged sleeve capped and rebalance rather than averaging aggressively after a leveraged loss.
- Retain the diversifier sleeves; they are the intended non-equity shock absorbers.
- Treat the bond/cash sleeve as liquidity, not as a guarantee; rates and inflation can hurt it too.
- Use the tested monthly EMA100 cash overlay only if accepting whipsaw and missed rebounds is preferable to staying invested through a large trend break.

**Protection test, 2021-01-01 through 2026-08-17**

| Version | CAGR | Max DD | Sharpe | Recovery days | Cash time |
|---|---:|---:|---:|---:|---:|
| Baseline | 17.4% | -14.6% | 1.35 | 491 | 0.0% |
| EMA100 monthly cash | 14.9% | -10.2% | 1.26 | Not recovered | 15.3% |
| 10% component crash cash | 17.3% | -11.7% | 1.51 | Not recovered | 12.5% |

**Interpretation:** The overlay is not free insurance. Compare the reduced drawdown against the CAGR lost, cash time, and the possibility of exiting immediately before a rebound.

## Bottom Line

A cash overlay can reduce drawdown, but it can also sell after a shock and miss the rebound. The component-crash rule is especially prone to false exits in portfolios containing TQQQ, UPRO, TMF, TYD, or SSO. The EMA100 rule is slower and more systematic, but it can still suffer a large loss before the monthly exit and can whipsaw in sideways markets.

The practical protection is position sizing, limiting leveraged overlap, preserving liquidity, and accepting that no rule guarantees protection from a synchronized selloff.
