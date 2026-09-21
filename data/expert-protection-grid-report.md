# Expert Protection Grid Report

Evaluated **3,000** portfolio/configuration permutations across the 10 expert candidates.
Signals use prior closing data and execute on the next session. The 2015-2020 period is used for train/validation; 2021-2026 is the locked test period.

## Search Design

The grid varied EMA span, review frequency, component crash threshold, trailing stop, de-risk fraction, trigger combination, and cooldown. A fixed-seed sample of 300 configurations was applied to each portfolio.

The consensus layer requires a rule to pass the robust filter on at least four portfolios. Robust filtering allows no more than one percentage point of test CAGR loss and requires non-worse test drawdown.

## Consensus Rules

| Rank | Trigger | Crash | Review | Cooldown | De-risk | Portfolios | Median CAGR Delta | Median DD Improvement |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | crash | 12.5% | Monthly | 60 days | 100% | 4 | -0.10% | +6.95% |
| 2 | crash | 12.5% | Monthly | 20 days | 100% | 7 | +0.39% | +4.95% |
| 3 | combined | 10.0% | Daily | 0 days | 50% | 4 | -0.37% | +5.52% |
| 4 | combined | 7.5% | Daily | 0 days | 50% | 4 | -0.61% | +5.72% |
| 5 | combined | 12.5% | Monthly | 20 days | 50% | 7 | -0.35% | +4.95% |
| 6 | combined | 12.5% | Monthly | 20 days | 50% | 4 | -0.48% | +4.91% |
| 7 | crash | 12.5% | Daily | 0 days | 100% | 5 | +0.13% | +3.05% |
| 8 | combined | 12.5% | Daily | 0 days | 50% | 4 | -0.68% | +4.84% |
| 9 | crash | 7.5% | Daily | 0 days | 50% | 4 | -0.77% | +5.09% |
| 10 | combined | 12.5% | Monthly | 0 days | 50% | 8 | -0.58% | +4.51% |

The preferred compromise is the rule that passes on the most candidates while keeping median CAGR impact near zero and improving drawdown. It is not selected because it has the highest isolated backtest CAGR.

## Best Robust Result Per Candidate

| Expert Rank | Portfolio | Method | Protection | Test CAGR Delta | Test DD Improvement |
|---:|---|---|---|---:|---:|
| 1 | Allocation 1 | `AbsoluteBand:Yearly:0.2` | `{"cooldown": 0, "crash_threshold": 0.125, "derisk": 1.0, "ema_span": 100, "review": "Daily", "trailing_stop": 0.15, "trigger": "crash"}` | +0.32% | +4.95% |
| 2 | Allocation 2 | `AbsoluteBand:Yearly:0.2` | `{"cooldown": 20, "crash_threshold": 0.125, "derisk": 1.0, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.0, "trigger": "crash"}` | +0.86% | +9.82% |
| 3 | Allocation 8 | `AbsoluteBand:Yearly:0.05` | `{"cooldown": 20, "crash_threshold": 0.1, "derisk": 0.5, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.1, "trigger": "crash"}` | +0.09% | +5.63% |
| 4 | Allocation 5 | `Calendar:Yearly` | `{"cooldown": 20, "crash_threshold": 0.125, "derisk": 1.0, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.0, "trigger": "crash"}` | +0.43% | +3.80% |
| 5 | Allocation 1 | `Calendar:Yearly` | `{"cooldown": 0, "crash_threshold": 0.125, "derisk": 1.0, "ema_span": 100, "review": "Daily", "trailing_stop": 0.15, "trigger": "crash"}` | +0.63% | +5.43% |
| 6 | Allocation 7 | `Calendar:Yearly` | `{"cooldown": 20, "crash_threshold": 0.125, "derisk": 1.0, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.0, "trigger": "crash"}` | +0.39% | +4.08% |
| 7 | Allocation 2 | `Calendar:Yearly` | `{"cooldown": 20, "crash_threshold": 0.125, "derisk": 1.0, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.0, "trigger": "crash"}` | +0.88% | +9.88% |
| 8 | Allocation 4 | `EMA100:replacement:Monthly` | `{"cooldown": 60, "crash_threshold": 0.15, "derisk": 0.5, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.0, "trigger": "crash"}` | +0.10% | +0.00% |
| 9 | Allocation 8 | `EMA100:pro_rata:Monthly` | `{"cooldown": 60, "crash_threshold": 0.1, "derisk": 0.5, "ema_span": 100, "review": "Weekly", "trailing_stop": 0.15, "trigger": "crash"}` | -0.41% | +3.12% |
| 10 | Allocation 8 | `Calendar:Yearly` | `{"cooldown": 20, "crash_threshold": 0.1, "derisk": 0.5, "ema_span": 100, "review": "Monthly", "trailing_stop": 0.1, "trigger": "crash"}` | -0.24% | +4.38% |

## Cost Sensitivity

The top consensus rule's median test CAGR delta across the evaluated candidates is reported below:

| Cost | Median CAGR Delta | Median DD Improvement |
|---:|---:|---:|
| 0 bps | -0.77% | +3.94% |
| 10 bps | -0.86% | +3.94% |
| 25 bps | -1.00% | +3.94% |

## Caveats

- This is a sampled grid, not an exhaustive proof of the best future rule.
- The expert top 10 were selected from the same historical universe, so portfolio selection remains subject to selection bias.
- Component crash rules treat every allocated ticker as active, including assets that a dynamic EMA method may have sold; this is conservative but can over-trigger.
- Results exclude taxes, slippage beyond the sensitivity table, and investor execution errors.
