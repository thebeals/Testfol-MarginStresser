# LETF / Retirement Backtesting — Community Research

**Date:** 2026-07-11  
**Purpose:** Validate our build against community best practice before Phase 2+ usage.  
**Scope:** Tools, synthetic LETF methodology, daily data providers, gap analysis vs `retirement-bt`.

---

## User Decisions (2026-07-11)

Grilling session outcomes — these govern Phase 2+ usage:

| Decision | Choice |
|----------|--------|
| **Use case** | **Both** — free exploration for research + decision-grade rigor for final allocation picks |
| **Pre-inception synthetic LETF** | **Allowed** with prominent low-confidence warnings (pre-2009 paths are directional only) |
| **Spread semantics** | Document as `implementation_spread` — theoretical 40–50 bps vs tuned NAV calibration (133–189 bps) |
| **Data reproducibility** | Frozen cache snapshots with `fetch_date` + `data_hash`; warn when cache >30 days |
| **Testfolio cross-check** | Required reconciliation test (`tests/test_testfolio_parity.py`) before trusting long-horizon LETF CAGR |
| **Rebalance costs** | Model via `rebalance_cost_bps` slider (default 5 bps) |

---

## Executive Summary

The community has converged on **daily total-return synthesis with explicit borrow/financing cost** for pre-inception LETF backtests. The simplified Teddy Koker formula (`L × r − ER/252`) is still widely circulated on Reddit and blogs but is **known to overstate 3× LETF returns by ~60%** over high-rate windows (2015–2024 TQQQ: 33× vs 20× real). **Testfolio** is the de facto reference implementation for DIY retirement/LETF backtesting; **Portfolio Visualizer** remains popular for deep analytics but uses **monthly** data and does not natively support leveraged portfolios.

**Our build is directionally correct** on the hardest problem — we implement the full borrow-cost formula, validate against actual NAV since inception, and block backtests when validation fails. That puts us ahead of most Reddit/Bogleheads DIY scripts and on par with sma200-bt / Testfolio methodology.

**Critical risks remain:** (1) pre-inception history depends on Shiller monthly→daily and proxy chains with low intraday fidelity; (2) per-fund spread tuning (130–190 bps) is an empirical fudge, not a disclosed swap spread; (3) yfinance-only daily adjusted close has reproducibility and gap issues; (4) the simulator has no slippage, taxes, or execution-timing realism; (5) user asked for **daily candlesticks** but we only store/use a single adjusted close series.

**Recommendation:** Do not treat pre-2009 synthetic LETF results as decision-grade until cross-validated against Testfolio and until user requirements (accounts, rebalance, taxes) are captured. yfinance is acceptable for Phase 2 **inception-era** validation; consider Tiingo ($10–30/mo) before relying on multi-decade synthetic paths.

---

## 1. What the Community Builds (Tools & Stacks)

### Tier 1 — What retirement / LETF DIY investors actually use

| Tool | Role | Daily data? | LETF synthetic? | Notes |
|------|------|-------------|-------------------|-------|
| **[Testfolio](https://testfol.io/)** | Primary LETF-aware backtester | Yes | Yes (`?L=`, `?E=`, `_SIM` tickers) | Community reference for synthetic LETF; FEDFUNDS + 0.50% default spread; sim funds extend history |
| **[Portfolio Visualizer](https://www.portfoliovisualizer.com/)** | Deep analytics (Monte Carlo, factor regression) | **No (monthly)** | Limited / no native LETF | Still cited in Bogleheads HFEA threads; Tyler explicitly not a leverage fan |
| **Portfolio Charts** | Visual allocation research | Daily for many series | No LETF modifiers | Good for strategic allocation, not LETF synthesis |
| **Custom spreadsheets** | Bogleheads "Simba" extensions | Mixed | Manual LETF columns | [Bogleheads LETF modeling thread](https://www.bogleheads.org/forum/viewtopic.php?t=272007) — shared Google Sheets for 40/60 UPRO/TMF |

**Community sentiment (Bogleheads / Risk Parity Chronicles / Awalyt 2026):**
- Testfolio wins for **daily resolution + LETF modifiers + extended sim history**.
- Portfolio Visualizer wins for **Monte Carlo and factor work** but misses intra-month drawdowns.
- Testfolio trade-offs: steeper learning curve ("secret decoder ring"), occasional EODHD data disputes on Bogleheads, no Monte Carlo yet.

### Tier 2 — Python quant stacks (r/algotrading pattern)

Typical progression cited across [r/algotrading-adjacent guides](https://www.youngju.dev/blog/culture/2026-05-16-trading-bots-quant-tools-2026-lean-quantconnect-backtrader-zipline-freqtrade-hummingbot-nautilus-vectorbt-deep-dive.en):

1. **Hours 1–10:** `pandas` + **yfinance** — SMA crossover, buy-and-hold (exactly Teddy Koker's entry path).
2. **Hours 10–30:** **vectorbt** or **backtesting.py** — add fees/slippage, parameter sweeps.
3. **Hours 30+:** **Backtrader** — event-driven validation before any live capital.

| Library | Best for | Daily candles | LETF-specific |
|---------|----------|---------------|---------------|
| **Custom pandas** | LETF return series, portfolio CAGR | Yes (if data source provides) | Teddy Koker, sma200-bt, **our engine** |
| **vectorbt** | Parameter sweeps, walk-forward | Yes | No native LETF; must pre-build synthetic series |
| **Backtrader** | Execution realism, multi-asset rebalance | Yes | Teddy Koker UPRO/TMF 60/40 example |
| **bt** (pmorissette) | Portfolio-level weights | Daily | Used with custom LETF series |
| **Zipline-reloaded** | Factor pipelines | Daily | Heavy install; declining retail use |

**Key r/algotrading pitfalls (repeated across Medium, GitHub, PyQuant):**
- **Look-ahead bias:** signal and fill on same bar; fix = trade next open.
- **Survivorship bias:** free data drops delisted names (matters for stock screens, less for major ETFs).
- **Ignoring transaction costs** on rebalance-heavy strategies.
- **Overfitting** via unconstrained parameter search (vectorbt makes this easy).

### Tier 3 — LETF-specific Python libraries

| Library | What it does |
|---------|--------------|
| **[sma200-bt](https://github.com/prismlfx/sma200-bt)** | Testfolio-compatible `synthetic_letf_returns()` + `fetch_tbill_rate()`; extracted after author posted wrong Reddit numbers |
| **Teddy Koker blog/notebooks** | `sim_leverage()` simplified formula + Backtrader portfolio examples |
| **Testfol-MarginStresser** | Documents Testfolio local-engine methodology ([methodology.md](https://github.com/Acelogic/Testfol-MarginStresser/blob/master/docs/methodology.md)) |

### What they use for daily candles

| Use case | Typical source |
|----------|----------------|
| Prototyping / LETF validation | **yfinance** (`auto_adjust=True` → dividend+split adjusted OHLC) |
| Testfolio production | **EODHD** (community reports occasional dividend/CAGR discrepancies) |
| Testfolio fallback / shadow engine | **Polygon.io** → **yfinance** failover |
| Serious EOD research | **Tiingo** ($10–30/mo) |
| Survivorship-free equities | **Norgate** ($35–50/mo) — overkill for SPY/TLT/UPRO but gold standard for stock picking |
| Borrow rates | **^IRX** (13-week T-bill), **FRED FEDFUNDS/DFF**, historical **LIBOR** (pre-2001 splice) |

**Important:** Most retirement/LETF backtests use **adjusted close / total return**, not raw OHLC candlesticks. Candlestick charts are rare in published LETF research; total-return daily series is the consensus input.

---

## 2. LETF Synthetic Methodology Consensus

### Formula evolution

| Variant | Formula | Who uses it | Verdict |
|---------|---------|-------------|---------|
| **Simplified** | `L × r − ER/252` | Teddy Koker (2019), many Reddit/blog scripts | ❌ Wrong for post-2009 high-rate eras; +62% TQQQ drift 2015–2024 |
| **Full (canonical)** | `L × r − ER/252 − (L−1) × (rate + spread) / 252` | Testfolio, sma200-bt, **our build** | ✅ Community consensus since ~2024–2026 |
| **Swap-weighted** | Above × `swap_weight` + futures carry term | r/LETFs (u/cryptojam4004), sma200-bt issue #1 | 🔶 More accurate; SW≈0.92 for TQQQ today |
| **Academic monthly** | Avellaneda-Zhang / Tang-Xu with vol decay term | Bogleheads siamond blog | 🔶 For monthly-only history; quantifies "variance drain" |

Our implementation (`src/retirement_bt/synthetic/letf.py`):

```python
daily_return = L * r - ER/252 - (|L|-1) * (borrow_rate + spread) / 252
```

This matches Testfolio and sma200-bt. ✅

### Borrow rate sources (historical consensus)

| Source | Used by | History | Notes |
|--------|---------|---------|-------|
| **1-month LIBOR** | Bogleheads EfficientInvestor (original HFEA thread) | Pre-2021 | Deprecated; must splice to SOFR/EFFR for modern |
| **Fed Funds Effective (FRED DFF/FEDFUNDS)** | Testfolio, Bogleheads siamond (1955+ splice) | 1954+ monthly, 1955+ daily (DFF) | Testfolio default for `?L=` modifiers |
| **^IRX (13-week T-bill yield)** | sma200-bt, **our build** | ETF history via yfinance | Close to but not identical to FEDFUNDS; sma200-bt says "within a few bps CAGR" vs Testfolio |
| **Constant fallback** | Various lazy scripts | — | ❌ 3% constant (our fallback) degrades pre-2009 accuracy |

**Spread (implementation margin above risk-free):**
- Testfolio default: **50 bps** (`SP=` override available)
- sma200-bt default: **40 bps**
- Bogleheads siamond: **0.5% fudge for 2× equity, 1.0% for 3× equity** (empirical, not theoretical swap spread)
- ProShares prospectus language: ~T-bill + spread; community estimates **30–50 bps** typical
- **Our tuned spreads: 133–189 bps** — much higher than theoretical; see Gap Analysis

### Validation approach (consensus)

1. **Since-inception NAV match** — build synthetic from underlying (SPY, not extended proxy), compare to actual UPRO/TQQQ/TMF NAV.
2. **Correlation > 0.99** on normalized price series (Teddy Koker visual match; our automated gate).
3. **CAGR difference < ~0.5%** over overlap (our gate; Bogleheads users report ~4% CAGR gap = "$3M vs $10M" over 30 years when borrow omitted).
4. **Cross-check Testfolio** — sma200-bt author validated after Reddit correction ("does this match Testfolio?").
5. **Telltale charts** — Bogleheads siamond uses gross-return telltales before expense ratio.

### Common mistakes (frequently cited)

| Mistake | Impact | Source |
|---------|--------|--------|
| No borrow cost term | +60% wealth multiple over 10y (TQQQ) | [sma200.trade article](https://sma200.trade/learn/leveraged-etf-borrow-cost), sma200-bt README |
| Using **close** instead of **adj close / total return** | Dividend gaps distort leverage | Teddy Koker uses VFINX Adj Close; yfinance `auto_adjust=True` |
| Confusing **volatility decay** with expense drag | Misattribute path-dependency | Bogleheads wiki "Variance drain"; siamond monthly formula |
| Assuming leverage works on **monthly** returns | Overstates smooth-path leverage | Must daily-reset |
| Ignoring **TMF/treasury** synthesis difficulty | TMF "has some issues" pre-2013 | [Bogleheads t=272007](https://www.bogleheads.org/forum/viewtopic.php?t=272007) |
| Tuning spread on **extended proxy** chain | Double error compounding | Our validation uses direct underlying only ✅ |
| Publishing simple-formula Reddit numbers | Community backlash May 2026 | sma200.trade author corrected r/LETFs post |

---

## 3. Data Source Recommendations for Daily Accuracy

### Provider comparison (daily adjusted close / total return)

| Provider | Adj close quality | History depth | Cost | Survivorship | Retirement LETF fit |
|----------|-------------------|---------------|------|--------------|----------------------|
| **yfinance** | Good for major ETFs; **retroactive adj changes** | Decades for SPY/VFINX | Free | No delisted | ⭐⭐⭐ Prototype + inception validation |
| **Tiingo** | Clean EOD; documented adj methodology | 30+ years | Free tier / $10–30/mo | Active symbols | ⭐⭐⭐⭐ Best paid upgrade for EOD |
| **Polygon** | `adjusted=true` = splits only; dividends separate | 5y free / paid deeper | $29+/mo | Some delisted | ⭐⭐⭐ If you need OHLC + corporate actions control |
| **Alpha Vantage** | Free tier: daily only, strict rate limits | Limited | Free / paid | No | ⭐⭐ Avoid for bulk history |
| **Norgate** | Excellent adj EOD | Decades | $35–50/mo | **Survivorship-free** | ⭐⭐ Overkill for ETF-only; needed for stock screens |
| **FRED** | N/A (rates only) | 1950s+ | Free | — | ⭐⭐⭐⭐ Essential for borrow rates |
| **Shiller** | Monthly total return | 1871+ | Free | — | ⭐⭐ Extension only; not daily-native |

### Rankings for **retirement ETF backtesting**

1. **Inception-era (2009+ UPRO, 2010+ TMF):** yfinance is **good enough** if cached and validation-gated (what we do).
2. **Multi-decade strategic allocation (VFINX/TLT):** yfinance + mutual fund proxies — acceptable with documented limitations.
3. **Pre-1993 synthetic LETF paths:** Shiller monthly→daily is **low confidence** regardless of price source; community uses similar extensions (Testfolio sim funds, Portfolio Charts calculators).
4. **Decision-grade pre-inception LETF:** Cross-validate with **Testfolio**; consider **Tiingo** for underlying total return if yfinance gaps appear.
5. **Stock momentum / broad universe:** Norgate or similar — our momentum strategy flag already warns survivorship bias ✅

### Survivorship bias

- **Material for:** stock-picking, index membership strategies, delisted small caps (can inflate CAGR 1.5–2%/yr).
- **Low materiality for:** fixed ticker list (UPRO, TMF, VFINX, TLT) — these are the survivors by construction.
- Our simulator sets `survivorship_bias_warning` for momentum strategies ✅

### Is yfinance "good enough"?

**Yes for:**
- Phase 2 validation against known ETF NAV since inception
- Major liquid ETF/mutual fund proxies (SPY, VFINX, VUSTX, TLT)
- Prototyping and UI development

**No for:**
- Production allocation decisions on 1985–2008 synthetic LETF paths without secondary validation
- Strategies sensitive to exact dividend timing
- Reproducible research without **frozen/cache-versioned** datasets (adj prices change when new dividends post)
- Raw OHLC candlestick analysis (yfinance `auto_adjust=True` overwrites OHLC; not true unadjusted candles)

**Community quote (YouTube provider comparison, 2026):** yfinance fine for "~80% of retail traders testing daily/weekly strategies" but "always validate your data" before serious backtesting.

---

## 4. Gap Analysis: Our Build vs Best Practice

### What we got RIGHT ✅

| Area | Evidence |
|------|----------|
| **Full borrow-cost formula** | Matches Testfolio / sma200-bt canonical form |
| **Mandatory NAV validation gates** | corr > 0.99, CAGR diff < 0.5%; blocks backtest on failure |
| **Validation on direct underlying** | `_validation_underlying()` uses SPY/TLT/QQQ only, not Shiller chain |
| **Source hierarchy documented** | README + `pipeline.py` — actual → proxy → Shiller |
| **low_confidence flags** | Shiller pre-1993, managed futures proxy |
| **Borrow rate fallback chain** | ^IRX → FRED DGS3MO → 3% with warnings |
| **Daily simulation engine** | Appropriate for LETF daily reset |
| **Proxy chains** | SPY→VFINX→Shiller, TLT→VUSTX (community standard) |
| **Explicit simple-formula rejection** | Tests prove borrow cost impact; docs warn against simplified formula |
| **UI validation gate** | Analyze blocked until LETF validation passes |

### What we got WRONG or RISKY ⚠️

| Issue | Severity | Detail |
|-------|----------|--------|
| **No Testfolio cross-validation** | High | We self-validate against yfinance NAV but never compare CAGR to Testfolio `SPY?L=3&E=0.91` |
| **Tuned spreads 133–189 bps** | High | Community uses 40–50 bps theoretical; our calibration absorbs tracking error + all frictions into spread — **README table also mismatches code** (README says UPRO 146 bps; code has 169 bps) |
| **Shiller monthly→daily** | High | Forward-fill loses intramonth volatility; Bogleheads siamond uses daily vol proxies — our pre-1993 LETF paths are **directional only** |
| **No slippage / expense beyond LETF ER** | Medium | Community rebalance-sensitive tests (Testfolio 600-run tool); we rebalance at zero cost |
| **No tax modeling** | Medium | Testfolio MarginStresser has tax lots; retirement accounts differ (401k pre-tax vs Roth) |
| **Execution timing** | Medium | Rebalance first day of month/quarter/year; r/algotrading recommends next-open after signal |
| **yfinance-only, mutable adj prices** | Medium | No dataset versioning; reproducibility risk |
| **No swap_weight parameter** | Low-Med | r/LETFs / sma200-bt issue #1; TQQQ ~92% swap |
| **FEDFUNDS vs ^IRX** | Low | Testfolio uses FEDFUNDS; we use ^IRX — sma200-bt says few bps CAGR difference |
| **TMF treasury synthesis** | Medium | Bogleheads: "TMF has some issues" pre-2013; bond LETF synthesis harder than equity |
| **Candlesticks vs adj close** | Medium | User wants daily candlesticks; we only fetch/store single adjusted close series |
| **Contributions heuristic** | Low | `day <= 3` monthly contrib — arbitrary |
| **QQQ chain lacks extension** | Low | QQQ has no VFINX/Shiller fallback; TQQQ pre-1999 limited to QQQ history (Mar 1999) |
| **Managed futures proxy** | Low | DBMF→KMLM flagged but still tempting to misuse |

### What to change before Phase 2+ usage

1. **Add Testfolio reconciliation script** — same portfolio (e.g. 40/60 UPRO/TMF 1987–2024) must match within ~1% CAGR.
2. **Freeze data snapshots** — cache with fetch date; warn when adj prices drift.
3. **Document spread semantics** — rename `borrow_spread` to `implementation_spread` in UI; show 40 bps theoretical vs tuned value.
4. **Capture user requirements** (Section 5) before trusting pre-inception charts.
5. **Optional Tiingo backend** — environment flag, same validation gates.
6. **If candlesticks required:** fetch `auto_adjust=False` OHLC + separate adj factor, or use Tiingo/Polygon; do not use auto_adjust OHLC as "candles."
7. **Add slippage + rebalance cost** parameter (even 5 bps default).

---

## 5. Questions We SHOULD Have Asked the User

1. **Which accounts are you actually optimizing for?** (401k pre-tax, Traditional IRA, Roth, taxable brokerage — each has different tax/rebalance constraints.)
2. **What is the intended rebalancing rule?** (Calendar quarterly like HFEA? Threshold bands? Drift %? Day of month? Execute at close or next open?)
3. **Will you hold synthetic pre-inception LETF data (pre-2009 UPRO) or only live-era ETFs?** (Determines whether Shiller extension is in scope.)
4. **What maximum drawdown / tail risk is acceptable?** (LETF paths have −99% synthetic drawdowns — is that an useful chart or misleading?)
5. **Do you need tax-aware simulation?** (Dividend taxes, rebalance cap gains, Roth conversion timing.)
6. **What contribution pattern is realistic?** (Fixed monthly vs % of salary vs lump-sum; employer match?)
7. **Should we model transaction costs / slippage?** (Even 0.05% per rebalance changes multi-decade LETF rotation strategies.)
8. **Is the benchmark VFINX/SPY for comparison, or a policy portfolio (e.g. 60/40)?**
9. **Do you want to compare against Testfolio / Portfolio Visualizer outputs as ground truth?**
10. **Daily candlesticks: for display only, or signal generation (SMA200, etc.)?** (Affects whether adj close is sufficient.)
11. **Which LETFs are in scope beyond UPRO/TMF/TQQQ?** (Inverse, commodity, single-stock LETFs have different synthesis quality.)
12. **Is this tool for research exploration or for deciding real allocation percentages?** (Determines data budget: free yfinance vs paid Tiingo.)

---

## 6. Recommended Architecture Adjustments

### Keep (solid choices)

| Component | Rationale |
|-----------|-----------|
| **NiceGUI + Plotly** | Matches "Portfolio Visualizer inspired" goal; fine for personal retirement tool |
| **pandas return engine** | Community norm for LETF portfolios; Backtrader/vectorbt optional later |
| **Full LETF formula + validation gates** | Ahead of most DIY builds; aligns with Testfolio |
| **Shiller extension with low_confidence** | Honest about pre-1993 limits |
| **VFINX/VUSTX proxy chains** | Standard community approach |
| **CLI `retirement-bt validate`** | Good pre-flight check |

### Replace or defer

| Component | Recommendation |
|-----------|----------------|
| **yfinance as sole source** | Keep default; add **Tiingo optional backend** before decision-grade use |
| **Spread tuning as opaque bps** | Refactor to `theoretical_spread + tracking_error_calibration` with UI disclosure |
| **Single-series "prices"** | If candlesticks required, add parallel OHLC store — don't pretend adj close is candles |

### Add (before trusting long-horizon results)

| Feature | Priority |
|---------|----------|
| Testfolio / sma200-bt CAGR reconciliation test | P0 |
| Frozen dataset versioning + adj drift warning | P0 |
| Slippage + rebalance cost parameter | P1 |
| FEDFUNDS toggle (vs ^IRX) for Testfolio parity | P1 |
| `swap_weight` per fund (from prospectus) | P2 |
| Tax lot / account-type modeling | P2 |
| Execution at next-day open option | P2 |

### Daily candlestick handling (specifics)

User wants daily candlesticks. Community LETF backtests **do not** typically use candlestick bodies for synthesis — they use **total return daily series**. Recommended approach:

1. **For LETF synthesis:** Continue using adjusted close / total return (correct).
2. **For chart display:** Optionally show underlying OHLC from `auto_adjust=False` with separate dividend adjustment — label clearly as "unadjusted OHLC" vs "total return line."
3. **For indicators (SMA200):** Compute on adjusted close or total return index, not unadjusted close — matches sma200.trade methodology.
4. **Do not** use yfinance `auto_adjust=True` OHLC as candlesticks; those are adjusted and will look like candles but aren't tradable prices.

---

## Appendix A — Key Community Sources

| Source | URL | Key insight |
|--------|-----|-------------|
| Teddy Koker LETF simulation | https://teddykoker.com/2019/04/simulating-historical-performance-of-leveraged-etfs-in-python/ | Simplified formula; VFINX proxy; Backtrader follow-up |
| Teddy Koker Backtrader LETF | https://teddykoker.com/2019/04/backtesting-portfolios-of-leveraged-etfs-in-python-with-backtrader/ | UPRO/TMF 60/40; rebalance every 20 days |
| sma200.trade borrow cost | https://sma200.trade/learn/leveraged-etf-borrow-cost | +62% simple formula error; Testfolio as reference |
| sma200-bt | https://github.com/prismlfx/sma200-bt | Testfolio-compatible Python; ^IRX + 40 bps |
| Testfolio methodology | https://github.com/Acelogic/Testfol-MarginStresser/blob/master/docs/methodology.md | FEDFUNDS + 50 bps; synthetic `?L=` |
| Bogleheads HFEA / LETF thread | https://www.bogleheads.org/forum/viewtopic.php?t=272007 | LIBOR + ER formula; 19% vs 23.75% CAGR when borrow omitted |
| Bogleheads LETF modeling (siamond) | https://www.bogleheads.org/blog/2022/12/25/leveraged-funds-historical-modeling/ | Fudge factors; daily vs monthly methods; data splicing |
| r/LETFs correction thread | Referenced in sma200.trade (May 2026) | "Does this match Testfolio?" caught 60% error |
| r/LETFs swap weight | https://github.com/prismlfx/sma200-bt/issues/1 (u/cryptojam4004) | SW parameter for futures vs swaps |
| Portfolio Visualizer vs Testfolio | https://awalyt.com/insights/best-portfolio-backtesting-tools-2026 | PV monthly vs Testfolio daily |

## Appendix B — Our Implementation Map

| Community pattern | Our file |
|-------------------|----------|
| Data pipeline / stitching | `src/retirement_bt/data/pipeline.py` |
| yfinance fetch | `src/retirement_bt/data/fetchers.py` |
| ^IRX / FRED borrow | `src/retirement_bt/data/fred.py` |
| Shiller extension | `src/retirement_bt/data/shiller.py` |
| LETF synthesis | `src/retirement_bt/synthetic/letf.py` |
| NAV validation | `src/retirement_bt/synthetic/validation.py` |
| Backtest engine | `src/retirement_bt/engine/simulator.py` |
| Web UI | `src/retirement_bt/ui/app.py` |
| Methodology docs | `README.md` § Data Accuracy Methodology |

---

*Research conducted 2026-07-11. Reddit direct search yielded limited indexed results; r/LETFs insights cited via sma200.trade correction article and sma200-bt GitHub (author published original numbers to r/LETFs). Bogleheads forum threads provide primary community consensus for HFEA/UPRO/TMF methodology.*
