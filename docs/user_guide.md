# User Guide

Welcome to the Testfol Margin Stresser User Guide. This application allows you to simulate leveraged portfolio performance over historical periods, tracking margin debt, equity levels, and potential margin calls.

### Preset Categories

The preset selector groups strategies into five stable categories while preserving
each preset's original name and settings:

- **Core NDXMEGA**: primary NDXMEGA, QQUP, and NDXMEGASPLIT strategies.
- **Single Asset & Proxies**: one-asset portfolios and simulated ETF proxies.
- **Classic & Diversified**: conventional balanced and multi-asset allocations.
- **Leveraged Strategies**: HFEA and leveraged split portfolios.
- **Research & Experimental**: dynamic, cherry-picked, and exploratory strategies.

## Getting Started

### 1. Configure Global Settings
The sidebar contains all the configuration options for your backtest.

**Date Range**
- **Start/End Date**: Select the simulation period.
- **Range**: You can backtest as far back as **1884** (ticker data permitting).

**Portfolios**
- **Save/Load**: Use the "Saved Portfolios" section to persist your favorite strategies.
- **Allocation**: Enter your portfolio weights in the table. Must sum to 100%.

### 2. Data Provider Setup (Optional)
The app fetches price data from multiple sources with automatic fallback:
-   **Default**: yfinance (free, no setup required)
-   **Polygon.io**: Better coverage for delisted tickers. Set `POLYGON_API_KEY` environment variable.
-   **Testfol.io**: Used for server-side backtesting. Falls back to local engine if unavailable.

If the Testfol API goes down, backtests automatically run locally using the Shadow Engine with prices from your configured providers. A warning banner will notify you when this happens.

### 3. Strategy Configuration Tabs

#### 💼 Portfolio
- **Start Value**: Initial capital (e.g., $10,000).
- **Cashflow**: Periodic contributions (positive) or withdrawals (negative).
- **Tax Simulation**:
    -   **Filing Status**: Affects tax brackets.
    -   **Other Income**: Used to determine your base tax bracket.
    -   **Calculation Method**:
        -   *Historical Smart (Default)*: Uses actual historical inclusion rates and brackets.
        -   *Historical Max*: Applies the top capital gains rate for that year.
        -   *2026 Fixed*: Applies modern brackets to all years.

#### 🏦 Margin & Financing
- **Tax Payment Mode**: 
    -   *Pay from Cash*: Sell assets to pay taxes (reduces compounding).
    -   *Pay with Margin*: Borrow to pay taxes (increases leverage).
- **Loan Config**: Set starting loan or initial equity %.
- **Margin Rate Model**: Select how interest is calculated:
    -   **Tiered (Blended)**: (Default) Uses IBKR Pro-style tiered rates, blending different spreads based on your loan size against the historical Fed Funds benchmark.
    -   **Variable (Fed + Spread)**: Uses the historical **Fed Funds Rate** + a fixed spread.
    -   **Fixed Annual %**: Uses a constant interest rate.
- **Maintenance**: Set the default maintenance requirement.

#### ⚙️ Settings
- **Chart Style**: Switch between "Classic", "Dashboard", and "Candlestick".
- **Log Scale**: Toggle logarithmic axis for long-term growth charts.

---

## Ticker Modifiers
You can use Testfol modifiers in the ticker symbol to simulate leverage or expense ratios:

| Modifier | Description | Example |
| :--- | :--- | :--- |
| `?L=X` | **Leverage**: Multiplies daily returns by X and subtracts Testfol-compatible DFF financing plus the default 0.40% spread for synthetic exposure. | `SPY?L=2` (2x S&P 500) |
| `?E=X` | **Expense Ratio**: Applies annual fund operating expense ratio (%). This is separate from leverage financing cost. | `QQQ?E=0.20` (0.20% annual fee) |
| `?D=X` | **Drag**: Applies annual drag (legacy alias for expense ratio). | `SPY?D=0.50` (0.50% fee) |
| `?UE=X` | **Underlying return adjustment**: Adds `X% / 252` before caps and leverage. | `SPY?UE=1` |
| `?CU=X&CL=Y` | **Daily caps**: Caps the underlying daily return at `X%` and `Y%` before leverage. | `SPY?CU=2&CL=-2` |
| `?SW=X&SP=Y` | **Financing controls**: Overrides swap exposure and annual spread. | `SPY?L=2&SW=1&SP=0.5` |
| `?FR=X` | **Funding reference**: Uses `EFFRX`, `CASHX`/`TBILL`, or a FRED `DGS*` yield series. | `SPY?L=2&FR=DGS3MO` |
| `_SIM` | **Simulation**: Often used to extend history. | `QQUPSIM` (QQUP-like 2x NDXMEGA history) |

You can combine modifiers: `NDXMEGASIM?L=2&E=0.95` (2x leveraged with 0.95% expense ratio).

Use `QQUPSIM` when the goal is to model the traded QQUP product before its
inception. It already contains 2x daily leverage, DFF financing, the 0.40%
spread, and the 0.95% expense ratio before inception, followed by actual QQUP
adjusted returns. Do not add `?L=2&E=0.95` to `QQUPSIM` unless you intentionally
want to lever the already-levered product again. Use
`NDXMEGASIM?L=2&E=0.95` for a theoretical 2x total-return-index sleeve instead.

For local-engine LETF simulations, `?L` is rate-sensitive. The simulator matches Testfol's trading-day convention: FRED's daily `DFF` rate, `SP`, and `E` are converted with simple `/252`; `SW` defaults to 1.10 and `SP` to `sign(L) × 0.40%`. Weekends and market holidays do not add observations. A flat 4% funding rate is used only when DFF cannot be loaded. Example: `QQQSIM?L=3&E=0.82` applies 3x daily QQQ exposure, subtracts financing on the extra exposure, then subtracts the explicit `0.82%` expense ratio once. Unsupported advanced Testfol modifiers are surfaced in the local-engine log rather than silently ignored.

---

## Dynamic Nasdaq Rotation Tickers

The app supports dynamic pseudo-tickers that expand into historical Nasdaq component sleeves during the local backtest. These are useful when you want a rules-based alternative to hindsight baskets such as "hold today's Mag 7 forever."

### Supported Patterns

| Pattern | Source | Meaning |
| :--- | :--- | :--- |
| `NDX_TOP{N}_ANN` | Historical Nasdaq-100 weights | Each year, select the top N Nasdaq-100 companies using the prior December NDX weights. |
| `NDXMEGA_TOP{N}_ANN` | Local NDXMEGA constituent history | Each annual NDXMEGA reconstruction date, select the top N NDXMEGA companies. |

Examples:
- `NDX_TOP2_ANN`
- `NDX_TOP5_ANN`
- `NDX_TOP8_ANN`
- `NDXMEGA_TOP8_ANN`

The `{N}` value is flexible, so `NDX_TOP4_ANN` and `NDX_TOP5_ANN` work even if they are not saved as presets.

### Query Parameters

Dynamic tickers can use the standard leverage and expense modifiers, plus a weighting modifier:

| Parameter | Example | Meaning |
| :--- | :--- | :--- |
| `L=2` | `NDX_TOP5_ANN?L=2` | Simulate each selected component as a 2x daily-reset instrument. |
| `E=AUTO` | `NDX_TOP5_ANN?L=2&E=AUTO` | Apply per-name individual 2x LETF expense ratios when known. Unknown names use the fallback ER, currently `0.92%`. |
| `W=EQUAL` | `NDX_TOP5_ANN?L=2&E=AUTO&W=EQUAL` | Equal-weight the selected names. This is the default. |
| `W=CAP` | `NDX_TOP2_ANN?L=2&E=AUTO&W=CAP` | Weight selected names by their historical NDX company weights. |
| `W=INV_RANK` | `NDX_TOP2_ANN?L=2&E=AUTO&W=INV_RANK` | Inverse-rank weight the selected names. For top 2, rank 1 receives 2/3 of the sleeve and rank 2 receives 1/3. |
| `W=RANK` | `NDX_TOP5_ANN?L=2&E=AUTO&W=RANK` | Rank-weight selected names linearly, with rank 1 heaviest. |

The `W` parameter controls only the dynamic sleeve allocation. It is removed before the engine creates the underlying component tickers, so a ticker such as:

```text
NDX_TOP2_ANN?L=2&E=AUTO&W=INV_RANK
```

expands into ordinary component tickers such as:

```text
NVDA?L=2&E=0.92
AAPL?L=2&E=0.96
```

### Rules And Data Timing

- `NDX_TOP{N}_ANN` uses prior December Nasdaq-100 weights and becomes effective on January 1 of the following year. When an official Nasdaq-100 membership snapshot is available, the selector filters to those official members before ranking the reconstructed weights.
- `NDXMEGA_TOP{N}_ANN` uses the NDXMEGA reconstruction rows from the local simulation data.
- Dual share classes and historical symbol changes are collapsed at the company level before ranking. For example, `GOOG` and `GOOGL` are treated as one company, historical `FB` membership is matched to `META`, `RIMM` to `BBRY`, and `SYMC` to `GEN`.
- The selected component sleeve is rebalanced when the annual selection changes. Between annual rebalances, holdings drift with returns.
- If the requested start date predates the first available annual selection, the local engine starts the dynamic run at the first available rotation point instead of backfilling a future basket.
- These tickers force the local Shadow Engine because the Testfol API does not know how to expand local dynamic pseudo-tickers.

### Presets Using Dynamic Nasdaq Tickers

Current saved presets include:

| Preset | Dynamic sleeve |
| :--- | :--- |
| `NDXMEGASPLIT Research - Annual NDXMEGA Top 8 30% Sleeve (No ER)` | `NDXMEGA_TOP8_ANN?L=2` |
| `NDXMEGASPLIT Research - Annual NDXMEGA Top 8 30% Sleeve (w/ ERs)` | `NDXMEGA_TOP8_ANN?L=2&E=AUTO` |
| `NDXMEGASPLIT Research - Annual NDX Top 8 30% Sleeve (w/ ERs)` | `NDX_TOP8_ANN?L=2&E=AUTO` |
| `NDX Top 2 Research - Aggressive 40/40/15/5 (w/ ERs)` | `NDX_TOP2_ANN?L=2&E=AUTO&W=CAP` |
| `NDX Top 2 Research - Medium 40/40/20 (w/ ERs)` | `NDX_TOP2_ANN?L=2&E=AUTO&W=INV_RANK` |
| `NDX Top 2 Research - Safe 35/35/30 (w/ ERs)` | `NDX_TOP2_ANN?L=2&E=AUTO&W=INV_RANK` |

### Backtesting Bias Note

Dynamic Nasdaq rotation tickers are intended to reduce hindsight bias. A historical "Mag 7" backtest is anachronistic before the Mag 7 label existed, and a static basket of today's winners can overstate historical performance. A rule such as `NDX_TOP5_ANN?L=2&E=AUTO&W=EQUAL` is more defensible because it uses only historical index membership and weights known at each annual rebalance.

---

## Understanding the Results

### Summary Metrics
- **CAGR**: Compound Annual Growth Rate.
- **Sharpe Ratio**: Risk-adjusted return (using risk-free rate = 0).
- **Max Drawdown**: Largest peak-to-trough decline.
- **Post-Tax Net Equity**: The final liquidation value of your portfolio after paying all loan balances and taxes.

### 4. Chart & Returns Analysis

#### 📈 Chart Tab (Primary)
The **Chart** tab contains the main visualization tools, organized into sub-tabs:

*   **🧮 Margin Calcs**: (Default) Main portfolio chart (Gross vs Net), margin usage, and detailed safety metrics.
*   **📉 200DMA**: 200-Day Moving Average analysis.
*   **📉 150MA**: 150-Day Moving Average & **Stan Weinstein Stage Analysis**.
*   **📜 Cheat Sheet**: Technical Pivot Points and Support/Resistance levels.

#### 🔮 Monte Carlo Analysis (New)
A robust stochastic simulation to stress-test your strategy against thousands of alternative history timelines.
-   **Configuration**:
    -   **Scenarios**: Choose between 100 (fast) to 5,000 (deep) iterations.
    -   **Start Value / Monthly Add**: Override your backtest settings to answer "What if I started with $X?"
-   **Charts**:
    -   **Fan Chart**: Shows the cone of probability (P10 to P90) over 10 years. Toggle "Show Paths" to see individual random outcomes.
    -   **Distribution Histogram**: Visualizes the bell curve of final portfolio values. Uses explicit "B" (Billion) formatting and supports Log/Linear scales.
-   **Metrics**:
    -   **Median CAGR**: The most likely annual return.
    -   **Worst Case DD (P90)**: The 90th percentile worst drawdown.
    -   **Chance of Loss**: Probability of ending with less money than you started.

#### 🏗️ Margin Details (Margin Calcs Tab)
-   **Combined Chart**: Shows Portfolio Value (Gross Assets), Net Equity, and Loan Balance on one chart to visualize leverage dynamics.
-   **Tax Analysis**: Breakdown of annual tax liabilities and the difference between paying with cash vs. margin.

#### 📉 Technical Analysis (Chart Sub-tabs)
The application includes a comprehensive technical analysis suite to help you identify market trends and regimes.

-   **Moving Averages (150MA & 200MA)**: 
    -   Dedicated sub-tabs in the **Chart** section.
    -   **Metrics**: View "Time Under MA", "Longest Period Under", and "Max Depth" to understand historical drawdown behaviors.
-   **Stan Weinstein Stage Analysis (150MA Tab)**:
    -   Estimates the current market phase based on Price vs. 150MA and the Slope of the MA.
    -   **Stage 1 (Basing)** 🟡: Flat MA after a decline. Accumulation/bottoming phase.
    -   **Stage 2 (Advancing)** 🟢: Bull market trend (Price > Rising MA). Includes "Corrections" when price dips below rising MA.
    -   **Stage 3 (Topping)** 🟠: Flat MA after an advance. Distribution/exhaustion phase.
    -   **Stage 4 (Declining)** 🔴: Bear market trend (Price < Falling MA). Includes "Bear Rallies" when price pops above falling MA.
    -   **Methodology**: Uses adaptive volatility-based thresholds, distinguishes Stage 1 from 3 based on prior trend context, and applies 5-day smoothing to reduce daily noise.
-   **Trader's Cheat Sheet**:
    -   Located in the **Cheat Sheet** sub-tab.
    -   Displays standard deviation levels, Fibonacci retracements, and Pivot Points for quick reference.

#### 📊 Returns Analysis Tab
(Formerly "Analysis")
Focuses on periodic performance metrics:
-   **Seasonal Summary**: Monthly return heatmaps.
-   **Annual/Quarterly/Monthly**: Detailed return tables.

#### 🩻 X-Ray Analyzer
The X-Ray engine allows you to peer inside your ETFs to see the actual underlying exposures. It supports a wide range of assets including:

-   **Standard ETFs**: SPY, QQQ, VTI, BND, etc.
-   **Leveraged ETFs**: Automatically decomposes leveraged funds (e.g., `UPRO`) into their 3x underlying components (e.g., 300% SPY).
-   **Simulated Tickers**: Full support for extended history tickers from the `shadow_backtest` engine:
    -   **Markets**: `SPYSIM`, `QQQSIM`, `VTVSIM` (Value), `VUGSIM` (Growth).
    -   **Sectors**: `XLBSIM` (Materials), `XLKSIM` (Tech), `XLFSIM` (Financials), etc.
    -   **Bonds**: `IEFSIM` (7-10y), `SHYSIM` (1-3y), `TLTSIM` (20y+).
    -   **Modifiers**: Works seamlessly with leverage modifiers like `SPYSIM?L=2`.
