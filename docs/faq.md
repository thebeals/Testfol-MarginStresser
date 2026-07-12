# FAQ & Troubleshooting

## Common Issues

### "Why is my Start Date capped?"
**Cause**: The simulation start date is strictly limited by the **earliest common data availability** for the assets in your portfolio.
- **Explanation**: The local tax engine requires real daily price data to track tax lots. It cannot use synthetic data for this purpose. Use the inception date of your *youngest* asset as a guide.
- **Example**: If your portfolio contains an ETF that launched in 2002 (e.g., TLT), the entire simulation cannot start before that date.

### "Weights must sum to 100%"
**Cause**: The total percentage in your allocation table does not equal exactly 100.00%.
**Fix**: Adjust the "Weight %" column. Ensure no hidden rows have residual values.

### API Request Failed
**Cause**: Connection issue with Testfol.io or invalid ticker symbol.
**What happens**: The app automatically falls back to the **local Shadow Engine** using price data from Polygon.io or yfinance. A warning banner will appear indicating which portfolios were computed locally.
**Fix**: If you want to use the Testfol API specifically, verify your internet connection and that testfol.io is reachable. Otherwise, the local fallback provides equivalent results.

---

## Technical Questions

### How accurate is the tax simulation?
**Very High Fidelity (Mechanically)**.
-   The app tracks every single share purchase and sale (FIFO Tax Lots).
-   It distinguishes Short-Term vs. Long-Term gains daily.
-   It applies historical tax brackets appropriate for each year.

**However**, it is **NOT** a replacement for a CPA. It does not account for:
-   Wash Sale rules (30-day window).
-   Your personal deductions or credits.
-   State-specific tax brackets (it uses a flat state rate).

### Can I simulate Short Selling?
**No**. The simulator currently supports "Long Only" strategies with margin leverage. Short selling involves borrowing shares (not cash) and has different margin maintenance rules not yet modeled here.

### Does `?E` include the full cost of a leveraged ETF?
**No.** `?E` represents the explicit fund operating expense ratio. For local synthetic leverage such as `QQQSIM?L=3&E=0.82` or `NDXMEGASIM?L=2&E=0.95`, the app also models financing on the borrowed/synthetic exposure.

The local engine currently uses:

```text
DFF / 252 + sign(L) × 0.40% / 252 implementation spread
```

for every synthetic `?L` ticker, scaled by `SW × (L - 1)`, with a flat 4% fallback only if DFF cannot be loaded. The explicit `E / 252` expense is deducted separately and only once. This synthetic convention is intentionally different from the Actual/360 calendar-day accrual used for real USD margin debt.

The local engine also supports Testfol-style `UE`, `CU`/`CL`, `SW`, `SP`, and
`FR` modifiers. EFFRX, CASHX/TBILL, and FRED `DGS*` funding references use one
simple `/252` charge per XNYS trading session. Advanced modifiers that the
local engine cannot reproduce are explicitly listed in its warning log.

Real leveraged ETF tickers such as `TQQQ`, `QLD`, `SSO`, and `QQUP` use their actual price history, so those embedded financing/tracking costs are already reflected in the return series.

### Why can local results differ from Testfol API results?
Some portfolios route through Testfol's API, while local-only tickers such as `NDXMEGASIM`, `NDX30SIM`, threshold rebalancing, no-rebalance mode, and dynamic Nasdaq rotation tickers route through the local Shadow Engine.

For standard API-supported expressions such as `QQQSIM?L=3`, the API may use its own leverage and financing assumptions. For local synthetic leverage, this app now uses historical Fed Funds plus the default implementation spread. When comparing local-only NDXMEGA strategies to API-supported TQQQ/GLD strategies, check whether the comparison is mixed-engine or forced-local.

### Why is there a difference between "Pay with Margin" and "Pay from Cash"?
-   **Pay with Margin**: You keep your assets compounding but accumulate a compounding debt liability. Good in bull markets.
-   **Pay from Cash**: You stay debt-free (regarding taxes) but interrupt the compound growth of your assets by selling them. Often safer but lower total return.

### How do I set up Polygon.io for better price data?
**Optional but recommended.** Polygon.io provides more complete historical data (including delisted tickers) than yfinance.
1.  Sign up at [polygon.io](https://polygon.io) (free tier available).
2.  Set the environment variable: `export POLYGON_API_KEY=your_key_here`
3.  The app will automatically prefer Polygon.io over yfinance for price data.
4.  If no key is set, yfinance is used as the default (free, no setup needed).

### What happens if Testfol.io goes down?
The app is designed to be resilient. If the Testfol API is unreachable:
-   Backtests are automatically routed to the **local Shadow Engine**.
-   Price data is fetched from **Polygon.io** (if configured) or **yfinance**.
-   A warning banner is displayed showing which portfolios used the fallback.
-   Results may differ slightly from Testfol due to minor differences in price data sources.
