# App Architecture Reference

## Overview

The `app/` directory contains the modular Streamlit application for portfolio backtesting with margin simulation and tax analysis. It follows a clean separation of concerns across 5 submodules.

---

## Architecture Diagram

```mermaid
graph TD
    subgraph Entry Point
        MAIN[testfol_charting.py]
    end

    subgraph app/ui
        SB[sidebar.py]
        CF[configuration.py]
        RS[results.py]
        CH[charts.py]
        AE[asset_explorer.py]
    end

    subgraph app/core
        SH[shadow_backtest.py]
        TX[tax_library.py]
        MC[monte_carlo.py]
        CA[calculations.py]
    end

    subgraph app/services
        API[testfol_api.py]
        BE[backend.py]
        PP[price_providers.py]
        DS[data_service.py]
    end

    subgraph app/common
        UT[utils.py]
    end

    subgraph app/reporting
        RG[report_generator.py]
    end

    subgraph External
        TESTFOL[testfol.io API]
        YF[yfinance]
        PG[Polygon.io]
        MEGA[NDXMEGASIM/2SIM CSVs]
    end

    MAIN --> SB
    MAIN --> CF
    MAIN --> RS
    
    SB --> MAIN
    CF --> MAIN
    
    RS --> CH
    RS --> AE
    RS --> RG
    
    MAIN --> API --> TESTFOL
    MAIN --> SH
    
    SH --> TX
    SH --> CA
    SH --> PP
    SH --> MEGA

    PP --> PG
    PP --> YF
    DS --> PP
    
    RS --> MC
    MC --> CA
    
    BE --> API

    subgraph Data Modules
        NDX[data/ndx_simulation]
        XRAY[data/etf_xray]
    end
    
    NDX --> MEGA
    XRAY --> API
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as UI Layer
    participant API as testfol_api
    participant Shadow as Shadow Engine
    participant Tax as Tax Library
    participant Charts as Charts/Results

    User->>UI: Configure Portfolio
    UI->>API: Fetch Backtest Data
    API-->>UI: Portfolio Value Series
    
    UI->>Shadow: Run Shadow Backtest
    Shadow->>Tax: Calculate Tax Lots
    Tax-->>Shadow: Gains by Year
    Shadow-->>UI: Trades, Taxes, Composition
    
    UI->>Charts: Render Results
    Charts-->>User: Interactive Dashboard
```

---

## Module Reference

### `app/core/` - Business Logic

| File | Purpose | Key Functions |
|------|---------|---------------|
| `shadow_backtest.py` | FIFO tax lot tracking, rebalancing simulation | `run_shadow_backtest()` |
| `tax_library.py` | Historical US tax rates (1913-2025), gain calculations | `calculate_gains()`, `get_tax_rate()` |
| `monte_carlo.py` | Bootstrap simulation for risk analysis | `run_simulation()`, `plot_cone()` |
| `calculations.py` | Statistics, returns, Sharpe, **Stage Analysis** | `analyze_stage()`, `generate_stats()` |

### `app/ui/` - User Interface

| File | Purpose | Key Functions |
|------|---------|---------------|
| `sidebar.py` | Date range picker, run button | `render_sidebar()` |
| `configuration.py` | Portfolio config, margin params, benchmarks | `render_config()` |
| `results.py` | Main results orchestration, tabs | `render_results()` |
| `charts.py` | All chart generation (Plotly) | `render_ma_analysis_tab()`, `render_multi_portfolio_chart()` |
| `asset_explorer.py` | Deep-dive analysis for individual assets | `render_asset_explorer()` |

### `app/services/` - External Communication

| File | Purpose | Key Functions |
|------|---------|---------------|
| `testfol_api.py` | API wrapper & Margin Logic | `fetch_backtest()`, `simulate_margin()` |
| `data_service.py` | Complex data sourcing (FRED, splicing, SIM tickers) | `get_fed_funds_rate()`, `fetch_component_data()` |
| `price_providers.py` | Multi-provider price data abstraction (Polygon.io, yfinance) | `get_price_provider()`, `ChainedProvider` |
| `backend.py` | Legacy price utilities | `get_component_prices()` |

### `app/common/` - Utilities

| File | Purpose | Key Functions |
|------|---------|---------------|
| `utils.py` | Documentation rendering, formatting | `render_documentation()` |

### `app/reporting/` - Export

| File | Purpose | Key Functions |
|------|---------|---------------|
| `report_generator.py` | HTML report generation | `generate_html_report()` |

---

## Key Concepts

### Hybrid Engine Mode

The app operates in three modes:

1. **Standard Mode**: Testfol API for backtesting, provider chain for component prices
   - Best for standard tickers (SPY, QQQ, etc.)

2. **Local Mode** (auto-enabled for NDXMEGASIM/threshold rebalancing): Shadow engine with provider chain for prices
   - Loads local CSV for simulated indices
   - Splices post-launch history to the official FRED Nasdaq total-return index
   - Uses QBIG/QQUP only as live quote fallbacks, never as historical index data
   - Uses local Shadow Engine for tax calculations

3. **Failover Mode** (auto-enabled when Testfol is down): Same as Local Mode, triggered automatically with user notification

### Tax Lot System (FIFO)

```mermaid
graph LR
    BUY1[Buy 100 @ $10<br>Lot 1] --> HOLD
    BUY2[Buy 50 @ $15<br>Lot 2] --> HOLD
    HOLD --> SELL[Sell 75 shares]
    SELL --> LOT1[Lot 1: Sell 75<br>Basis: $750]
    LOT1 --> REMAIN[Lot 1: 25 remaining<br>Lot 2: 50 remaining]
```

### Chart Types

| Chart | Location | Description |
|-------|----------|-------------|
| Portfolio Performance | `charts.py` | Main value chart with margin overlay |
| Composition Stacked | `charts.py` | Asset allocation over time |
| Monthly Returns Heatmap | `charts.py` | Calendar heatmap of returns |
| Sankey Diagram | `charts.py` | Rebalancing flows visualization |
| Monte Carlo Cone | `monte_carlo.py` | Future probability distribution |

---

## File Sizes

| File | Size | Complexity |
|------|------|------------|
| `charts.py` | 80KB | High - All visualization logic |
| `results.py` | 58KB | High - Main orchestration |
| `shadow_backtest.py` | 37KB | High - Tax lot engine |
| `tax_library.py` | 32KB | Medium - Tax rate tables |
| `configuration.py` | 16KB | Medium - UI forms |
| `asset_explorer.py` | 13KB | Medium - Asset drill-down |
| `monte_carlo.py` | 12KB | Medium - Simulation |
| `report_generator.py` | 9KB | Low - HTML export |
| `calculations.py` | 8KB | Low - Math utilities |
| `testfol_api.py` | 8KB | Low - API wrapper |
| `price_providers.py` | 7KB | Low - Provider abstraction |
| `backend.py` | 8KB | Low - Data utilities |
| `sidebar.py` | 4KB | Low - Simple UI |
| `utils.py` | 3KB | Low - Helpers |
