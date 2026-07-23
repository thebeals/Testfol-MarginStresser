from pydantic import BaseModel, Field
from typing import Optional


class RebalanceConfig(BaseModel):
    mode: str = "Standard"
    freq: str = "Yearly"
    month: int = 1
    day: int = 1
    threshold_pct: float = 5.0
    compare_std: bool = False


class CashflowConfig(BaseModel):
    start_val: float = 10000.0
    amount: float = 0.0
    freq: str = "Monthly"
    invest_div: bool = True
    pay_down_margin: bool = False


class DcaConfig(BaseModel):
    mode: str = "Proportional"
    target_ticker: str = ""


class TaxConfig(BaseModel):
    """Passthrough for tax_config dict — kept flexible."""
    st_rate: float = 0.0
    lt_rate: float = 0.0
    nii_rate: float = 0.0
    state_rate: float = 0.0


class PortfolioConfig(BaseModel):
    name: str = "Portfolio"
    allocation: dict[str, float]
    maint_pcts: dict[str, float] = Field(default_factory=dict)
    pm_maint_pcts: dict[str, float] = Field(default_factory=dict)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    dca: DcaConfig = Field(default_factory=DcaConfig)


class BacktestRequest(BaseModel):
    portfolio: PortfolioConfig
    start_date: str
    end_date: str
    cashflow: CashflowConfig = Field(default_factory=CashflowConfig)
    tax_config: dict = Field(default_factory=dict)
    bearer_token: Optional[str] = None


class MultiBacktestRequest(BaseModel):
    portfolios: list[PortfolioConfig]
    start_date: str
    end_date: str
    cashflow: CashflowConfig = Field(default_factory=CashflowConfig)
    tax_config: dict = Field(default_factory=dict)
    bearer_token: Optional[str] = None
    pm_config: Optional[dict] = None


# --- Response models use dicts for flexibility (DataFrames serialized as JSON strings) ---

class BacktestResult(BaseModel):
    """Single portfolio result. DataFrames/Series are serialized via pandas to_json(orient='split')."""
    name: str
    series_json: Optional[str] = None
    stats: dict = Field(default_factory=dict)
    twr_series_json: Optional[str] = None
    daily_returns_df_json: Optional[str] = None
    trades_df_json: Optional[str] = None
    pl_by_year_json: Optional[str] = None
    composition_df_json: Optional[str] = None
    unrealized_pl_df_json: Optional[str] = None
    component_prices_json: Optional[str] = None
    allocation: dict[str, float] = Field(default_factory=dict)
    logs: list = Field(default_factory=list)
    raw_response: dict = Field(default_factory=dict)
    is_local: bool = False
    api_failover: bool = False
    start_val: float = 10000.0
    sim_range: str = ""
    shadow_range: str = ""
    effective_start_date: Optional[str] = None
    wmaint: float = 0.25
    wmaint_pm: float = 0.0
    pm_blocked_dates: list = Field(default_factory=list)


class MultiBacktestResponse(BaseModel):
    results: list[BacktestResult]
    bench_results: list[BacktestResult] = Field(default_factory=list)
    common_start: Optional[str] = None
