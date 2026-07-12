"""
Shared backtest orchestration logic.

Pure Python module — no Streamlit, no FastAPI, no Pydantic imports.
All params are plain types (dict, str, float, bool).
Returns raw dicts containing pandas objects.

Dependency injection: callers can pass cached versions of fetch_backtest
and run_shadow_backtest via the `fetch_backtest_fn` / `run_shadow_fn` params.
"""

import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from app.common.constants import DcaMode, Freq, RebalMode, Tickers
from app.core import calculations
from app.core.shadow_backtest import run_shadow_backtest as _default_shadow
from app.services.data_service import clip_component_data_to_synced_end, fetch_component_data
from app.services.ndx_mega_rotation import (
    build_dynamic_allocation_plan,
    expand_dynamic_ticker_universe,
)
from app.services.testfol_api import fetch_backtest as _default_fetch

logger = logging.getLogger(__name__)
MAX_ORCHESTRATOR_WORKERS = 8


def _return_anchor_start(start_date) -> str:
    """Fetch enough lead-in data to retain one prior trading observation."""
    return (pd.Timestamp(start_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")


def calc_rebal_offset(reb: dict, r_freq: str) -> int:
    """Calculate rebalance offset from custom rebalance config."""
    r_mode = reb.get("mode", RebalMode.STANDARD)
    if r_mode != RebalMode.CUSTOM:
        return 0
    r_month = reb.get("month", 1)
    r_day = reb.get("day", 1)
    try:
        if r_freq == Freq.YEARLY:
            end_of_year = pd.Timestamp("2024-12-31")
            target_date = pd.Timestamp(f"2024-{r_month}-{r_day}")
            days_remaining = (end_of_year - target_date).days
            return max(0, int(days_remaining * (252.0 / 366.0)))
        else:
            days_remaining = 31 - r_day
            return int(days_remaining * (21.0 / 31.0))
    except Exception:
        return 0


def _base_tickers(tickers) -> list[str]:
    """Return unique base tickers while preserving order."""
    return list(dict.fromkeys(str(ticker).split("?")[0] for ticker in tickers))


def _orchestrator_worker_count(task_count: int) -> int:
    """Cap parallelism to avoid oversubscribing the process."""
    if task_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 4
    return min(task_count, max(2, min(MAX_ORCHESTRATOR_WORKERS, cpu_count)))


def _slice_prefetched_component_prices(
    prefetched_component_prices: pd.DataFrame | None,
    tickers,
    start_date,
    end_date,
) -> pd.DataFrame | None:
    """Return the requested component subset or None if the shared fetch is incomplete."""
    if prefetched_component_prices is None or prefetched_component_prices.empty:
        return None

    expected_cols = _base_tickers(tickers)
    missing_cols = [ticker for ticker in expected_cols if ticker not in prefetched_component_prices.columns]
    if missing_cols:
        return None

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    sliced = prefetched_component_prices.loc[:, expected_cols].sort_index()
    prior = sliced.loc[sliced.index < start_ts].tail(1)
    sliced = pd.concat([prior, sliced.loc[start_ts:end_ts]])
    return clip_component_data_to_synced_end(sliced, expected_cols)


def _prefetch_component_universe(
    tickers,
    start_date,
    end_date,
    *,
    purpose: str,
) -> pd.DataFrame | None:
    """Best-effort shared component fetch for multi-portfolio runs."""
    unique_bases = _base_tickers(tickers)
    if not unique_bases:
        return None

    try:
        expanded_tickers = expand_dynamic_ticker_universe(tickers, start_date, end_date)
        try:
            return fetch_component_data(
                expanded_tickers,
                _return_anchor_start(start_date),
                end_date,
                sync_end=False,
            )
        except TypeError as exc:
            if "sync_end" not in str(exc):
                raise
            return fetch_component_data(
                expanded_tickers, _return_anchor_start(start_date), end_date
            )
    except Exception as exc:
        logger.warning("Failed shared component fetch for %s: %s", purpose, exc)
        return None


def _normalize_dca_config(dca_config: dict | None) -> dict:
    cfg = dca_config or {}
    mode = str(cfg.get("mode", DcaMode.PROPORTIONAL)).strip()
    if mode.lower().replace("_", " ") not in {"single", "single asset", "target", "targeted", "target asset"}:
        mode = DcaMode.PROPORTIONAL
    else:
        mode = DcaMode.SINGLE_ASSET
    return {
        "mode": mode,
        "target_ticker": str(cfg.get("target_ticker") or cfg.get("target") or "").strip(),
    }


def _uses_single_asset_dca(dca_config: dict | None, cashflow_amount: float, pay_down_margin: bool) -> bool:
    cfg = _normalize_dca_config(dca_config)
    return (
        cashflow_amount > 0
        and not pay_down_margin
        and cfg["mode"] == DcaMode.SINGLE_ASSET
        and bool(cfg["target_ticker"])
    )


def _run_pass1_portfolio(
    index: int,
    portfolio: dict,
    *,
    start_date: str,
    end_date: str,
    start_val: float,
    cashflow_amount: float,
    cashflow_freq: str,
    invest_div: bool,
    pay_down_margin: bool,
    tax_config: dict,
    bearer_token: str | None,
    fetch_fn,
    shadow_fn,
    pm_config: dict | None,
    prefetched_component_prices: pd.DataFrame | None,
):
    """Execute one pass-1 portfolio and its optional comparison benchmark."""
    raw = run_single_backtest(
        allocation=portfolio["allocation"],
        maint_pcts=portfolio.get("maint_pcts", {}),
        rebalance=portfolio.get("rebalance", {}),
        start_date=start_date,
        end_date=end_date,
        start_val=start_val,
        cashflow_amount=cashflow_amount,
        cashflow_freq=cashflow_freq,
        invest_div=invest_div,
        pay_down_margin=pay_down_margin,
        tax_config=tax_config,
        bearer_token=bearer_token,
        name=portfolio.get("name", "Portfolio"),
        fetch_backtest_fn=fetch_fn,
        run_shadow_fn=shadow_fn,
        pm_maint_pcts=portfolio.get("pm_maint_pcts"),
        pm_config=pm_config,
        prefetched_component_prices=prefetched_component_prices,
        dca_config=portfolio.get("dca"),
    )

    bench_series = None
    reb = portfolio.get("rebalance", {})
    dca_config = _normalize_dca_config(portfolio.get("dca"))
    if reb.get("compare_std", False) and reb.get("mode") == "Custom":
        try:
            if _uses_single_asset_dca(dca_config, cashflow_amount, pay_down_margin):
                bench_raw = run_single_backtest(
                    allocation=portfolio["allocation"],
                    maint_pcts=portfolio.get("maint_pcts", {}),
                    rebalance={"mode": RebalMode.STANDARD, "freq": Freq.YEARLY},
                    start_date=start_date,
                    end_date=end_date,
                    start_val=start_val,
                    cashflow_amount=cashflow_amount,
                    cashflow_freq=cashflow_freq,
                    invest_div=invest_div,
                    pay_down_margin=pay_down_margin,
                    tax_config=tax_config,
                    bearer_token=bearer_token,
                    name=f"{portfolio.get('name')} (Standard)",
                    fetch_backtest_fn=fetch_fn,
                    run_shadow_fn=shadow_fn,
                    pm_maint_pcts=portfolio.get("pm_maint_pcts"),
                    pm_config=pm_config,
                    prefetched_component_prices=prefetched_component_prices,
                    dca_config=dca_config,
                )
                bench_series = bench_raw.get("series")
            else:
                bench_series, _, _ = fetch_fn(
                    start_date=start_date,
                    end_date=end_date,
                    start_val=start_val,
                    cashflow=0.0 if pay_down_margin else cashflow_amount,
                    cashfreq=cashflow_freq,
                    rolling=60,
                    invest_div=invest_div,
                    rebalance="Yearly",
                    allocation=portfolio["allocation"],
                    return_raw=False,
                    bearer_token=bearer_token,
                )
            if bench_series is not None:
                bench_series.name = f"{portfolio.get('name')} (Standard)"
        except Exception as exc:
            logger.warning("Failed standard comparison: %s", exc)

    return index, raw, bench_series


def _rerun_result_for_common_start(
    index: int,
    res: dict,
    *,
    common_start,
    end_date: str,
    global_start_val: float,
    cashflow_amount: float,
    cashflow_freq: str,
    invest_div: bool,
    pay_down_margin: bool,
    tax_config: dict,
    bearer_token: str | None,
    fetch_fn,
    shadow_fn,
    pm_config: dict | None,
    pm_draw_start_date,
    pm_retirement_date,
    prefetched_component_prices: pd.DataFrame | None,
):
    """Apply common-start alignment logic for a single result."""
    series = res.get("series")
    if series is None or series.empty:
        return index, res
    if series.index[0] >= common_start - pd.Timedelta(days=3):
        return index, res

    alloc_map = res.get("_engine_allocation", res["allocation"])
    dynamic_schedule = res.get("_dynamic_allocation_schedule")
    dca_config = _normalize_dca_config(res.get("_dca_config"))
    reb = res["_reb"]
    r_mode = res["_r_mode"]
    r_freq = reb.get("freq", "Yearly")
    r_threshold = reb.get("threshold_pct", 5.0)
    uses_threshold = r_mode in (RebalMode.THRESHOLD, RebalMode.THRESHOLD_CALENDAR)

    if not res.get("is_local", False):
        rebal_offset = calc_rebal_offset(reb, r_freq)
        try:
            new_series, new_stats, new_extra = fetch_fn(
                start_date=common_start.strftime("%Y-%m-%d"),
                end_date=end_date,
                start_val=global_start_val,
                cashflow=0.0 if pay_down_margin else cashflow_amount,
                cashfreq=cashflow_freq,
                rolling=60,
                invest_div=invest_div,
                rebalance=r_freq,
                rebalance_offset=rebal_offset,
                allocation=alloc_map,
                return_raw=False,
                include_raw=True,
                bearer_token=bearer_token,
            )
            if not new_series.empty:
                res["series"] = new_series
                res["port_series"] = new_series
                res["original_api_stats"] = res.get("stats", {})
                res["stats"] = new_stats
                res["raw_response"] = new_extra
                logger.debug(
                    "Re-fetched %s from %s - CAGR: %s",
                    res["name"],
                    common_start.date(),
                    new_stats.get("cagr"),
                )

                try:
                    shadow_cf = 0.0 if pay_down_margin else cashflow_amount
                    _pm_cfg_p2 = pm_config or {}
                    new_trades, new_pl, new_comp, new_unrealized, new_logs, _, new_twr, *_p2_rest = shadow_fn(
                        allocation=alloc_map,
                        start_val=global_start_val,
                        start_date=common_start.strftime("%Y-%m-%d"),
                        end_date=end_date,
                        api_port_series=new_series,
                        rebalance_freq="Custom" if r_mode == "Custom" else r_freq,
                        cashflow=shadow_cf,
                        cashflow_freq=cashflow_freq,
                        invest_dividends=invest_div,
                        pay_down_margin=pay_down_margin,
                        tax_config=tax_config,
                        custom_rebal_config=reb if r_mode == "Custom" else {},
                        rebalance_month=reb.get("month", 1),
                        rebalance_day=reb.get("day", 1),
                        custom_freq=reb.get("freq", "Yearly"),
                        threshold_pct=r_threshold,
                        pm_buy_block=_pm_cfg_p2.get("pm_buy_block", False),
                        pm_buy_block_threshold=_pm_cfg_p2.get("pm_buy_block_threshold", 100000.0),
                        starting_loan=_pm_cfg_p2.get("starting_loan", 0.0),
                        margin_rate_annual=_pm_cfg_p2.get("margin_rate_annual", 8.0),
                        draw_monthly=_pm_cfg_p2.get("draw_monthly", 0.0),
                        draw_start_date=pm_draw_start_date,
                        draw_monthly_retirement=_pm_cfg_p2.get("draw_monthly_retirement", 0.0),
                        retirement_date=pm_retirement_date,
                        dca_in_retirement=_pm_cfg_p2.get("dca_in_retirement", True),
                        loan_repayment=_pm_cfg_p2.get("cashflow_for_loan", 0.0) if pay_down_margin else 0.0,
                        loan_repayment_freq=_pm_cfg_p2.get("cashflow_freq", "Monthly"),
                        dynamic_allocation_schedule=dynamic_schedule,
                        dca_mode=dca_config["mode"],
                        dca_target_ticker=dca_config["target_ticker"],
                    )
                    res["trades_df"] = new_trades
                    res["trades"] = new_trades
                    res["pl_by_year"] = new_pl
                    res["composition_df"] = new_comp
                    res["composition"] = new_comp
                    res["unrealized_pl_df"] = new_unrealized
                    res["logs"] = new_logs
                    res["twr_series"] = new_twr
                    if _p2_rest:
                        res["pm_blocked_dates"] = list(_p2_rest[0]) if _p2_rest[0] else []

                    new_extra = res.get("raw_response")
                    if new_extra and "daily_returns" in new_extra:
                        d_rets = new_extra["daily_returns"]
                        if d_rets:
                            try:
                                df_tmp = pd.DataFrame(d_rets, columns=["Date", "Pct", "Val"])
                                df_tmp["Date"] = pd.to_datetime(df_tmp["Date"])
                                df_tmp = df_tmp.set_index("Date").sort_index()
                                api_twr = (1 + df_tmp["Pct"] / 100.0).cumprod()
                                api_twr.name = "TWR (API)"
                                res["twr_series"] = api_twr
                            except Exception:
                                pass
                    res["shadow_range"] = f"{common_start.date()} to {end_date}"
                except Exception as shadow_exc:
                    logger.warning("Failed to re-run shadow for %s: %s", res["name"], shadow_exc)

        except Exception as exc:
            logger.warning("Failed to re-fetch %s: %s", res["name"], exc)
            twr = res.get("twr_series")
            if twr is not None and not twr.empty:
                twr_slice = twr[twr.index >= common_start]
                if not twr_slice.empty:
                    scale = twr_slice / twr_slice.iloc[0]
                    res["series"] = scale * global_start_val
                    res["port_series"] = res["series"]
                    res["stats"] = calculations.generate_stats(res["series"])
        return index, res

    twr = res.get("twr_series")
    if twr is not None and not twr.empty:
        twr_slice = twr[twr.index >= common_start]
        if not twr_slice.empty:
            new_series = (twr_slice / twr_slice.iloc[0]) * global_start_val
            res["series"] = new_series
            res["port_series"] = new_series
            res["stats"] = calculations.generate_stats(new_series)

            try:
                prices_df_new = _slice_prefetched_component_prices(
                    prefetched_component_prices,
                    alloc_map.keys(),
                    common_start.strftime("%Y-%m-%d"),
                    end_date,
                )
                if prices_df_new is None:
                    prices_df_new = fetch_component_data(
                        list(alloc_map.keys()),
                        _return_anchor_start(common_start),
                        end_date,
                    )
                shadow_cf = 0.0 if pay_down_margin else cashflow_amount
                _pm_cfg2 = pm_config or {}
                new_trades, new_pl, new_comp, new_unrealized, new_logs, new_port, new_twr, *_p2_rest2 = shadow_fn(
                    allocation=alloc_map,
                    start_val=global_start_val,
                    start_date=common_start.strftime("%Y-%m-%d"),
                    end_date=end_date,
                    api_port_series=None,
                    rebalance_freq="None" if r_mode == RebalMode.NONE else (r_mode if uses_threshold else reb.get("freq", "Yearly")),
                    cashflow=shadow_cf,
                    cashflow_freq=cashflow_freq,
                    invest_dividends=invest_div,
                    pay_down_margin=pay_down_margin,
                    tax_config=tax_config,
                    custom_rebal_config=reb if r_mode == "Custom" else {},
                    prices_df=prices_df_new,
                    rebalance_month=reb.get("month", 1),
                    rebalance_day=reb.get("day", 1),
                    custom_freq=reb.get("freq", "Yearly"),
                    threshold_pct=r_threshold,
                    pm_buy_block=_pm_cfg2.get("pm_buy_block", False),
                    pm_buy_block_threshold=_pm_cfg2.get("pm_buy_block_threshold", 100000.0),
                    starting_loan=_pm_cfg2.get("starting_loan", 0.0),
                    margin_rate_annual=_pm_cfg2.get("margin_rate_annual", 8.0),
                    draw_monthly=_pm_cfg2.get("draw_monthly", 0.0),
                    draw_start_date=pm_draw_start_date,
                    draw_monthly_retirement=_pm_cfg2.get("draw_monthly_retirement", 0.0),
                    retirement_date=pm_retirement_date,
                    dca_in_retirement=_pm_cfg2.get("dca_in_retirement", True),
                    loan_repayment=_pm_cfg2.get("cashflow_for_loan", 0.0) if pay_down_margin else 0.0,
                    loan_repayment_freq=_pm_cfg2.get("cashflow_freq", "Monthly"),
                    dynamic_allocation_schedule=dynamic_schedule,
                    dca_mode=dca_config["mode"],
                    dca_target_ticker=dca_config["target_ticker"],
                )
                res["trades_df"] = new_trades
                res["trades"] = new_trades
                res["pl_by_year"] = new_pl
                res["composition_df"] = new_comp
                res["composition"] = new_comp
                res["unrealized_pl_df"] = new_unrealized
                res["logs"] = new_logs
                res["twr_series"] = new_twr
                if _p2_rest2:
                    res["pm_blocked_dates"] = list(_p2_rest2[0]) if _p2_rest2[0] else []
                res["shadow_range"] = f"{common_start.date()} to {end_date}"
                if new_port is not None and not new_port.empty:
                    res["series"] = new_port
                    res["port_series"] = new_port
                elif new_twr is not None and not new_twr.empty:
                    synced = (new_twr / new_twr.iloc[0]) * global_start_val
                    res["series"] = synced
                    res["port_series"] = synced
                res["stats"] = calculations.generate_stats(
                    new_twr if new_twr is not None and not new_twr.empty else new_series
                )
            except Exception as shadow_exc:
                logger.warning("Failed to re-run shadow for local %s: %s", res["name"], shadow_exc)

    return index, res


def run_single_backtest(
    allocation: dict,
    maint_pcts: dict,
    rebalance: dict,
    start_date: str,
    end_date: str,
    start_val: float,
    cashflow_amount: float,
    cashflow_freq: str,
    invest_div: bool,
    pay_down_margin: bool,
    tax_config: dict,
    bearer_token: str | None,
    name: str = "Portfolio",
    fetch_backtest_fn=None,
    run_shadow_fn=None,
    pm_maint_pcts: dict | None = None,
    pm_config: dict | None = None,
    prefetched_component_prices: pd.DataFrame | None = None,
    dca_config: dict | None = None,
) -> dict:
    """
    Run a single portfolio backtest (API or local engine).

    Returns a raw dict with all result fields (not yet serialized).
    Uses dependency injection for the two expensive functions so callers
    can pass @st.cache_data-wrapped versions.
    """
    fetch_fn = fetch_backtest_fn or _default_fetch
    shadow_fn = run_shadow_fn or _default_shadow
    dca_config = _normalize_dca_config(dca_config)

    original_allocation = allocation
    dynamic_plan = build_dynamic_allocation_plan(
        allocation,
        maint_pcts,
        pm_maint_pcts,
        start_date,
        end_date,
    )
    alloc_map = dynamic_plan.engine_allocation
    engine_maint_pcts = dynamic_plan.maint_pcts
    engine_pm_maint_pcts = dynamic_plan.pm_maint_pcts
    shared_prices_df = _slice_prefetched_component_prices(
        prefetched_component_prices,
        alloc_map.keys(),
        start_date,
        end_date,
    )

    # Weighted maintenance
    maintenance_allocation = alloc_map
    if dynamic_plan.dynamic_schedule:
        maintenance_allocation = next(iter(sorted(dynamic_plan.dynamic_schedule.items())))[1]
        start_ts = pd.Timestamp(start_date)
        for effective_date, target_allocation in sorted(dynamic_plan.dynamic_schedule.items()):
            if effective_date <= start_ts:
                maintenance_allocation = target_allocation
            else:
                break

    total_w = sum(maintenance_allocation.values())
    d_maint = 25.0
    current_wmaint = 0.0
    if total_w > 0:
        for ticker, weight in maintenance_allocation.items():
            m = engine_maint_pcts.get(ticker.split("?")[0], d_maint)
            current_wmaint += (weight / total_w) * (m / 100)
    else:
        current_wmaint = d_maint / 100.0

    # PM weighted maintenance
    current_wmaint_pm = 0.0
    if engine_pm_maint_pcts and total_w > 0:
        for ticker, weight in maintenance_allocation.items():
            m_pm = engine_pm_maint_pcts.get(ticker.split("?")[0], 0.0)
            if m_pm > 0:
                current_wmaint_pm += (weight / total_w) * (m_pm / 100)

    # PM buy block config
    _pm_cfg = pm_config or {}
    pm_buy_block = _pm_cfg.get("pm_buy_block", False)
    pm_buy_block_threshold = _pm_cfg.get("pm_buy_block_threshold", 100000.0)
    pm_draw_monthly = _pm_cfg.get("draw_monthly", 0.0)
    pm_draw_monthly_retirement = _pm_cfg.get("draw_monthly_retirement", 0.0)
    pm_dca_in_retirement = _pm_cfg.get("dca_in_retirement", True)
    _raw_retirement_date = _pm_cfg.get("retirement_date", None)
    _raw_draw_start = _pm_cfg.get("draw_start_date", None)
    # Clamp draw_start_date to backtest range
    _bt_start = pd.Timestamp(start_date).date() if isinstance(start_date, str) else start_date
    _bt_end = pd.Timestamp(end_date).date() if isinstance(end_date, str) else end_date
    if _raw_draw_start is not None:
        if isinstance(_raw_draw_start, str):
            _raw_draw_start = pd.Timestamp(_raw_draw_start).date()
        elif isinstance(_raw_draw_start, datetime.datetime):
            _raw_draw_start = _raw_draw_start.date()
        pm_draw_start_date = max(_raw_draw_start, _bt_start)
        pm_draw_start_date = min(pm_draw_start_date, _bt_end)
    else:
        pm_draw_start_date = _bt_start
    # Clamp retirement_date to backtest range
    if _raw_retirement_date is not None:
        if isinstance(_raw_retirement_date, str):
            _raw_retirement_date = pd.Timestamp(_raw_retirement_date).date()
        elif isinstance(_raw_retirement_date, datetime.datetime):
            _raw_retirement_date = _raw_retirement_date.date()
        pm_retirement_date = max(_raw_retirement_date, _bt_start)
        pm_retirement_date = min(pm_retirement_date, _bt_end)
    else:
        pm_retirement_date = None
    pm_margin_rate = _pm_cfg.get("margin_rate_annual", 8.0)
    pm_cf_for_loan = _pm_cfg.get("cashflow_for_loan", 0.0)
    pm_cf_freq = _pm_cfg.get("cashflow_freq", "Monthly")

    _orch_log = logging.getLogger("orchestrator")
    _orch_log.info("Orchestrator draw config: draw=$%.0f/mo ret_draw=$%.0f/mo draw_start=%s ret_date=%s dca_in_ret=%s",
                   pm_draw_monthly, pm_draw_monthly_retirement, pm_draw_start_date, pm_retirement_date, pm_dca_in_retirement)

    # Rebalance
    reb = rebalance
    r_mode = reb.get("mode", RebalMode.STANDARD)
    r_freq = reb.get("freq", Freq.YEARLY)
    r_threshold = reb.get("threshold_pct", 5.0)

    # Determine engine
    has_ndxmega = dynamic_plan.has_dynamic or any(
        (
            Tickers.NDXMEGASIM in t
            or Tickers.NDXMEGA2SIM in t
            or Tickers.QQUPSIM in t
            or Tickers.NDX30SIM in t
        )
        for t in alloc_map
    )
    uses_threshold = r_mode in (RebalMode.THRESHOLD, RebalMode.THRESHOLD_CALENDAR)
    no_rebal = r_mode == RebalMode.NONE
    use_local_engine = has_ndxmega or uses_threshold or no_rebal or _uses_single_asset_dca(dca_config, cashflow_amount, pay_down_margin)

    # Cashflow settings
    bt_cashflow = 0.0 if pay_down_margin else cashflow_amount
    shadow_cashflow = 0.0 if pay_down_margin else cashflow_amount

    port_series = pd.Series(dtype=float)
    stats: dict = {}
    trades_df = pd.DataFrame()
    extra_data: dict = {}
    df_rets = None
    twr_series = None
    pl_by_year = pd.DataFrame()
    composition_df = pd.DataFrame()
    unrealized_pl_df = pd.DataFrame()
    logs: list = []
    prices_df = pd.DataFrame()
    pm_blocked_dates: list = []
    api_failover = False

    if not use_local_engine:
        # --- API Path (with automatic failover to local engine) ---
        try:
            rebal_offset = calc_rebal_offset(reb, r_freq)

            port_series, stats_api, extra_data = fetch_fn(
                start_date=start_date,
                end_date=end_date,
                start_val=max(1.0, start_val),
                cashflow=bt_cashflow,
                cashfreq=cashflow_freq,
                rolling=60,
                invest_div=invest_div,
                rebalance=r_freq,
                rebalance_offset=rebal_offset,
                allocation=alloc_map,
                return_raw=False,
                include_raw=True,
                bearer_token=bearer_token,
            )
            stats = stats_api
            logger.debug(f"API: Tickers={list(alloc_map.keys())} | API Stats CAGR={stats.get('cagr')}")

            # Extract TWR Series from API
            api_twr_series = None
            if extra_data and "daily_returns" in extra_data:
                d_rets = extra_data["daily_returns"]
                if d_rets:
                    try:
                        df_rets = pd.DataFrame(d_rets, columns=["Date", "Pct", "Val"])
                        df_rets["Date"] = pd.to_datetime(df_rets["Date"])
                        df_rets = df_rets.set_index("Date").sort_index()
                        df_rets["Factor"] = 1 + (df_rets["Pct"] / 100.0)
                        api_twr_series = df_rets["Factor"].cumprod()
                        api_twr_series.name = "TWR (API)"
                    except Exception as e:
                        logger.warning(f"Failed to build API TWR: {e}")

            # Fetch component prices for cheat sheet
            if shared_prices_df is not None:
                prices_df = shared_prices_df
            else:
                try:
                    prices_df = fetch_component_data(list(alloc_map.keys()), start_date, end_date)
                except Exception as e:
                    logger.warning(f"Failed to fetch component prices: {e}")
                    prices_df = pd.DataFrame()

            # Run shadow backtest (tax tracking)
            if not port_series.empty:
                trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, _, twr_series, *_ = shadow_fn(
                    allocation=alloc_map,
                    start_val=start_val,
                    start_date=start_date,
                    end_date=end_date,
                    api_port_series=port_series,
                    rebalance_freq="Custom" if r_mode == "Custom" else r_freq,
                    cashflow=shadow_cashflow,
                    cashflow_freq=cashflow_freq,
                    invest_dividends=invest_div,
                    pay_down_margin=pay_down_margin,
                    tax_config=tax_config,
                    custom_rebal_config=reb if r_mode == "Custom" else {},
                    rebalance_month=reb.get("month", 1),
                    rebalance_day=reb.get("day", 1),
                    custom_freq=reb.get("freq", "Yearly"),
                    pm_buy_block=pm_buy_block,
                    pm_buy_block_threshold=pm_buy_block_threshold,
                    starting_loan=_pm_cfg.get("starting_loan", 0.0),
                    margin_rate_annual=pm_margin_rate,
                    draw_monthly=pm_draw_monthly,
                    draw_start_date=pm_draw_start_date,
                    draw_monthly_retirement=pm_draw_monthly_retirement,
                    retirement_date=pm_retirement_date,
                    dca_in_retirement=pm_dca_in_retirement,
                    loan_repayment=pm_cf_for_loan if pay_down_margin else 0.0,
                    loan_repayment_freq=pm_cf_freq,
                    dynamic_allocation_schedule=dynamic_plan.dynamic_schedule,
                    dca_mode=dca_config["mode"],
                    dca_target_ticker=dca_config["target_ticker"],
                )
                if api_twr_series is not None:
                    twr_series = api_twr_series
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            logger.warning(f"Testfol API unavailable ({e}), falling back to local engine")
            use_local_engine = True
            api_failover = True

    if use_local_engine:
        # --- Pure Local Path (NDXMEGASIM, threshold rebalancing, or API failover) ---
        tickers = list(alloc_map.keys())
        if shared_prices_df is not None:
            prices_df = shared_prices_df
        else:
            prices_df = fetch_component_data(
                tickers, _return_anchor_start(start_date), end_date
            )

        _shadow_result = shadow_fn(
            allocation=alloc_map,
            start_val=start_val,
            start_date=start_date,
            end_date=end_date,
            api_port_series=None,
            rebalance_freq="None" if no_rebal else (r_mode if (uses_threshold or r_mode == "Custom") else reb.get("freq", "Yearly")),
            cashflow=shadow_cashflow,
            cashflow_freq=cashflow_freq,
            invest_dividends=invest_div,
            pay_down_margin=pay_down_margin,
            tax_config=tax_config,
            custom_rebal_config=reb if r_mode == "Custom" else {},
            prices_df=prices_df,
            rebalance_month=reb.get("month", 1),
            rebalance_day=reb.get("day", 1),
            custom_freq=reb.get("freq", "Yearly"),
            threshold_pct=r_threshold,
            pm_buy_block=pm_buy_block,
            pm_buy_block_threshold=pm_buy_block_threshold,
            starting_loan=_pm_cfg.get("starting_loan", 0.0),
            margin_rate_annual=pm_margin_rate,
            draw_monthly=pm_draw_monthly,
            draw_start_date=pm_draw_start_date,
            draw_monthly_retirement=pm_draw_monthly_retirement,
            retirement_date=pm_retirement_date,
            dca_in_retirement=pm_dca_in_retirement,
            loan_repayment=pm_cf_for_loan if pay_down_margin else 0.0,
            loan_repayment_freq=pm_cf_freq,
            dynamic_allocation_schedule=dynamic_plan.dynamic_schedule,
            dca_mode=dca_config["mode"],
            dca_target_ticker=dca_config["target_ticker"],
        )
        # Unpack: shadow backtest returns 7-tuple or 8-tuple (with pm_blocked_dates)
        if isinstance(_shadow_result, tuple) and len(_shadow_result) == 8:
            trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series, pm_blocked_dates = _shadow_result
        else:
            trades_df, pl_by_year, composition_df, unrealized_pl_df, logs, port_series, twr_series = _shadow_result
            pm_blocked_dates = []

        if dynamic_plan.has_dynamic and dynamic_plan.notes:
            logs = list(dynamic_plan.notes) + list(logs)

        if not port_series.empty:
            _stats_series = twr_series if (twr_series is not None and not twr_series.empty) else port_series
            stats = calculations.generate_stats(_stats_series)

    return {
        "name": name,
        "series": port_series,
        "port_series": port_series,
        "stats": stats,
        "twr_series": twr_series,
        "daily_returns_df": df_rets,
        "is_local": use_local_engine,
        "api_failover": api_failover,
        "trades": trades_df,
        "trades_df": trades_df,
        "pl_by_year": pl_by_year,
        "unrealized_pl_df": unrealized_pl_df,
        "component_prices": prices_df,
        "allocation": original_allocation,
        "maint_pcts": maint_pcts,
        "rebalance": reb,
        "logs": logs,
        "composition": composition_df,
        "composition_df": composition_df,
        "raw_response": extra_data,
        "start_val": start_val,
        "start_date": pd.Timestamp(start_date),
        "sim_range": f"{start_date} to {end_date}",
        "shadow_range": f"{start_date} to {end_date}",
        "wmaint": current_wmaint,
        "wmaint_pm": current_wmaint_pm,
        "pm_blocked_dates": pm_blocked_dates,
        # Internal: kept for pass-2 re-fetch
        "_engine_allocation": alloc_map,
        "_engine_maint_pcts": engine_maint_pcts,
        "_engine_pm_maint_pcts": engine_pm_maint_pcts,
        "_dynamic_allocation_schedule": dynamic_plan.dynamic_schedule,
        "_dynamic_allocation_notes": dynamic_plan.notes or [],
        "_dca_config": dca_config,
        "_reb": reb,
        "_r_mode": r_mode,
    }


def run_multi_backtest(
    portfolios: list[dict],
    start_date: str,
    end_date: str,
    start_val: float,
    cashflow_amount: float,
    cashflow_freq: str,
    invest_div: bool,
    pay_down_margin: bool,
    tax_config: dict,
    bearer_token: str | None,
    fetch_backtest_fn=None,
    run_shadow_fn=None,
    pm_config: dict | None = None,
) -> tuple[list[dict], list]:
    """
    Run backtests for multiple portfolios with common-start alignment.

    Each entry in `portfolios` is a plain dict with keys:
        name, allocation, maint_pcts, rebalance

    Returns (results_list, bench_series_list).
    """
    fetch_fn = fetch_backtest_fn or _default_fetch
    shadow_fn = run_shadow_fn or _default_shadow

    results_list: list[dict] = [None] * len(portfolios)
    bench_by_index: list = [None] * len(portfolios)

    pass1_component_prices = _prefetch_component_universe(
        [ticker for portfolio in portfolios for ticker in portfolio.get("allocation", {})],
        start_date,
        end_date,
        purpose="pass1",
    )

    pass1_workers = _orchestrator_worker_count(len(portfolios))
    if pass1_workers == 1:
        for index, portfolio in enumerate(portfolios):
            _, raw, bench_series = _run_pass1_portfolio(
                index,
                portfolio,
                start_date=start_date,
                end_date=end_date,
                start_val=start_val,
                cashflow_amount=cashflow_amount,
                cashflow_freq=cashflow_freq,
                invest_div=invest_div,
                pay_down_margin=pay_down_margin,
                tax_config=tax_config,
                bearer_token=bearer_token,
                fetch_fn=fetch_fn,
                shadow_fn=shadow_fn,
                pm_config=pm_config,
                prefetched_component_prices=pass1_component_prices,
            )
            results_list[index] = raw
            bench_by_index[index] = bench_series
    else:
        with ThreadPoolExecutor(max_workers=pass1_workers) as executor:
            futures = [
                executor.submit(
                    _run_pass1_portfolio,
                    index,
                    portfolio,
                    start_date=start_date,
                    end_date=end_date,
                    start_val=start_val,
                    cashflow_amount=cashflow_amount,
                    cashflow_freq=cashflow_freq,
                    invest_div=invest_div,
                    pay_down_margin=pay_down_margin,
                    tax_config=tax_config,
                    bearer_token=bearer_token,
                    fetch_fn=fetch_fn,
                    shadow_fn=shadow_fn,
                    pm_config=pm_config,
                    prefetched_component_prices=pass1_component_prices,
                )
                for index, portfolio in enumerate(portfolios)
            ]
            for future in as_completed(futures):
                index, raw, bench_series = future.result()
                results_list[index] = raw
                bench_by_index[index] = bench_series

    bench_series_list = [bench for bench in bench_by_index if bench is not None]

    # --- Pass 2: Common start date alignment ---
    start_dates = []
    for res in results_list:
        s = res.get("series")
        if s is not None and not s.empty:
            start_dates.append(s.index.min())
    for b in bench_series_list:
        if b is not None and not b.empty:
            start_dates.append(b.index.min())

    # Include data availability from failed portfolios so that common_start
    # reflects the newest ticker across ALL portfolios (not just successful ones).
    _failed_indices = []
    if pass1_component_prices is not None and not pass1_component_prices.empty:
        for _fi, _fp in enumerate(portfolios):
            _fs = results_list[_fi].get("series") if results_list[_fi] else None
            if _fs is None or _fs.empty:
                _ftickers = _base_tickers(
                    expand_dynamic_ticker_universe(
                        _fp.get("allocation", {}).keys(),
                        start_date,
                        end_date,
                    )
                )
                _latest = None
                for _ft in _ftickers:
                    if _ft in pass1_component_prices.columns:
                        _fvi = pass1_component_prices[_ft].first_valid_index()
                        if _fvi is not None and (_latest is None or _fvi > _latest):
                            _latest = _fvi
                if _latest is not None:
                    start_dates.append(_latest)
                    _failed_indices.append(_fi)

    common_start = max(start_dates) if start_dates else None
    global_start_val = start_val

    # Re-run failed portfolios from common_start (they had no data at the
    # original start_date but may have data from common_start onward).
    if common_start and _failed_indices:
        _cs_str = common_start.strftime("%Y-%m-%d")
        for _fi in _failed_indices:
            _, _raw, _bench = _run_pass1_portfolio(
                _fi,
                portfolios[_fi],
                start_date=_cs_str,
                end_date=end_date,
                start_val=start_val,
                cashflow_amount=cashflow_amount,
                cashflow_freq=cashflow_freq,
                invest_div=invest_div,
                pay_down_margin=pay_down_margin,
                tax_config=tax_config,
                bearer_token=bearer_token,
                fetch_fn=fetch_fn,
                shadow_fn=shadow_fn,
                pm_config=pm_config,
                prefetched_component_prices=pass1_component_prices,
            )
            results_list[_fi] = _raw
            if _bench is not None:
                bench_by_index[_fi] = _bench
        bench_series_list = [b for b in bench_by_index if b is not None]

    # Resolve draw_start_date for Pass 2 re-runs
    _p2_cfg = pm_config or {}
    _p2_bt_start = pd.Timestamp(start_date).date() if isinstance(start_date, str) else start_date
    _p2_bt_end = pd.Timestamp(end_date).date() if isinstance(end_date, str) else end_date
    _p2_raw_draw_start = _p2_cfg.get("draw_start_date", None)
    if _p2_raw_draw_start is not None:
        if isinstance(_p2_raw_draw_start, str):
            _p2_raw_draw_start = pd.Timestamp(_p2_raw_draw_start).date()
        elif isinstance(_p2_raw_draw_start, datetime.datetime):
            _p2_raw_draw_start = _p2_raw_draw_start.date()
        pm_draw_start_date = max(_p2_raw_draw_start, _p2_bt_start)
        pm_draw_start_date = min(pm_draw_start_date, _p2_bt_end)
    else:
        pm_draw_start_date = _p2_bt_start
    # Resolve retirement_date for Pass 2
    _p2_raw_ret = _p2_cfg.get("retirement_date", None)
    if _p2_raw_ret is not None:
        if isinstance(_p2_raw_ret, str):
            _p2_raw_ret = pd.Timestamp(_p2_raw_ret).date()
        elif isinstance(_p2_raw_ret, datetime.datetime):
            _p2_raw_ret = _p2_raw_ret.date()
        # Clamp retirement_date to [common_start, end_date] for Pass-2
        _p2_cs = common_start.date() if hasattr(common_start, 'date') else common_start
        pm_retirement_date = max(_p2_raw_ret, _p2_cs) if _p2_cs else _p2_raw_ret
        pm_retirement_date = min(pm_retirement_date, _p2_bt_end)
    else:
        pm_retirement_date = None
    pm_draw_monthly_retirement = _p2_cfg.get("draw_monthly_retirement", 0.0)

    if common_start:
        local_pass2_component_prices = _prefetch_component_universe(
            [
                ticker
                for res in results_list
                if res.get("is_local", False)
                and res.get("series") is not None
                and not res["series"].empty
                and res["series"].index[0] < common_start - pd.Timedelta(days=3)
                for ticker in res.get("_engine_allocation", res.get("allocation", {}))
            ],
            common_start.strftime("%Y-%m-%d"),
            end_date,
            purpose="pass2",
        )

        rerun_candidates = [
            (index, res)
            for index, res in enumerate(results_list)
            if res.get("series") is not None
            and not res["series"].empty
            and res["series"].index[0] < common_start - pd.Timedelta(days=3)
        ]

        pass2_workers = _orchestrator_worker_count(len(rerun_candidates))
        if pass2_workers == 1:
            for index, res in rerun_candidates:
                _, updated_res = _rerun_result_for_common_start(
                    index,
                    res,
                    common_start=common_start,
                    end_date=end_date,
                    global_start_val=global_start_val,
                    cashflow_amount=cashflow_amount,
                    cashflow_freq=cashflow_freq,
                    invest_div=invest_div,
                    pay_down_margin=pay_down_margin,
                    tax_config=tax_config,
                    bearer_token=bearer_token,
                    fetch_fn=fetch_fn,
                    shadow_fn=shadow_fn,
                    pm_config=pm_config,
                    pm_draw_start_date=pm_draw_start_date,
                    pm_retirement_date=pm_retirement_date,
                    prefetched_component_prices=local_pass2_component_prices,
                )
                results_list[index] = updated_res
        else:
            with ThreadPoolExecutor(max_workers=pass2_workers) as executor:
                futures = [
                    executor.submit(
                        _rerun_result_for_common_start,
                        index,
                        res,
                        common_start=common_start,
                        end_date=end_date,
                        global_start_val=global_start_val,
                        cashflow_amount=cashflow_amount,
                        cashflow_freq=cashflow_freq,
                        invest_div=invest_div,
                        pay_down_margin=pay_down_margin,
                        tax_config=tax_config,
                        bearer_token=bearer_token,
                        fetch_fn=fetch_fn,
                        shadow_fn=shadow_fn,
                        pm_config=pm_config,
                        pm_draw_start_date=pm_draw_start_date,
                        pm_retirement_date=pm_retirement_date,
                        prefetched_component_prices=local_pass2_component_prices,
                    )
                    for index, res in rerun_candidates
                ]
                for future in as_completed(futures):
                    index, updated_res = future.result()
                    results_list[index] = updated_res

    # Align benchmarks to common_start (Bug 11 fix)
    if common_start:
        for j, b in enumerate(bench_series_list):
            if b is not None and not b.empty and b.index[0] < common_start:
                bench_series_list[j] = b[b.index >= common_start]

    return results_list, bench_series_list
