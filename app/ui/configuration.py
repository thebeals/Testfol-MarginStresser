import streamlit as st
import pandas as pd
import json
import os
from app.common import utils
from app.core import tax_library
from app.services import testfol_api as api
from . import asset_explorer
from . import ndx_scanner


DEFAULT_GLOBAL_CASHFLOW = {
    "start_val": 10000.0,
    "amount": 0.0,
    "freq": "Monthly",
    "invest_div": True,
    "pay_down_margin": False,
    "fund_dca_margin": False,
}


def clear_runtime_caches(session_state=None) -> dict[str, int]:
    """Clear Streamlit/in-memory caches without deleting anything under data/."""
    state = session_state if session_state is not None else st.session_state
    removed_session_keys = 0

    st.cache_data.clear()
    st.cache_resource.clear()

    for key in ("ae_cache", "ae_future"):
        if key in state:
            del state[key]
            removed_session_keys += 1

    try:
        from app.services.price_providers import reset_provider
        reset_provider()
    except Exception:
        pass

    try:
        from app.services import testfol_auth
        testfol_auth._token_cache.clear()
    except Exception:
        pass

    try:
        tax_library._STATE_TAX_TABLES.clear()
    except Exception:
        pass

    return {
        "streamlit_data": 1,
        "streamlit_resource": 1,
        "session_keys_removed": removed_session_keys,
    }


def _presets_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "../../data/presets.json")


def _sort_preset_names(presets: list[dict]) -> list[str]:
    """Sort regular presets first, with research variants grouped last."""
    names = [p["name"] for p in presets]

    def is_ndx(name: str) -> bool:
        return name.upper().startswith("NDX")

    def is_research(name: str) -> bool:
        return "RESEARCH" in name.upper()

    regular_ndx = sorted(name for name in names if is_ndx(name) and not is_research(name))
    regular_other = sorted(name for name in names if not is_ndx(name) and not is_research(name))
    research = sorted(name for name in names if is_research(name))
    return regular_ndx + regular_other + research


@st.cache_data(show_spinner=False)
def _load_presets(preset_mtime: float | None = None):
    """Load and sort preset names from disk (cached)."""
    del preset_mtime  # Included in the cache key so file edits invalidate presets.
    preset_path = _presets_path()
    if not os.path.exists(preset_path):
        return [], []
    with open(preset_path, "r") as f:
        presets = json.load(f)
    return _sort_preset_names(presets), presets

def render():
    """Renders the configuration tabs and returns a config dictionary."""
    
    st.subheader("Strategy Configuration")
    
    tab_port, tab_margin, tab_asset, tab_ndx, tab_settings = st.tabs([
        "💼 Portfolio",
        "🏦 Margin & Financing",
        "🧩 Asset Explorer",
        "📊 NDX Scanner",
        "⚙️ Settings",
    ])

    config = {}

    # Per-portfolio widget key bases — suffixed with portfolio id at runtime
    _PF_KEY_BASES = [
        "p_name", "p_rmode", "p_rfreq", "p_rmon", "p_rday", "p_cmp",
        "p_rthresh", "p_rfreq_tc", "p_rthresh_tc", "p_rfreq_std",
        "p_dca_mode", "p_dca_target", "p_editor",
    ]

    def _pop_pf_keys(pid):
        """Clear widget keys for a specific portfolio id."""
        for base in _PF_KEY_BASES:
            st.session_state.pop(f"{base}_{pid}", None)

    def _portfolio_fragment():
        # --- initialize state ---
        if "portfolios" not in st.session_state:
            _default_alloc = pd.DataFrame([
                {"Ticker": "NDXMEGASIM?L=2&E=0.95", "Weight %": 60.0, "Maint %": 50.0, "PM Maint %": 30.0},
                {"Ticker": "GLDSIM?E=0.40", "Weight %": 20.0, "Maint %": 25.0, "PM Maint %": 15.0},
                {"Ticker": "VXUSSIM?E=0.05", "Weight %": 15.0, "Maint %": 25.0, "PM Maint %": 9.0},
                {"Ticker": "QQQSIM?L=3&E=0.82", "Weight %": 5.0, "Maint %": 75.0, "PM Maint %": 30.0}
            ])
            st.session_state.portfolios = [{
                "id": "p1",
                "name": "NDXMEGASPLIT (w/ ERs)",
                "alloc_df": _default_alloc,
                "rebalance": {
                    "mode": "Custom", "freq": "Yearly", "month": 1, "day": 1, "compare_std": False, "threshold_pct": 5.0
                },
                "dca": {"mode": "Proportional", "target_ticker": ""},
                "cashflow": {
                     "start_val": 10000.0, "amount": 0.0, "freq": "Monthly", "invest_div": True, "pay_down_margin": False
                }
            }]

        # Migrate stale names that can live in Streamlit widget/session state
        # after preset cleanup. Without this, text_input keys can keep showing
        # old labels even after data/presets.json and code defaults are fixed.
        _canonical_portfolio_names = {
            "NDXMEGASPLIT - Legacy 60/20/15/5 (w/ ERs)": "NDXMEGASPLIT (w/ ERs)",
            "NDXMEGASPLIT - Legacy 60/20/15/5 (No ER)": "NDXMEGASPLIT",
        }
        for _portfolio in st.session_state.portfolios:
            _pid = _portfolio.get("id")
            _old_name = _portfolio.get("name")
            if _old_name in _canonical_portfolio_names:
                _portfolio["name"] = _canonical_portfolio_names[_old_name]
            _name_key = f"p_name_{_pid}"
            if _name_key in st.session_state:
                _widget_name = st.session_state[_name_key]
                if _widget_name in _canonical_portfolio_names:
                    st.session_state[_name_key] = _canonical_portfolio_names[_widget_name]
            _portfolio.setdefault("dca", {"mode": "Proportional", "target_ticker": ""})

        if "global_cashflow" not in st.session_state:
            st.session_state.global_cashflow = DEFAULT_GLOBAL_CASHFLOW.copy()
        else:
            st.session_state.global_cashflow.setdefault("fund_dca_margin", False)

        if "active_tab_idx" not in st.session_state:
            st.session_state.active_tab_idx = 0

        # Sync active portfolio name from its text_input key
        _aidx = st.session_state.active_tab_idx
        if _aidx < len(st.session_state.portfolios):
            _active_pid = st.session_state.portfolios[_aidx]["id"]
            _name_key = f"p_name_{_active_pid}"
            if _name_key in st.session_state:
                st.session_state.portfolios[_aidx]["name"] = st.session_state[_name_key]

        portfolio_names = [port["name"] for port in st.session_state.portfolios]

        # Build unique display names for segmented control (handles duplicate names)
        display_names = list(portfolio_names)
        _seen = {}
        for i, name in enumerate(display_names):
            count = _seen.get(name, 0) + 1
            _seen[name] = count
            if count > 1:
                display_names[i] = f"{name} ({count})"

        # Clamp active_tab_idx
        idx = min(st.session_state.active_tab_idx, len(display_names) - 1)
        if idx < 0:
            idx = 0
        st.session_state.active_tab_idx = idx

        # --- Render Global Section ---
        st.markdown("##### 💰 Global Capital & Cashflow")
        gc1, gc2, gc3, gc4, gc5, gc6 = st.columns([2, 2, 2, 1.5, 1.5, 1.5])
        with gc1:
            _start_value = st.session_state.get("g_start", st.session_state.global_cashflow["start_val"])
            st.session_state.global_cashflow["start_val"] = utils.num_input(
                "Start Value ($)",
                "g_start",
                st.session_state.global_cashflow["start_val"],
                1000.0,
                format=utils.optional_cents_format(_start_value),
            )
        with gc2:
            _cashflow_value = st.session_state.get("g_cf", st.session_state.global_cashflow["amount"])
            st.session_state.global_cashflow["amount"] = utils.num_input(
                "Cashflow ($)",
                "g_cf",
                st.session_state.global_cashflow["amount"],
                100.0,
                format=utils.optional_cents_format(_cashflow_value),
            )
        with gc3:
             st.session_state.global_cashflow["freq"] = st.selectbox("Freq", ["Monthly", "Quarterly", "Yearly"], index=["Monthly", "Quarterly", "Yearly"].index(st.session_state.global_cashflow["freq"]), key="g_freq")
        with gc4:
             st.markdown("<br>", unsafe_allow_html=True)
             st.session_state.global_cashflow["invest_div"] = st.checkbox("Re-invest Divs", st.session_state.global_cashflow["invest_div"], key="g_div")
        with gc5:
             st.markdown("<br>", unsafe_allow_html=True)
             st.session_state.global_cashflow["pay_down_margin"] = st.checkbox("Pay Down Margin", st.session_state.global_cashflow["pay_down_margin"], key="g_paydown")
        with gc6:
             st.markdown("<br>", unsafe_allow_html=True)
             st.session_state.global_cashflow["fund_dca_margin"] = st.checkbox("Fund DCA w/ Margin", st.session_state.global_cashflow.get("fund_dca_margin", False), key="g_fund_margin")

        st.divider()

        # --- Portfolio Management Toolbar ---
        c_tool1, c_tool2, c_tool3 = st.columns([1, 2, 1])

        with c_tool1:
            if st.button("➕ Add Empty", use_container_width=True):
                if len(st.session_state.portfolios) < 5:
                    import uuid
                    new_id = f"p_{uuid.uuid4().hex[:8]}"
                    st.session_state.portfolios.append({
                        "id": new_id,
                        "name": f"Portfolio {len(st.session_state.portfolios) + 1}",
                        "alloc_df": pd.DataFrame([{"Ticker":"SPY", "Weight %": 100, "Maint %": 25, "PM Maint %": 0.0}]),
                        "rebalance": {"mode": "Standard", "freq": "Yearly", "month": 1, "day": 1, "compare_std": False, "threshold_pct": 5.0},
                        "dca": {"mode": "Proportional", "target_ticker": ""}
                    })
                    st.session_state.active_tab_idx = len(st.session_state.portfolios) - 1
                    st.session_state.pop("portfolio_selector", None)
                    st.rerun()
                else:
                    st.warning("Max 5 portfolios.")

        try:
            preset_path = _presets_path()
            preset_mtime = os.path.getmtime(preset_path) if os.path.exists(preset_path) else None
            preset_names, presets = _load_presets(preset_mtime)

            with c_tool2:
                selected_preset = st.selectbox(
                    "Select Preset",
                    preset_names if preset_names else ["No Presets"],
                    key="preset_selector",
                    label_visibility="collapsed"
                )

            with c_tool3:
                if st.button("⬇️ Load Preset", use_container_width=True):
                    if preset_names and selected_preset and selected_preset != "No Presets":
                        if len(st.session_state.portfolios) < 5:
                            p_data = next(p for p in presets if p["name"] == selected_preset)
                            import uuid
                            new_id = f"p_pre_{uuid.uuid4().hex[:8]}"
                            reb = p_data.get("rebalance", {})
                            month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
                            r_month = month_map.get(reb.get("month_str", "Jan"), 1)
                            _preset_df = pd.DataFrame(p_data["allocation"])
                            if "PM Maint %" not in _preset_df.columns:
                                _preset_df["PM Maint %"] = 0.0
                            st.session_state.portfolios.append({
                                "id": new_id,
                                "name": p_data["name"],
                                "alloc_df": _preset_df,
                                "rebalance": {
                                    "mode": reb.get("mode", "Standard"),
                                    "freq": reb.get("freq", "Yearly"),
                                    "month": r_month,
                                    "day": reb.get("day", 1),
                                    "compare_std": False,
                                    "threshold_pct": reb.get("threshold_pct", 5.0),
                                },
                                "cashflow": {
                                    "start_val": 10000.0, "amount": 0.0, "freq": "Monthly",
                                    "invest_div": True, "pay_down_margin": False
                                },
                                "dca": {"mode": "Proportional", "target_ticker": ""}
                            })
                            st.session_state.active_tab_idx = len(st.session_state.portfolios) - 1
                            st.session_state.pop("portfolio_selector", None)
                            st.rerun()
                        else:
                            st.warning("Max 5 portfolios.")
        except Exception as e:
            st.error(f"Error: {e}")

        st.divider()

        # --- Portfolio Selector (outside fragment — tab switch = full rerun) ---
        def _on_tab_change():
            sel = st.session_state.portfolio_selector
            if sel in display_names:
                new_idx = display_names.index(sel)
                new_idx = min(new_idx, len(st.session_state.portfolios) - 1)
                st.session_state.active_tab_idx = new_idx

        # Ensure selector is valid before rendering
        _sel = st.session_state.get("portfolio_selector")
        if _sel is None or _sel not in display_names:
            st.session_state.portfolio_selector = display_names[idx]

        st.segmented_control(
            "Portfolio",
            display_names,
            key="portfolio_selector",
            on_change=_on_tab_change,
        )

        # --- Per-portfolio content (inside fragment — stable keys, fast reruns) ---
        @st.fragment
        def _portfolio_content():
            idx = st.session_state.active_tab_idx
            p = st.session_state.portfolios[idx]

            pid = p["id"]

            with st.container():
                c_name, c_save, c_del = st.columns([6, 1, 1])
                with c_name:
                    _nk = f"p_name_{pid}"
                    if _nk not in st.session_state:
                        st.session_state[_nk] = p["name"]
                    p["name"] = st.text_input("Portfolio Name", key=_nk, label_visibility="collapsed")

                with c_save:
                    if st.button("💾", key="p_save", help="Save as new Preset", use_container_width=True):
                        alloc_list = p["alloc_df"].to_dict("records")
                        preset_data = {
                            "name": p["name"],
                            "allocation": alloc_list,
                            "rebalance": {
                                "mode": p["rebalance"]["mode"],
                                "freq": p["rebalance"]["freq"],
                                "day": p["rebalance"]["day"]
                            }
                        }
                        month_map_inv = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
                        preset_data["rebalance"]["month_str"] = month_map_inv.get(p["rebalance"].get("month", 1), "Jan")
                        utils.save_preset(preset_data)
                        _load_presets.clear()  # invalidate cached presets
                        st.success(f"Saved!")
                        st.rerun(scope="app")

                with c_del:
                    if st.button("🗑️", key="p_del", help="Delete Portfolio", use_container_width=True):
                        if len(st.session_state.portfolios) > 1:
                            deleted_pid = p["id"]
                            st.session_state.portfolios.pop(idx)
                            new_idx = max(0, idx-1)
                            st.session_state.active_tab_idx = new_idx
                            st.session_state.pop("portfolio_selector", None)
                            _pop_pf_keys(deleted_pid)
                            st.rerun(scope="app")
                        else:
                            st.warning("Last portfolio")

            # --- Rebalancing Strategy ---
            with st.expander("📅 Rebalancing Strategy", expanded=False):
                mode_options = ["None", "Standard", "Custom", "Threshold", "Threshold+Calendar"]
                try:
                    mode_idx = mode_options.index(p["rebalance"]["mode"])
                except (ValueError, KeyError):
                    mode_idx = 0
                r_mode = st.radio("Mode", mode_options, index=mode_idx, key=f"p_rmode_{pid}", horizontal=True, label_visibility="collapsed")
                p["rebalance"]["mode"] = r_mode

                c_r1, c_r2, c_r3, c_r4 = st.columns(4)

                if r_mode == "None":
                    st.caption("No rebalancing — positions will drift with market returns.")

                elif r_mode == "Custom":
                    with c_r1:
                        freq_opts = ["Yearly", "Quarterly", "Monthly"]
                        try: f_idx = freq_opts.index(p["rebalance"]["freq"])
                        except (ValueError, KeyError): f_idx = 0
                        p["rebalance"]["freq"] = st.selectbox("Frequency", freq_opts, index=f_idx, key=f"p_rfreq_{pid}")

                    with c_r2:
                        if p["rebalance"]["freq"] == "Yearly":
                            p["rebalance"]["month"] = st.selectbox("Rebalance Month", range(1, 13), index=p["rebalance"]["month"]-1, format_func=lambda x: pd.to_datetime(f"2024-{x}-1").strftime("%b"), key=f"p_rmon_{pid}")
                        else:
                            p["rebalance"]["month"] = 1
                            st.markdown("")

                    with c_r3:
                        p["rebalance"]["day"] = st.number_input("Day of Month", 1, 31, p["rebalance"]["day"], key=f"p_rday_{pid}")

                    with c_r4:
                        st.markdown("<br>", unsafe_allow_html=True)
                        p["rebalance"]["compare_std"] = st.checkbox("Compare vs Standard", p["rebalance"]["compare_std"], key=f"p_cmp_{pid}")

                elif r_mode == "Threshold":
                    with c_r1:
                        p["rebalance"]["threshold_pct"] = st.number_input(
                            "Drift Threshold (%)", 1.0, 50.0,
                            float(p["rebalance"].get("threshold_pct", 5.0)),
                            step=1.0, key=f"p_rthresh_{pid}",
                            help="Rebalance when any position drifts more than X pp from target. Checked daily."
                        )
                    p["rebalance"]["freq"] = "Yearly"
                    p["rebalance"]["month"] = 1
                    p["rebalance"]["day"] = 1

                elif r_mode == "Threshold+Calendar":
                    with c_r1:
                        freq_opts = ["Yearly", "Quarterly", "Monthly"]
                        try: f_idx = freq_opts.index(p["rebalance"]["freq"])
                        except (ValueError, KeyError): f_idx = 0
                        p["rebalance"]["freq"] = st.selectbox("Check Frequency", freq_opts, index=f_idx, key=f"p_rfreq_tc_{pid}")
                    with c_r2:
                        p["rebalance"]["threshold_pct"] = st.number_input(
                            "Drift Threshold (%)", 1.0, 50.0,
                            float(p["rebalance"].get("threshold_pct", 5.0)),
                            step=1.0, key=f"p_rthresh_tc_{pid}",
                            help="Only rebalance at scheduled check dates if drift exceeds threshold."
                        )
                    p["rebalance"]["month"] = 1
                    p["rebalance"]["day"] = 1

                else: # Standard Mode
                    with c_r1:
                        p["rebalance"]["freq"] = st.selectbox("Frequency", ["Yearly", "Quarterly", "Monthly"], index=["Yearly", "Quarterly", "Monthly"].index(p["rebalance"]["freq"]), key=f"p_rfreq_std_{pid}")
                    p["rebalance"]["month"] = 1
                    p["rebalance"]["day"] = 1

            # --- DCA Routing ---
            p.setdefault("dca", {"mode": "Proportional", "target_ticker": ""})
            dca_mode_options = ["Proportional", "Single Asset"]
            if p["dca"].get("mode") not in dca_mode_options:
                p["dca"]["mode"] = "Proportional"
            dca_tickers = [
                str(ticker).strip()
                for ticker in p["alloc_df"].get("Ticker", pd.Series(dtype=str)).tolist()
                if str(ticker).strip()
            ]
            with st.expander("💵 DCA Routing", expanded=False):
                dca_mode_idx = dca_mode_options.index(p["dca"].get("mode", "Proportional"))
                p["dca"]["mode"] = st.radio(
                    "Mode",
                    dca_mode_options,
                    index=dca_mode_idx,
                    key=f"p_dca_mode_{pid}",
                    horizontal=True,
                    label_visibility="collapsed",
                )
                if p["dca"]["mode"] == "Single Asset":
                    if dca_tickers:
                        current_target = p["dca"].get("target_ticker") or dca_tickers[0]
                        if current_target not in dca_tickers:
                            current_target = dca_tickers[0]
                        p["dca"]["target_ticker"] = st.selectbox(
                            "DCA Target",
                            dca_tickers,
                            index=dca_tickers.index(current_target),
                            key=f"p_dca_target_{pid}",
                        )
                    else:
                        p["dca"]["target_ticker"] = ""
                else:
                    p["dca"]["target_ticker"] = ""

            st.markdown("##### 🥧 Asset Allocation")
            # Ensure PM Maint % column exists (backward compat)
            if "PM Maint %" not in p["alloc_df"].columns:
                p["alloc_df"]["PM Maint %"] = 0.0
            new_alloc_df = st.data_editor(
                p["alloc_df"],
                key=f"p_editor_{pid}",
                num_rows="dynamic",
                column_order=["Ticker", "Weight %", "Maint %", "PM Maint %"],
                column_config={
                    "Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                    "Maint %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
                    "PM Maint %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
                },
                use_container_width=True
            )

            if not new_alloc_df.equals(p["alloc_df"]):
                p["alloc_df"] = new_alloc_df
                st.rerun(scope="fragment")

            # Validation & Metrics
            try:
                p_alloc_preview, p_maint_preview = api.table_to_dicts(p["alloc_df"])
                p_total_weight = sum(p_alloc_preview.values())
                d_maint = st.session_state.get('default_maint', config.get('default_maint', 25.0))
                p_wmaint = sum(
                    (wt/100) * (p_maint_preview.get(t.split("?")[0], d_maint)/100)
                    for t, wt in p_alloc_preview.items()
                )
            except Exception:
                p_total_weight = 0.0
                p_wmaint = 0.0

            # Compute PM weighted maint
            p_wmaint_pm = 0.0
            has_pm = False
            try:
                for _, _row in p["alloc_df"].iterrows():
                    _pm_val = float(_row.get("PM Maint %", 0))
                    if _pm_val > 0:
                        has_pm = True
                        _t = _row["Ticker"].split("?")[0]
                        _wt = float(_row.get("Weight %", 0))
                        p_wmaint_pm += (_wt / 100) * (_pm_val / 100)
            except Exception:
                pass

            st.markdown("---")
            if has_pm:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total Allocation", f"{p_total_weight:.2f}%", delta=None if abs(p_total_weight - 100) < 0.01 else "Must be 100%", delta_color="off" if abs(p_total_weight - 100) < 0.01 else "inverse")
                mc2.metric("Weighted Maint Req", f"{p_wmaint*100:.2f}%")
                mc3.metric("Weighted PM Maint", f"{p_wmaint_pm*100:.2f}%")
            else:
                mc1, mc2 = st.columns(2)
                mc1.metric("Total Allocation", f"{p_total_weight:.2f}%", delta=None if abs(p_total_weight - 100) < 0.01 else "Must be 100%", delta_color="off" if abs(p_total_weight - 100) < 0.01 else "inverse")
                mc2.metric("Weighted Maint Req", f"{p_wmaint*100:.2f}%")

        _portfolio_content()
        
    with tab_port:
        _portfolio_fragment()
        config['portfolios'] = st.session_state.portfolios
        config['global_cashflow'] = st.session_state.get('global_cashflow', DEFAULT_GLOBAL_CASHFLOW.copy())

    # Margin widget keys to persist across portfolio switches
    _MARGIN_WIDGET_KEYS = [
        "tax_sim_mode", "other_income", "filing_status", "state_code",
        "tax_method", "starting_loan", "equity_init", "starting_cash",
        "draw_monthly", "draw_start_date", "draw_monthly_retirement",
        "retirement_date", "dca_in_retirement", "retirement_income",
        "default_maint", "margin_rate_model", "rate_annual",
        "spread_pct", "t1_spread", "t2_spread", "t3_spread", "t4_spread",
        "pm_mode", "pm_buy_block", "pm_buy_block_threshold",
    ]

    # Restore margin settings from backup (survives fragment widget resets)
    for _k, _v in st.session_state.get("_margin_store", {}).items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    @st.fragment
    def _margin_fragment():
        # Move Tax Simulation to top to control state of other inputs
        tax_sim_mode = st.radio(
            "Tax Payment Simulation",
            ["None (Gross)", "Pay from Cash", "Pay with Margin"],
            index=2,
            key="tax_sim_mode",
            horizontal=True, # Make it horizontal to save space at top
            help="**None (Gross)**: Show raw pre-tax returns.\n**Pay from Cash**: Simulate selling shares to pay taxes (reduces equity).\n**Pay with Margin**: Simulate borrowing to pay taxes (increases loan)."
        )
        
        # Map selection to flags
        config['pay_tax_margin'] = (tax_sim_mode == "Pay with Margin")
        config['pay_tax_cash'] = (tax_sim_mode == "Pay from Cash")
        
        with st.expander("Tax Configuration", expanded=True):
            c_tax1, c_tax2, c_tax3 = st.columns(3)
            with c_tax1:
                config['other_income'] = utils.num_input("Annual Income ($)", "other_income", 100000.0, 5000.0)
            with c_tax2:
                config['filing_status'] = st.selectbox("Filing Status", ["Single", "Married Filing Jointly", "Head of Household", "Married Filing Separately"], index=0, key="filing_status")
            with c_tax3:
                state_options = ["(None)"] + [f"{code} - {name}" for code, name in tax_library.US_STATES]
                state_sel = st.selectbox("State", state_options, index=0, key="state_code")
                config['state_code'] = state_sel.split(" - ")[0] if " - " in state_sel else ""

                if config['state_code'] in tax_library.NO_INCOME_TAX_STATES:
                    st.caption("No state income tax")
                elif config['state_code']:
                    top = tax_library.get_state_top_rate(config['state_code'])
                    st.caption(f"Progressive brackets (top rate {top*100:.2f}%)")

            st.caption("Federal tax brackets are automatically applied based on income & filing status.")
            
            tax_method_selection = st.radio(
                "Tax Calculation Method",
                ["Smart (Historical Brackets)", "Max (Top Rate)", "Fixed (2026 Rates)"],
                index=0,
                horizontal=True,
                key="tax_method",
                help="Smart: Uses historical inclusion rates. Max: Flat historical max rate. Fixed: Modern 0/15/20% for all years."
            )
            
            if "Smart" in tax_method_selection:
                 config['tax_method'] = "historical_smart"
            elif "Max" in tax_method_selection:
                 config['tax_method'] = "historical_max_rate"
            else:
                 config['tax_method'] = "2026_fixed"
        
        # Disable margin inputs if "Pay from Cash" is selected (per user request)
        # This implies a "Cash Only" mindset for this mode
        margin_disabled = config['pay_tax_cash']

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Loan Configuration")
            config['starting_loan'] = utils.num_input(
                "Starting Loan ($)", "starting_loan", 0.0, 100.0,
                on_change=utils.sync_equity,
                disabled=margin_disabled
            )
            config['equity_init'] = utils.num_input(
                "Initial Equity %", "equity_init", 100.0, 1.0,
                on_change=utils.sync_loan,
                disabled=margin_disabled
            )

            # Starting Cash (mutually exclusive with starting loan)
            if config['starting_loan'] == 0.0 and not margin_disabled:
                config['starting_cash'] = utils.num_input(
                    "Starting Cash ($)", "starting_cash", 0.0, 100.0,
                )
            else:
                config['starting_cash'] = 0.0

            # Calculate leverage for display (handle zero division)
            curr_start_val = st.session_state.get('global_cashflow', {}).get('start_val', 10000.0)
            if config['starting_loan'] < curr_start_val:
                current_lev = curr_start_val / (curr_start_val - config['starting_loan'])
                st.caption(f"Current Leverage: **{current_lev:.2f}x**")
            else:
                st.caption("Current Leverage: **∞x**")

            st.divider()
            st.markdown("##### Withdrawal & Retirement")

            # Pre-retirement draws
            _wd1, _wd2 = st.columns(2)
            with _wd1:
                config['draw_monthly'] = utils.num_input("Pre-Retirement Draw ($)", "draw_monthly", 0.0, 100.0)
            with _wd2:
                if config['draw_monthly'] > 0:
                    config['draw_start_date'] = st.date_input(
                        "Draw Start Date",
                        value=None,
                        key="draw_start_date",
                        min_value=pd.Timestamp("2000-01-01").date(),
                        max_value=pd.Timestamp("2060-12-31").date(),
                        help="Date when pre-retirement draws begin. Defaults to backtest start if not set.",
                    )
                else:
                    config['draw_start_date'] = None

            # Retirement draws (independent of pre-retirement)
            _rd1, _rd2 = st.columns(2)
            with _rd1:
                config['draw_monthly_retirement'] = utils.num_input(
                    "Retirement Draw ($)", "draw_monthly_retirement", 0.0, 100.0,
                )
            with _rd2:
                if config['draw_monthly_retirement'] > 0:
                    config['retirement_date'] = st.date_input(
                        "Retirement Date",
                        value=None,
                        key="retirement_date",
                        min_value=pd.Timestamp("2000-01-01").date(),
                        max_value=pd.Timestamp("2060-12-31").date(),
                        help="Date when retirement draws begin (replaces pre-retirement draw if set).",
                    )
                else:
                    config['retirement_date'] = None

            # Retirement options (show when any draw is configured)
            _has_any_draw = config['draw_monthly'] > 0 or config['draw_monthly_retirement'] > 0
            if _has_any_draw:
                _rc1, _rc2 = st.columns(2)
                with _rc1:
                    config['dca_in_retirement'] = st.checkbox(
                        "Continue DCA in Retirement",
                        value=False,
                        key="dca_in_retirement",
                        help="If unchecked, DCA contributions stop at the draw/retirement start date.",
                    )
                with _rc2:
                    config['retirement_income'] = utils.num_input(
                        "Retirement Income ($)", "retirement_income", 0.0, 5000.0,
                    )
                st.caption("Annual non-portfolio income after retirement date (replaces Annual Income for tax brackets).")
            else:
                config['dca_in_retirement'] = True
                config['retirement_income'] = None

            st.divider()
            config['default_maint'] = utils.num_input("Default Maint %", "default_maint", 25.0, 1.0, disabled=margin_disabled)

        with c2:
            st.markdown("##### Rates & Maintenance")

            
            margin_mode = st.selectbox("Margin Rate Model", ["Fixed", "Variable (Fed + Spread)", "Tiered (Blended)"],
                                      index=2,
                                      key="margin_rate_model",
                                      help="**Fixed**: Constant annual rate.\n**Variable**: Fed Funds Rate (Daily) + User Spread.\n**Tiered**: Blended rate based on loan size (Base + Tiered Spread).",
                                      disabled=margin_disabled)
            
            margin_config = {}
            if margin_mode == "Fixed":
                margin_config = {
                    "type": "Fixed",
                    "rate_pct": utils.num_input("Annual Interest %", "rate_annual", 8.0, 0.5, disabled=margin_disabled)
                }
            elif margin_mode == "Variable (Fed + Spread)":
                from app.services import data_service
                fed_series = data_service.get_fed_funds_rate()
                
                # Show Preview
                curr_rate = fed_series.iloc[-1] if fed_series is not None and not fed_series.empty else 0.0
                st.caption(f"Current Base: **{curr_rate:.2f}%** (Fed Effective)")
                
                spread = utils.num_input("Spread over Base %", "spread_pct", 1.5, 0.1, disabled=margin_disabled)
                
                margin_config = {
                    "type": "Variable",
                    "base_series": fed_series,
                    "spread_pct": spread
                }
            else: # Tiered
                from app.services import data_service
                fed_series = data_service.get_fed_funds_rate()

                curr_rate = fed_series.iloc[-1] if fed_series is not None and not fed_series.empty else 0.0
                st.caption(f"Current Base: **{curr_rate:.2f}%**")
                
                st.markdown("**IBKR Pro Spreads (Blended)**")
                # IBKR Pro Tiers: 0-100k, 100k-1M, 1M-50M, >50M
                c_t1, c_t2 = st.columns(2)
                with c_t1:
                    t1_spread = st.number_input("Tier 1 (<100k) %", value=1.5, step=0.1, key="t1_spread")
                    t3_spread = st.number_input("Tier 3 (1M-50M) %", value=0.75, step=0.05, key="t3_spread")
                with c_t2:
                    t2_spread = st.number_input("Tier 2 (100k-1M) %", value=1.0, step=0.1, key="t2_spread")
                    t4_spread = st.number_input("Tier 4 (>50M) %", value=0.5, step=0.05, key="t4_spread")
                
                tiers = [
                    (0, t1_spread),
                    (100000, t2_spread),
                    (1000000, t3_spread),
                    (50000000, t4_spread)
                ]
                
                margin_config = {
                    "type": "Tiered",
                    "base_series": fed_series,
                    "tiers": tiers
                }
            
            config['rate_annual'] = margin_config
            
            # Portfolio Margin Controls
            st.divider()
            st.markdown("##### Portfolio Margin (PM)")
            pm_mode = st.selectbox(
                "PM Comparison Mode",
                ["Off", "Compare"],
                index=0,
                help="**Off**: Reg-T only.\n**Compare**: Show Reg-T and PM usage curves side by side.",
                disabled=margin_disabled,
                key="pm_mode",
            )
            config['pm_mode'] = pm_mode
            config['pm_threshold'] = 110000.0

            pm_buy_block = st.checkbox(
                "Simulate PM Buy Block (Equity < threshold)",
                value=False,
                help="When enabled, the shadow backtest skips BUY trades at rebalance points where estimated PM equity < threshold. Sells still execute.",
                disabled=margin_disabled,
                key="pm_buy_block",
            )
            config['pm_buy_block'] = pm_buy_block
            if pm_buy_block:
                config['pm_buy_block_threshold'] = utils.num_input("PM Buy Block Threshold ($)", "pm_buy_block_threshold", 100000.0, 1000.0, disabled=margin_disabled)
            else:
                config['pm_buy_block_threshold'] = 100000.0

            # Backward compat
            config['pm_enabled'] = pm_mode != "Off"

        # Save margin settings to stable backup
        _backup = {}
        for _k in _MARGIN_WIDGET_KEYS:
            if _k in st.session_state:
                _backup[_k] = st.session_state[_k]
        st.session_state._margin_store = _backup

    with tab_margin:
        _margin_fragment()

    with tab_asset:
        # Asset Explorer is self-contained.
        # It doesn't modify the simulation config, just visualizes data.
        asset_explorer.render_asset_explorer()

    with tab_ndx:
        # NDX-100 Moving Average Scanner
        ndx_scanner.render_ndx_scanner()

    @st.fragment
    def _settings_fragment():
        c1, c2 = st.columns(2)
        with c1:
            config['chart_style'] = st.selectbox(
                "Chart Style",
                ["Classic (Combined)", "Candlestick"],
                index=0
            )
            config['timeframe'] = st.selectbox(
                "Chart Timeframe",
                ["1D", "1W", "1M", "3M", "1Y"],
                index=2
            )
            config['log_scale'] = st.checkbox("Logarithmic Scale", value=True)
        with c2:
            config['show_range_slider'] = st.checkbox("Show Range Slider", value=True)
            config['show_volume'] = st.checkbox("Show Range/Volume Panel", value=True)
            
            st.markdown("---")
            st.markdown("**Cache Management**")
            if st.button("🧹 Clear Runtime Caches", help="Clears Streamlit and in-memory app caches only. Leaves everything under data/ untouched."):
                summary = clear_runtime_caches()
                st.success(
                    "Runtime caches cleared "
                    f"(data={summary['streamlit_data']}, resources={summary['streamlit_resource']}, "
                    f"session={summary['session_keys_removed']})."
                )

    with tab_settings:
        _settings_fragment()

    return config
