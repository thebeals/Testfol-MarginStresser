import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

from app.common.utils import color_return
from app.core import calculations
from app.services.live_prices import build_live_returns_snapshot, fetch_yahoo_live_price_series
from app.ui.charts.stats_helpers import excess_kurtosis, skewness


def _resample_returns(series, rule):
    """Resample series and compute period returns, including the first period."""
    resampled = series.resample(rule).last()
    ret = resampled.pct_change(fill_method=None)
    # First period has no prior — compute return from actual series start
    if len(resampled) > 0 and not ret.empty:
        first_val = series.iloc[0]
        if first_val != 0:
            ret.iloc[0] = (resampled.iloc[0] / first_val) - 1
    return ret.dropna()


def _period_timeline_state(series, rule, selected_period_end=None):
    """Return a series/return slice through the selected timeline point."""
    working_series = pd.Series(series).dropna()
    if working_series.empty:
        return working_series, pd.Series(dtype=float), [], None

    if not isinstance(working_series.index, pd.DatetimeIndex):
        working_series.index = pd.to_datetime(working_series.index)
    working_series = working_series.sort_index()

    full_period_ret = _resample_returns(working_series, rule)
    timeline_options = list(full_period_ret.index)
    if not timeline_options:
        return working_series.iloc[0:0], full_period_ret, [], None

    selected_end = pd.Timestamp(selected_period_end) if selected_period_end is not None else timeline_options[-1]
    visible_options = [option for option in timeline_options if option <= selected_end]
    selected_end = visible_options[-1] if visible_options else timeline_options[0]

    visible_series = working_series.loc[:selected_end]
    visible_period_ret = _resample_returns(visible_series, rule)
    return visible_series, visible_period_ret, timeline_options, selected_end


def _monthly_timeline_state(series, selected_month_end=None):
    return _period_timeline_state(series, "ME", selected_month_end)


def _quarterly_timeline_state(series, selected_quarter_end=None):
    return _period_timeline_state(series, "QE", selected_quarter_end)


def _format_month_timeline_label(month_end):
    return pd.Timestamp(month_end).strftime("%b %Y")


def _format_quarter_timeline_label(quarter_end):
    quarter_end = pd.Timestamp(quarter_end)
    return f"Q{quarter_end.quarter} {quarter_end.year}"


@st.cache_data(ttl=45, show_spinner=False)
def _cached_fetch_live_price_series(symbols: tuple[str, ...], reference_date: str):
    return fetch_yahoo_live_price_series(symbols, pd.Timestamp(reference_date))


def _streamlit_live_price_fetcher(symbols: tuple[str, ...], reference_date: pd.Timestamp):
    return _cached_fetch_live_price_series(
        tuple(symbols),
        pd.Timestamp(reference_date).strftime("%Y-%m-%d"),
    )

def render_cheat_sheet(port_series, portfolio_name, unique_id, component_data=None):
    # Determine target series and name
    target_series = port_series
    target_name = portfolio_name
    
    if component_data is not None and not component_data.empty:
        if isinstance(component_data, pd.DataFrame):
            cols = list(component_data.columns)
            if len(cols) == 1:
                    target_name = cols[0]
                    target_series = component_data[target_name]
            elif len(cols) > 1:
                    target_name = st.selectbox("Select Asset to Analyze", cols, key=f"cs_sel_{unique_id}")
                    target_series = component_data[target_name]
                    
    # Fetch OHLC for Pivot Points
    ohlc_data = None
    try:
            from app.core.shadow_backtest import parse_ticker
            import yfinance as yf
            
            mapped_ticker, _ = parse_ticker(target_name)
            if not target_series.empty:
                last_dt = target_series.index[-1]
                start_f = (last_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                end_f = (last_dt + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
                
                base_yf = yf.Ticker(mapped_ticker)
                hist_ohlc = base_yf.history(start=start_f, end=end_f, auto_adjust=True, timeout=10)
                
                if hist_ohlc.index.tz is not None:
                    hist_ohlc.index = hist_ohlc.index.tz_localize(None)

                if not hist_ohlc.empty:
                    if last_dt in hist_ohlc.index:
                        ref_bar = hist_ohlc.loc[last_dt]
                    else:
                        ref_bar = hist_ohlc.iloc[-1]
                    
                    ohlc_data = {
                        'High': ref_bar['High'], 
                        'Low': ref_bar['Low'], 
                        'Close': ref_bar['Close']
                    }
    except Exception:
            pass
            
    st.subheader(f"{target_name} Trader's Cheat Sheet")
    cs_df = calculations.calculate_cheat_sheet(target_series, ohlc_data=ohlc_data)
    
    if cs_df is None or cs_df.empty:
            st.info("Insufficient data for Technical Analysis.")
    else:
            # Transform to 3-Column Layout (Barchart Style)
            # Left Column: High/Low, StdDev, Pivot S/R, Session Levels
            # Right Column: Moving Averages, Fibonacci, Pivot Point (P)
            
            left_types = ["High/Low", "StdDev", "Pivot Support", "Pivot Resistance", "Session Level"]
            
            display_rows = []
            for _, row in cs_df.iterrows():
                lbl = row['Label']
                typ = row['Type']
                px = row['Price']
                
                # Format Price as String immediately to allow centering (Numeric defaults to Right)
                px_str = "{:,.2f}".format(px)
                
                if typ == "Current":
                    display_rows.append({"Support/Resistance Levels": "Latest", "Price": px_str, "Key Turning Points": "Latest", "Type": "Current"})
                elif typ in left_types:
                    display_rows.append({"Support/Resistance Levels": lbl, "Price": px_str, "Key Turning Points": "", "Type": "Left"})
                else:
                    display_rows.append({"Support/Resistance Levels": "", "Price": px_str, "Key Turning Points": lbl, "Type": "Right"})
        
            disp_df = pd.DataFrame(display_rows)
            
            # Identify Current Price Row Index for coloring
            try:
                curr_idx = disp_df[disp_df["Type"] == "Current"].index[0]
            except IndexError:
                curr_idx = -1

            # Dark Mode Toggle
            is_dark = st.toggle("Dark Mode Colors", value=True, key=f"cs_dark_{unique_id}")
            
            if is_dark:
                # Dark Mode Palette
                c_res_bg = "#4a1c1c" # Dark Red BG
                c_res_txt = "#ffcdd2" # Light Red Text
                c_sup_bg = "#1b3e20" # Dark Green BG
                c_sup_txt = "#c8e6c9" # Light Green Text
                c_neutral_bg = "#262730" # Streamlit Secondary Dark BG
                c_neutral_txt = "#fafafa" # White Text
                c_current_bg = "#FFD700"
                c_current_txt = "black"
            else:
                # Light Mode Palette (Barchart)
                c_res_bg = "#FFEBEE"
                c_res_txt = "#B71C1C"
                c_sup_bg = "#E8F5E9"
                c_sup_txt = "#1B5E20"
                c_neutral_bg = "white"
                c_neutral_txt = "#333333"
                c_current_bg = "#FFD700"
                c_current_txt = "black"

            def style_barchart(row):
                idx = row.name
                styles = []

                if curr_idx == -1:
                    return [""] * len(row)

                # Determine Active Colors based on Row Position
                if idx < curr_idx: # Resistance
                    bg_active = c_res_bg
                    txt_active = c_res_txt
                    weight = "600"
                elif idx > curr_idx: # Support
                    bg_active = c_sup_bg
                    txt_active = c_sup_txt
                    weight = "600"
                else: # Current
                    # Current Row is special: Full Gold
                    return [f'background-color: {c_current_bg}; color: {c_current_txt}; font-weight: bold; text-align: center !important; vertical-align: middle;'] * len(row)

                # Default (Empty/Price) Style
                bg_neutral = c_neutral_bg
                txt_neutral = c_neutral_txt
                
                # Helper to check if string is non-empty
                def is_populated(val):
                    return bool(val and str(val).strip())

                # Col 0: Support/Resistance Levels (Left) => Right Align
                # Using iloc to access by position is safer if col names change slightly
                val_left = row.iloc[0] 
                if is_populated(val_left):
                    s_left = f'background-color: {bg_active}; color: {txt_active}; font-weight: {weight};'
                else:
                    s_left = f'background-color: {bg_neutral}; color: {txt_neutral};'
                styles.append(f'{s_left} text-align: right !important; padding-right: 15px; vertical-align: middle;')
                
                # Col 1: Price (Center) => Center Align
                styles.append(f'background-color: {bg_neutral}; color: {txt_neutral}; font-weight: normal; text-align: center !important; vertical-align: middle;')
                
                # Col 2: Key Turning Points (Right) => Left Align
                val_right = row.iloc[2]
                if is_populated(val_right):
                    s_right = f'background-color: {bg_active}; color: {txt_active}; font-weight: {weight};'
                else:
                    s_right = f'background-color: {bg_neutral}; color: {txt_neutral};'
                styles.append(f'{s_right} text-align: left !important; padding-left: 15px; vertical-align: middle;')
                
                return styles

            # Display
            final_view = disp_df.drop(columns=["Type"])
            
            st.dataframe(
                final_view.style.apply(style_barchart, axis=1), 
                use_container_width=True, 
                height=(len(final_view) + 1) * 35,
                hide_index=True
            )
            
            # Legend and Explanation
            st.markdown("""
            <div style="margin-top: 20px; font-size: 0.9em; color: #888;">
            <p><strong>Standard deviation</strong> is calculated using the closing price over the past 20-periods. To calculate standard deviation:</p>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li><strong>Step 1:</strong> Average = Calculate the average closing price over the past 20-days.</li>
                <li><strong>Step 2:</strong> Difference = Calculate the variance from the Average for each Price.</li>
                <li><strong>Step 3:</strong> Square the variance of each data point.</li>
                <li><strong>Step 4:</strong> Sum of the squared variance value.</li>
                <li><strong>Step 5:</strong> For Standard Deviation 2 multiple the result by 2. For Standard Deviation 3 multiple the result by 3.</li>
                <li><strong>Step 6:</strong> Divide the result by the number of data points in the series less 1.</li>
                <li><strong>Step 7:</strong> The final result is the Square root of the result of Step 6.</li>
            </ul>
            
            <p><strong>Legend:</strong></p>
            <ul style="list-style-type: none; padding-left: 10px;">
                <li><span style="color: #E8F5E9; background-color: #E8F5E9; border: 1px solid #ccc;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <strong>Green areas below the Last Price</strong> will tend to provide support to limit the downward move.</li>
                <li><span style="color: #FFEBEE; background-color: #FFEBEE; border: 1px solid #ccc;">&nbsp;&nbsp;&nbsp;&nbsp;</span> <strong>Red areas above the Last Price</strong> will tend to provide resistance to limit the upward move.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

def render_returns_analysis(port_series, bench_series=None, comparison_series=None, unique_id="", portfolio_name="Strategy", component_data=None, raw_port_series=None, stats=None, raw_response=None, fresh_yearly=None, fresh_series=None, allocation=None, composition_df=None):
    # Toggle: use fresh-start series for period breakdowns
    has_fresh_data = fresh_yearly and len(fresh_yearly) > 0 and fresh_series is not None
    use_fresh = False
    if has_fresh_data:
        use_fresh = st.toggle("Use Fresh Start returns for period breakdowns", value=True, key=f"fresh_toggle_{unique_id}")

    active_series = fresh_series if use_fresh else port_series

    # --- DISTRIBUTION HELPER ---
    def render_distribution(ret_series, period_label, freq_label):
        """Render histogram + summary stats for a returns series."""
        if ret_series.empty or len(ret_series) < 2:
            return

        vals = ret_series.values
        median_val = np.nanmedian(vals)

        st.subheader(f"{period_label} Returns Histogram")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vals * 100,
            nbinsx=max(10, min(50, len(vals) // 2)),
            marker_color="#42A5F5",
            name=f"Frequency of {freq_label}",
        ))
        fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=1)
        fig.add_vline(
            x=median_val * 100, line_dash="dash", line_color="#FFC107", line_width=2,
            annotation_text=f"Median: {median_val*100:.2f}%",
            annotation_position="top",
            annotation_font_color="#FFC107",
        )
        fig.update_layout(
            xaxis_title=f"{period_label} Return",
            yaxis_title=f"Frequency of {freq_label}",
            template="plotly_dark",
            height=400,
            showlegend=False,
            bargap=0.05,
            xaxis_tickformat=".2f",
            xaxis_ticksuffix="%",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Statistics")
        pcts = [1, 5, 25, 50, 75, 95, 99]
        pct_vals = np.nanpercentile(vals, pcts)
        stat_rows = {
            "Minimum": f"{np.nanmin(vals)*100:.2f}%",
            "1st Percentile": f"{pct_vals[0]*100:.2f}%",
            "5th Percentile": f"{pct_vals[1]*100:.2f}%",
            "25th Percentile": f"{pct_vals[2]*100:.2f}%",
            "50th Percentile (Median)": f"{pct_vals[3]*100:.2f}%",
            "75th Percentile": f"{pct_vals[4]*100:.2f}%",
            "95th Percentile": f"{pct_vals[5]*100:.2f}%",
            "99th Percentile": f"{pct_vals[6]*100:.2f}%",
            "Maximum": f"{np.nanmax(vals)*100:.2f}%",
            "Mean": f"{np.nanmean(vals)*100:.2f}%",
            "Std Deviation": f"{np.nanstd(vals, ddof=1)*100:.2f}%",
            "Skewness": f"{skewness(vals):.3f}",
            "Excess Kurtosis": f"{excess_kurtosis(vals):.3f}",
        }
        col_name = f"{portfolio_name} {period_label} Returns"
        df_stats = pd.DataFrame(
            {"Statistic": stat_rows.keys(), col_name: stat_rows.values()}
        )
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # --- HEATMAP HELPERS ---
    def render_seasonal_summary(series, suffix=""):
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty: return
        
        # 1. Prepare Data (Same as Monthly View)
        m_ret = _resample_returns(series, "ME")
        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month

        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # 2. Add Yearly Return Column
        yearly_col = []
        years = pivot.index
        for y in years:
            row = pivot.loc[y]
            ret = (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)

        # Add Yearly column to pivot for display/calculation
        display_pivot = pivot.copy()
        display_pivot.columns = month_names
        # display_pivot["Yearly Return"] = yearly_col # Removed per user request
        
        # 3. Calculate Summary Statistics Table
        st.subheader("Summary")
        
        # Create a DF for stats calc: Join pivot (months) and yearly_col
        stats_source = display_pivot.copy() # Columns: Jan..Dec, Yearly Return
        
        # Define rows
        stats_rows = ["Average", "Avg YTD", "Now YTD", "% Positive", "% Negative", "Median", "Best", "Worst", "Abs Average", "Abs Best", "Abs Worst"]
        stats_df = pd.DataFrame(index=stats_rows, columns=stats_source.columns)

        ytd_by_month = pd.DataFrame(index=stats_source.index, columns=stats_source.columns, dtype=float)
        for month_idx, col in enumerate(stats_source.columns):
            through_month = stats_source.iloc[:, :month_idx + 1]
            complete_years = through_month.dropna(how="any")
            if complete_years.empty:
                continue
            ytd_by_month.loc[complete_years.index, col] = (1 + complete_years).prod(axis=1) - 1
        latest_years = ytd_by_month.dropna(how="all")
        latest_year = int(latest_years.index.max()) if not latest_years.empty else None
        
        for col in stats_source.columns:
            s_data = stats_source[col].dropna()
            if s_data.empty:
                continue
                
            stats_df.loc["Average", col] = s_data.mean()
            stats_df.loc["Avg YTD", col] = ytd_by_month[col].dropna().mean()
            if latest_year is not None:
                stats_df.loc["Now YTD", col] = ytd_by_month.loc[latest_year, col]
            stats_df.loc["% Positive", col] = (s_data > 0).mean()
            stats_df.loc["% Negative", col] = (s_data < 0).mean()
            stats_df.loc["Median", col] = s_data.median()
            stats_df.loc["Best", col] = s_data.max()
            stats_df.loc["Worst", col] = s_data.min()
            stats_df.loc["Abs Average", col] = s_data.abs().mean()
            stats_df.loc["Abs Best", col] = s_data.abs().max()
            stats_df.loc["Abs Worst", col] = s_data.abs().min()
            
        stats_df = stats_df.astype(float)
        
        return_rows = ["Average", "Avg YTD", "Now YTD", "Median", "Best", "Worst"]
        pct_rows = ["% Positive", "% Negative"]
        abs_rows = ["Abs Average", "Abs Best", "Abs Worst"]
        
        def style_summary(df):
            return df.style.format("{:.2%}", na_rep="") \
                .map(color_return, subset=pd.IndexSlice[return_rows, :]) \
                .background_gradient(cmap="Greens", subset=pd.IndexSlice["% Positive", :], vmin=0, vmax=1) \
                .background_gradient(cmap="Reds", subset=pd.IndexSlice["% Negative", :], vmin=0, vmax=1) \
                .background_gradient(cmap="Oranges", subset=pd.IndexSlice[abs_rows, :])

        st.dataframe(style_summary(stats_df), use_container_width=True)
        now_caption = f" Now YTD uses {latest_year}." if latest_year is not None else ""
        st.caption(
            "Avg YTD compounds January through each column month, then averages those same-month returns across years."
            + now_caption
        )

    # --- HEATMAP HELPERS ---
    def render_quarterly_returns_view(series, suffix=""):
        # Combine unique_id with suffix for truly unique keys
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty:
            return None

        visible_series, quarterly_ret, timeline_options, selected_timeline_end = _quarterly_timeline_state(series)
        if quarterly_ret.empty:
            st.info("Not enough quarterly data for the returns heatmap.")
            return None

        if len(timeline_options) > 1:
            selected_timeline_end = st.select_slider(
                "Timeline",
                options=timeline_options,
                value=selected_timeline_end,
                format_func=_format_quarter_timeline_label,
                key=f"quarterly_timeline_{full_suffix}",
            )
            visible_series, quarterly_ret, _, selected_timeline_end = _quarterly_timeline_state(
                series,
                selected_timeline_end,
            )

        if selected_timeline_end is not None:
            st.caption(
                f"Through {_format_quarter_timeline_label(selected_timeline_end)} | "
                f"{len(quarterly_ret):,} quarter{'s' if len(quarterly_ret) != 1 else ''}"
            )

        q_ret = quarterly_ret.to_frame(name="Return")
        q_ret["Year"] = q_ret.index.year
        q_ret["Quarter"] = q_ret.index.quarter
        q_ret["Quarter Name"] = "Q" + q_ret["Quarter"].astype(str)

        pivot = q_ret.pivot(index="Year", columns="Quarter", values="Return")
        for i in range(1, 5):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)

        quarter_names = ["Q1", "Q2", "Q3", "Q4"]
        quarterly_avgs = pivot.mean()
        z_data = pivot.values
        z_avgs = quarterly_avgs.values.reshape(1, -1)
        z_combined_main = np.concatenate([z_data, z_avgs], axis=0)

        years = pivot.index
        yearly_col = []
        for y in years:
            row = pivot.loc[y]
            ret = float("nan") if row.isna().all() else (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        # Fresh-start yearly column (suppress when toggle is on — whole table is already fresh)
        has_fresh = fresh_yearly and len(fresh_yearly) > 0 and not use_fresh
        yearly_label = "Continuous" if has_fresh else "Yearly"
        if has_fresh:
            fresh_col = [fresh_yearly.get(y, float("nan")) for y in years]
            fresh_avg = np.nanmean(fresh_col) if len(fresh_col) > 0 else float("nan")
            z_combined_fresh = np.array(fresh_col + [fresh_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]

        # Scaling
        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 2 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        if has_fresh:
            std_dev_fresh = np.nanstd(z_combined_fresh * 100)
            scale_fresh = 2.0 * std_dev_fresh if not np.isnan(std_dev_fresh) else scale_yearly
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]

        # Hover Text
        date_map = {}
        try:
            periods = visible_series.index.to_period("Q")
            for p, dates in visible_series.index.groupby(periods).items():
                if not dates.empty:
                    date_map[(p.year, p.quarter)] = f"{dates.min().strftime('%b %d')} - {dates.max().strftime('%b %d')}"
        except (AttributeError, TypeError, ValueError):
            pass

        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(2)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(quarter_names):
                val = z_rounded_main[i][j]
                if np.isnan(val): row_txt.append("")
                elif row_label == "Average": row_txt.append(f"Average<br>{col_label}: {val:+.2f}%")
                else:
                    dr = date_map.get((int(row_label), j+1), "") if row_label.isdigit() else ""
                    dr_str = f"<br>{dr}" if dr else ""
                    row_txt.append(f"Year: {row_label}<br>{col_label}: {val:+.2f}%{dr_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(2)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.2f}%"
            if row_label == "Average": hover_yearly.append([f"Average Annual<br>{val_str}"])
            else: hover_yearly.append([f"Year: {row_label}<br>{yearly_label}: {val_str}"])

        if has_fresh:
            hover_fresh = []
            z_rounded_fresh = (z_combined_fresh * 100).round(2)
            for i, row_label in enumerate(y_labels):
                val = z_rounded_fresh[i][0]
                val_str = "" if np.isnan(val) else f"{val:+.2f}%"
                if row_label == "Average": hover_fresh.append([f"Average Fresh<br>{val_str}"])
                else: hover_fresh.append([f"Year: {row_label}<br>Fresh Start: {val_str}"])

        # Plot
        n_years = len(y_labels) - 1
        n_cols = 3 if has_fresh else 2
        col_widths = [0.75, 0.125, 0.125] if has_fresh else [0.85, 0.15]
        fig = make_subplots(rows=2, cols=n_cols, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=col_widths,
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )

        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(2), x=quarter_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(2), x=[yearly_label], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        fig.add_trace(go.Heatmap(z=(z_combined_main[-1:] * 100).round(2), x=quarter_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[-1:] * 100).round(2), x=[yearly_label], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        if has_fresh:
            fig.add_trace(go.Heatmap(z=(z_combined_fresh[:-1] * 100).round(2), x=["Fresh Start"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_fresh, zmax=scale_fresh, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_fresh[:-1], showscale=False), row=1, col=3)
            fig.add_trace(go.Heatmap(z=(z_combined_fresh[-1:] * 100).round(2), x=["Fresh Start"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_fresh, zmax=scale_fresh, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_fresh[-1:], showscale=False), row=2, col=3)

        fig.update_layout(title="Quarterly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        if has_fresh:
            fig.update_yaxes(showticklabels=False, col=3)
        st.plotly_chart(fig, use_container_width=True, key=f"q_hm_{full_suffix}")
        
        st.subheader("Quarterly Returns List")
        quarterly_bal = visible_series.resample("QE").last()
        df_quarterly_list = q_ret.copy()
        df_quarterly_list["Period"] = df_quarterly_list.index.to_period("Q").astype(str)
        df_quarterly_list["Balance"] = quarterly_bal.reindex(df_quarterly_list.index).values
        df_quarterly_list = df_quarterly_list[["Period", "Return", "Balance"]].sort_index(ascending=False)

        st.dataframe(
            df_quarterly_list.style.format({"Return": "{:+.1%}", "Balance": "${:,.2f}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True
        )
        return visible_series, quarterly_ret, selected_timeline_end

    def render_monthly_returns_view(series, suffix=""):
        # Combine unique_id with suffix for truly unique keys
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty:
            return None

        visible_series, m_ret, timeline_options, selected_timeline_end = _monthly_timeline_state(series)
        if m_ret.empty:
            st.info("Not enough monthly data for the returns heatmap.")
            return None

        if len(timeline_options) > 1:
            selected_timeline_end = st.select_slider(
                "Timeline",
                options=timeline_options,
                value=selected_timeline_end,
                format_func=_format_month_timeline_label,
                key=f"monthly_timeline_{full_suffix}",
            )
            visible_series, m_ret, _, selected_timeline_end = _monthly_timeline_state(
                series,
                selected_timeline_end,
            )

        if selected_timeline_end is not None:
            st.caption(
                f"Through {_format_month_timeline_label(selected_timeline_end)} | "
                f"{len(m_ret):,} month{'s' if len(m_ret) != 1 else ''}"
            )

        df_monthly = m_ret.to_frame(name="Return")
        df_monthly["Year"] = df_monthly.index.year
        df_monthly["Month"] = df_monthly.index.month

        pivot = df_monthly.pivot(index="Year", columns="Month", values="Return")
        for i in range(1, 13):
            if i not in pivot.columns: pivot[i] = float("nan")
        pivot = pivot.sort_index(ascending=True)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avgs = pivot.mean()

        z_data = pivot.values
        z_combined_main = np.concatenate([z_data, monthly_avgs.values.reshape(1, -1)], axis=0)

        years = pivot.index
        yearly_col = []
        for y in years:
            row = pivot.loc[y]
            ret = float("nan") if row.isna().all() else (1 + row.fillna(0)).prod() - 1
            yearly_col.append(ret)
        yearly_avg = np.nanmean(yearly_col) if len(yearly_col) > 0 else float("nan")
        z_combined_yearly = np.array(yearly_col + [yearly_avg]).reshape(-1, 1)

        # Fresh-start yearly column (suppress when toggle is on — whole table is already fresh)
        has_fresh = fresh_yearly and len(fresh_yearly) > 0 and not use_fresh
        yearly_label = "Continuous" if has_fresh else "Yearly"
        if has_fresh:
            fresh_col = [fresh_yearly.get(y, float("nan")) for y in years]
            fresh_avg = np.nanmean(fresh_col) if len(fresh_col) > 0 else float("nan")
            z_combined_fresh = np.array(fresh_col + [fresh_avg]).reshape(-1, 1)

        y_labels = [str(y) for y in pivot.index] + ["Average"]

        std_dev_main = np.nanstd(z_combined_main * 100)
        scale_main = 2 * std_dev_main if not np.isnan(std_dev_main) else 10
        std_dev_yearly = np.nanstd(z_combined_yearly * 100)
        scale_yearly = 2.0 * std_dev_yearly if not np.isnan(std_dev_yearly) else 10
        if has_fresh:
            std_dev_fresh = np.nanstd(z_combined_fresh * 100)
            scale_fresh = 2.0 * std_dev_fresh if not np.isnan(std_dev_fresh) else scale_yearly
        colorscale_heatmap = [[0, '#E53935'], [0.5, '#FFFFFF'], [1, '#43A047']]

        hover_main = []
        z_rounded_main = (z_combined_main * 100).round(2)
        for i, row_label in enumerate(y_labels):
            row_txt = []
            for j, col_label in enumerate(month_names):
                val = z_rounded_main[i][j]
                val_str = "" if np.isnan(val) else f"{val:+.2f}%"
                row_txt.append(f"{row_label} {col_label}<br>{val_str}")
            hover_main.append(row_txt)

        hover_yearly = []
        z_rounded_yearly = (z_combined_yearly * 100).round(2)
        for i, row_label in enumerate(y_labels):
            val = z_rounded_yearly[i][0]
            val_str = "" if np.isnan(val) else f"{val:+.2f}%"
            hover_yearly.append([f"{row_label} {yearly_label}<br>{val_str}"])

        if has_fresh:
            hover_fresh = []
            z_rounded_fresh = (z_combined_fresh * 100).round(2)
            for i, row_label in enumerate(y_labels):
                val = z_rounded_fresh[i][0]
                val_str = "" if np.isnan(val) else f"{val:+.2f}%"
                hover_fresh.append([f"{row_label} Fresh Start<br>{val_str}"])

        n_years = len(y_labels) - 1
        n_cols = 3 if has_fresh else 2
        col_widths = [0.75, 0.125, 0.125] if has_fresh else [0.85, 0.15]
        fig = make_subplots(rows=2, cols=n_cols, shared_xaxes=True, shared_yaxes=True,
            horizontal_spacing=0.03, vertical_spacing=0.02, column_widths=col_widths,
            row_heights=[n_years/(n_years+1), 1/(n_years+1)]
        )

        fig.add_trace(go.Heatmap(z=(z_combined_main[:-1] * 100).round(2), x=month_names, y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_main, zmax=scale_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[:-1], showscale=False), row=1, col=1)
        fig.add_trace(go.Heatmap(z=(z_combined_yearly[:-1] * 100).round(2), x=[yearly_label], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_yearly, zmax=scale_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[:-1], showscale=False), row=1, col=2)
        # Compute tighter color scale for the average row
        avg_monthly_vals = z_combined_main[-1:] * 100
        avg_abs_max = np.nanmax(np.abs(avg_monthly_vals))
        scale_avg_main = max(avg_abs_max * 1.2, 0.5) if not np.isnan(avg_abs_max) else scale_main
        avg_yearly_val = z_combined_yearly[-1:] * 100
        avg_yearly_abs = np.nanmax(np.abs(avg_yearly_val))
        scale_avg_yearly = max(avg_yearly_abs * 1.2, 0.5) if not np.isnan(avg_yearly_abs) else scale_yearly

        fig.add_trace(go.Heatmap(z=avg_monthly_vals.round(2), x=month_names, y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_avg_main, zmax=scale_avg_main, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_main[-1:], showscale=False), row=2, col=1)
        fig.add_trace(go.Heatmap(z=avg_yearly_val.round(2), x=[yearly_label], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_avg_yearly, zmax=scale_avg_yearly, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_yearly[-1:], showscale=False), row=2, col=2)

        if has_fresh:
            fig.add_trace(go.Heatmap(z=(z_combined_fresh[:-1] * 100).round(2), x=["Fresh Start"], y=y_labels[:-1], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_fresh, zmax=scale_fresh, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_fresh[:-1], showscale=False), row=1, col=3)
            avg_fresh_val = z_combined_fresh[-1:] * 100
            avg_fresh_abs = np.nanmax(np.abs(avg_fresh_val))
            scale_avg_fresh = max(avg_fresh_abs * 1.2, 0.5) if not np.isnan(avg_fresh_abs) else scale_fresh
            fig.add_trace(go.Heatmap(z=avg_fresh_val.round(2), x=["Fresh Start"], y=y_labels[-1:], colorscale=colorscale_heatmap, zmid=0, zmin=-scale_avg_fresh, zmax=scale_avg_fresh, texttemplate="%{z:+.2f}%", hoverinfo="text", hovertext=hover_fresh[-1:], showscale=False), row=2, col=3)

        fig.update_layout(title="Monthly Returns Heatmap (%)", template="plotly_white", height=max(400, (len(y_labels)+1)*30), yaxis=dict(autorange="reversed", type="category"), yaxis3=dict(autorange="reversed", type="category"))
        fig.update_yaxes(showticklabels=False, col=2)
        if has_fresh:
            fig.update_yaxes(showticklabels=False, col=3)
        st.plotly_chart(fig, use_container_width=True, key=f"m_hm_{full_suffix}")
        return visible_series, m_ret, selected_timeline_end

    def render_year_drilldown_view(series, suffix=""):
        full_suffix = f"{unique_id}_{suffix}" if unique_id else suffix
        if series.empty:
            return

        working_series = pd.Series(series).dropna()
        if working_series.empty:
            return
        if not isinstance(working_series.index, pd.DatetimeIndex):
            working_series.index = pd.to_datetime(working_series.index)
        working_series = working_series.sort_index()

        monthly_ret = _resample_returns(working_series, "ME")
        if monthly_ret.empty:
            st.info("Not enough monthly data for a year drilldown.")
            return

        monthly_bal = working_series.resample("ME").last().reindex(monthly_ret.index)
        monthly_df = monthly_ret.to_frame(name="Monthly Return")
        monthly_df["Ending Balance"] = monthly_bal.values
        monthly_df["Year"] = monthly_df.index.year
        monthly_df["Month"] = monthly_df.index.strftime("%b")
        monthly_df["Month Number"] = monthly_df.index.month

        year_options = [int(y) for y in sorted(monthly_df["Year"].dropna().unique())]
        if not year_options:
            st.info("Not enough monthly data for a year drilldown.")
            return

        default_year = year_options[-1]
        selected_year = st.select_slider(
            "Year",
            options=year_options,
            value=default_year,
            key=f"year_drilldown_{full_suffix}",
        )

        year_df = monthly_df[monthly_df["Year"] == selected_year].copy()
        year_df = year_df.sort_values("Month Number")
        if year_df.empty:
            st.info(f"No monthly returns are available for {selected_year}.")
            return

        growth = 1.0 + year_df["Monthly Return"].fillna(0.0)
        starting_factor = growth.cumprod().shift(1, fill_value=1.0)
        year_df["Compounded Contribution"] = starting_factor * year_df["Monthly Return"].fillna(0.0)
        year_df["Cumulative Return"] = year_df["Compounded Contribution"].cumsum()
        year_df["Starting Factor"] = starting_factor

        end_balance = float(year_df["Ending Balance"].iloc[-1])
        first_growth = growth.iloc[0]
        start_balance = (
            float(year_df["Ending Balance"].iloc[0] / first_growth)
            if first_growth and np.isfinite(first_growth)
            else np.nan
        )
        ending_return = float((growth.prod() - 1.0))
        balance_change = end_balance - start_balance if np.isfinite(start_balance) else np.nan

        best_idx = year_df["Monthly Return"].idxmax()
        worst_idx = year_df["Monthly Return"].idxmin()
        best_label = f"{year_df.loc[best_idx, 'Month']} {year_df.loc[best_idx, 'Monthly Return']:+.2%}"
        worst_label = f"{year_df.loc[worst_idx, 'Month']} {year_df.loc[worst_idx, 'Monthly Return']:+.2%}"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ending Return", f"{ending_return:+.2%}")
        m2.metric("Start Value", f"${start_balance:,.0f}" if np.isfinite(start_balance) else "N/A")
        m3.metric(
            "End Value",
            f"${end_balance:,.0f}",
            f"${balance_change:+,.0f}" if np.isfinite(balance_change) else None,
        )
        m4.metric("Best / Worst Month", best_label, worst_label)

        chart_df = year_df.copy()
        chart_df["Contribution Label"] = chart_df["Compounded Contribution"].map(lambda x: f"{x*100:+.2f}%")
        chart_df["Cumulative Label"] = chart_df["Cumulative Return"].map(lambda x: f"{x*100:+.2f}%")
        bar_text = chart_df["Contribution Label"].tolist()
        if bar_text:
            bar_text[0] = ""

        cumulative_label_annotations = []
        for idx, row in chart_df.reset_index(drop=True).iterrows():
            cumulative_y = float(row["Cumulative Return"] * 100)
            contribution_y = float(row["Compounded Contribution"] * 100)
            inside_positive_bar = contribution_y > 0 and 0 <= cumulative_y <= contribution_y
            inside_negative_bar = contribution_y < 0 and contribution_y <= cumulative_y <= 0
            yshift = -20 if inside_positive_bar else (20 if inside_negative_bar else 14)
            cumulative_label_annotations.append(dict(
                x=row["Month"],
                y=cumulative_y,
                text=row["Cumulative Label"],
                showarrow=False,
                yshift=yshift,
                font=dict(color="#F8FAFC", size=12),
                bgcolor="rgba(15, 18, 28, 0.82)",
                bordercolor="rgba(248, 250, 252, 0.28)",
                borderpad=3,
            ))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_df["Month"],
            y=chart_df["Compounded Contribution"] * 100,
            name="Compounded Monthly Contribution",
            marker_color=["#00CC96" if x >= 0 else "#EF553B" for x in chart_df["Compounded Contribution"]],
            text=bar_text,
            textposition="auto",
            customdata=np.stack([
                (chart_df["Monthly Return"] * 100).map(lambda x: f"{x:+.2f}%"),
                chart_df["Starting Factor"],
                chart_df["Cumulative Label"],
            ], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Monthly return: %{customdata[0]}<br>"
                "Starting factor: %{customdata[1]:.4f}x<br>"
                "Compounded contribution: %{text}<br>"
                "Cumulative return: %{customdata[2]}"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=chart_df["Month"],
            y=chart_df["Cumulative Return"] * 100,
            name="Cumulative Year Return",
            mode="lines+markers",
            line=dict(color="#42A5F5", width=3),
            marker=dict(size=8),
            customdata=chart_df["Cumulative Label"],
            hovertemplate="<b>%{x}</b><br>Cumulative return: %{customdata}<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.45)")
        fig.update_layout(
            title=f"{selected_year} Return Build-Up",
            yaxis_title="Return vs. Start of Year (%)",
            xaxis_title="Month",
            template="plotly_dark",
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            annotations=cumulative_label_annotations,
            margin=dict(t=80),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"year_drill_fig_{full_suffix}")

        detail_df = chart_df[[
            "Month",
            "Monthly Return",
            "Starting Factor",
            "Compounded Contribution",
            "Cumulative Return",
            "Ending Balance",
        ]].copy()
        st.dataframe(
            detail_df.style.format({
                "Monthly Return": "{:+.2%}",
                "Starting Factor": "{:.4f}x",
                "Compounded Contribution": "{:+.2%}",
                "Cumulative Return": "{:+.2%}",
                "Ending Balance": "${:,.2f}",
            }).map(
                color_return,
                subset=["Monthly Return", "Compounded Contribution", "Cumulative Return"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    def render_live_returns_view():
        st.subheader(f"{portfolio_name} Live Returns")

        refresh_col, source_col = st.columns([1, 3])
        if refresh_col.button("Refresh Live Prices", key=f"live_refresh_{view_key}"):
            _cached_fetch_live_price_series.clear()
        source_col.caption(
            "Yahoo Finance live quotes with pre-market/post-market when available. "
            "SIM sleeves use live proxy tickers."
        )

        with st.spinner("Fetching live prices..."):
            snapshot = build_live_returns_snapshot(
                port_series,
                allocation=allocation,
                composition_df=composition_df,
                price_fetcher=_streamlit_live_price_fetcher,
            )

        if not snapshot.get("ok"):
            st.info(snapshot.get("error", "Live returns are unavailable."))
            return

        ref_date = snapshot["reference_date"]
        asof = snapshot.get("asof")
        asof_label = asof.strftime("%b %d, %I:%M %p") if asof is not None else "N/A"
        session_label = snapshot.get("market_session") or "Unknown session"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reference Value", f"${snapshot['reference_value']:,.0f}", ref_date.strftime("%Y-%m-%d"))
        c2.metric("Live Value", f"${snapshot['live_value']:,.0f}", f"{snapshot['live_return']*100:+.2f}%")
        c3.metric("Live Move", f"${snapshot['live_change']:,.0f}")
        c4.metric("Quote As Of", asof_label, session_label)

        rows = snapshot["rows"].copy()
        missing = rows[rows["Status"] != "OK"]["Ticker"].tolist()
        if missing:
            st.warning(
                "Live quotes unavailable for "
                + ", ".join(missing)
                + "; those sleeves are held flat in the estimate."
            )

        coverage = f"{snapshot['updated_count']}/{snapshot['position_count']}"
        st.caption(f"Reference close: {ref_date.strftime('%Y-%m-%d')} | Live coverage: {coverage}")

        display_cols = [
            "Ticker",
            "Live Ticker",
            "Weight",
            "Reference Price",
            "Live Price",
            "Underlying Return",
            "Effective Return",
            "Position Value",
            "Estimated Value",
            "Contribution",
            "Leverage",
            "Market Session",
            "Status",
        ]
        display_df = rows[display_cols].sort_values("Position Value", ascending=False)

        st.dataframe(
            display_df.style.format({
                "Weight": "{:.2%}",
                "Reference Price": "${:,.2f}",
                "Live Price": "${:,.2f}",
                "Underlying Return": "{:+.2%}",
                "Effective Return": "{:+.2%}",
                "Position Value": "${:,.0f}",
                "Estimated Value": "${:,.0f}",
                "Contribution": "${:+,.0f}",
                "Leverage": "{:.2f}x",
            }).map(color_return, subset=["Underlying Return", "Effective Return", "Contribution"]),
            use_container_width=True,
            hide_index=True,
        )

    view_key = unique_id or portfolio_name or "returns"
    returns_views = [
        "📋 Summary",
        "📅 Annual",
        "📆 Quarterly",
        "🗓️ Monthly",
        "🎚️ Year Drilldown",
        "📊 Daily",
        "⚡ Live",
        "📉 Drawdowns",
    ]
    selected_view = st.segmented_control(
        "Returns View",
        returns_views,
        default=returns_views[0],
        key=f"returns_view_{view_key}",
        label_visibility="collapsed",
    )

    if selected_view == returns_views[0]:
        st.subheader(f"{portfolio_name} Seasonal Summary")
        render_seasonal_summary(active_series)

        from app.ui.charts.rolling import render_rolling_metrics
        render_rolling_metrics(
            active_series,
            raw_response=raw_response,
            unique_id=unique_id,
        )

        from app.ui.charts.metrics import render_risk_return_metrics
        render_risk_return_metrics(
            active_series,
            stats=stats or {},
            raw_response=raw_response,
            unique_id=unique_id,
        )

    elif selected_view == returns_views[1]:
        annual_ret = _resample_returns(active_series, "YE")
        st.subheader(f"{portfolio_name} Annual Returns")

        has_fresh_annual = fresh_yearly and len(fresh_yearly) > 0 and not use_fresh

        if has_fresh_annual:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=annual_ret.index.year,
                y=annual_ret * 100,
                name="Continuous Sim",
                marker_color=["#00CC96" if x >= 0 else "#EF553B" for x in annual_ret],
                text=(annual_ret * 100).apply(lambda x: f"{x:+.1f}%"),
                textposition="auto",
            ))
            fresh_series_for_year = pd.Series(fresh_yearly)
            fresh_series_for_year = fresh_series_for_year.reindex(annual_ret.index.year).dropna()
            fig.add_trace(go.Bar(
                x=fresh_series_for_year.index,
                y=fresh_series_for_year * 100,
                name="Fresh Start",
                marker_color=["#42A5F5" if x >= 0 else "#FF7043" for x in fresh_series_for_year],
                text=(fresh_series_for_year * 100).apply(lambda x: f"{x:+.1f}%"),
                textposition="auto",
                opacity=0.7,
            ))
            fig.update_layout(barmode="group")
        else:
            colors = ["#00CC96" if x >= 0 else "#EF553B" for x in annual_ret]
            fig = go.Figure(go.Bar(
                x=annual_ret.index.year,
                y=annual_ret * 100,
                marker_color=colors,
                text=(annual_ret * 100).apply(lambda x: f"{x:+.1f}%"),
                textposition="auto",
            ))

        fig.update_layout(
            title="Annual Returns (%)",
            yaxis_title="Return (%)",
            xaxis_title="Year",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        annual_bal = port_series.resample("YE").last()
        ret_col = "Continuous" if has_fresh_annual else "Return"
        df_annual = pd.DataFrame({
            "Year": annual_ret.index.year,
            ret_col: annual_ret.values,
        }).sort_values("Year", ascending=False)
        if has_fresh_annual:
            df_annual["Fresh Start"] = df_annual["Year"].map(fresh_yearly)
        df_annual["Balance"] = df_annual["Year"].map(
            dict(zip(annual_bal.index.year, annual_bal.values))
        )

        fmt = {ret_col: "{:+.1%}", "Balance": "${:,.2f}"}
        style_cols = [ret_col]
        if has_fresh_annual:
            fmt["Fresh Start"] = "{:+.1%}"
            style_cols.append("Fresh Start")
        st.dataframe(
            df_annual.style.format(fmt).map(color_return, subset=style_cols),
            use_container_width=True,
            hide_index=True,
        )

        render_distribution(annual_ret, "Annual", "Years")

    elif selected_view == returns_views[2]:
        quarterly_views = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty:
            quarterly_views.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty:
            quarterly_views.append("Benchmark (Primary)")

        selected_quarterly_view = quarterly_views[0]
        if len(quarterly_views) > 1:
            selected_quarterly_view = st.segmented_control(
                "Quarterly View",
                quarterly_views,
                default=quarterly_views[0],
                key=f"quarterly_view_{view_key}",
                label_visibility="collapsed",
            )

        if selected_quarterly_view == portfolio_name:
            st.subheader(f"{portfolio_name} Quarterly Returns")
            quarterly_state = render_quarterly_returns_view(active_series)
            if quarterly_state is not None:
                _, quarterly_ret, _ = quarterly_state
                render_distribution(quarterly_ret, "Quarterly", "Quarters")
        elif selected_quarterly_view == "Benchmark (Comparison)":
            st.subheader("Standard Rebalance (Comparison) Quarterly Returns")
            render_quarterly_returns_view(comparison_series, suffix="_comp")
        else:
            st.subheader("Primary Benchmark Quarterly Returns")
            render_quarterly_returns_view(bench_series, suffix="_bench")

    elif selected_view == returns_views[3]:
        monthly_views = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty:
            monthly_views.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty:
            monthly_views.append("Benchmark (Primary)")

        selected_monthly_view = monthly_views[0]
        if len(monthly_views) > 1:
            selected_monthly_view = st.segmented_control(
                "Monthly View",
                monthly_views,
                default=monthly_views[0],
                key=f"monthly_view_{view_key}",
                label_visibility="collapsed",
            )

        if selected_monthly_view == portfolio_name:
            st.subheader(f"{portfolio_name} Monthly Returns")
            monthly_state = render_monthly_returns_view(active_series)

            if monthly_state is not None:
                visible_monthly_series, monthly_ret, _ = monthly_state

                st.subheader("Monthly Returns List")
                monthly_bal = visible_monthly_series.resample("ME").last()
                df_monthly_list = monthly_ret.to_frame(name="Return")
                df_monthly_list["Date"] = df_monthly_list.index.strftime("%Y-%m")
                df_monthly_list["Balance"] = monthly_bal.reindex(df_monthly_list.index).values
                df_monthly_list = df_monthly_list[["Date", "Return", "Balance"]].sort_index(ascending=False)
                st.dataframe(
                    df_monthly_list.style.format({"Return": "{:+.1%}", "Balance": "${:,.2f}"}).map(color_return, subset=["Return"]),
                    use_container_width=True,
                    hide_index=True,
                )

                render_distribution(monthly_ret, "Monthly", "Months")
        elif selected_monthly_view == "Benchmark (Comparison)":
            st.subheader("Standard Rebalance (Comparison) Monthly Returns")
            render_monthly_returns_view(comparison_series, suffix="_comp")
        else:
            st.subheader("Primary Benchmark Monthly Returns")
            render_monthly_returns_view(bench_series, suffix="_bench")

    elif selected_view == returns_views[4]:
        drilldown_views = [portfolio_name]
        if comparison_series is not None and not comparison_series.empty:
            drilldown_views.append("Benchmark (Comparison)")
        if bench_series is not None and not bench_series.empty:
            drilldown_views.append("Benchmark (Primary)")

        selected_drilldown_view = drilldown_views[0]
        if len(drilldown_views) > 1:
            selected_drilldown_view = st.segmented_control(
                "Year Drilldown View",
                drilldown_views,
                default=drilldown_views[0],
                key=f"year_drilldown_view_{view_key}",
                label_visibility="collapsed",
            )

        if selected_drilldown_view == portfolio_name:
            st.subheader(f"{portfolio_name} Year Drilldown")
            render_year_drilldown_view(active_series)
        elif selected_drilldown_view == "Benchmark (Comparison)":
            st.subheader("Standard Rebalance (Comparison) Year Drilldown")
            render_year_drilldown_view(comparison_series, suffix="_comp")
        else:
            st.subheader("Primary Benchmark Year Drilldown")
            render_year_drilldown_view(bench_series, suffix="_bench")

    elif selected_view == returns_views[5]:
        daily_ret = active_series.pct_change().dropna()
        st.subheader(f"{portfolio_name} Daily Returns")

        c1, c2, c3 = st.columns(3)
        c1.metric("Best Day", f"{daily_ret.max()*100:+.2f}%")
        c2.metric("Worst Day", f"{daily_ret.min()*100:+.2f}%")
        c3.metric("Positive Days", f"{(daily_ret > 0).mean()*100:.1f}%")

        st.subheader("Daily Returns List")
        df_daily_list = daily_ret.to_frame(name="Return")
        df_daily_list["Date"] = df_daily_list.index.date
        df_daily_list["Balance"] = active_series.reindex(df_daily_list.index).values
        df_daily_list = df_daily_list[["Date", "Return", "Balance"]].sort_index(ascending=False)

        st.dataframe(
            df_daily_list.style.format({"Return": "{:+.2%}", "Balance": "${:,.2f}"}).map(color_return, subset=["Return"]),
            use_container_width=True,
            hide_index=True,
        )

        render_distribution(daily_ret, "Daily", "Days")

    elif selected_view == returns_views[6]:
        render_live_returns_view()

    else:
        st.subheader(f"{portfolio_name} Corrections >5%")

        from app.services.data_service import fetch_component_data
        from app.core.calculations.stats import build_drawdown_table

        start_date = active_series.index[0].strftime("%Y-%m-%d")
        end_date = active_series.index[-1].strftime("%Y-%m-%d")

        try:
            spy_prices = fetch_component_data(["SPYSIM"], start_date, end_date)
            spy_col = spy_prices.columns[0]
            spy_raw = spy_prices[spy_col].reindex(active_series.index).ffill().bfill()
            spy_norm = spy_raw / spy_raw.iloc[0] * active_series.iloc[0]
        except Exception:
            spy_norm = active_series.copy()
            st.warning("Could not load SPY benchmark data. SPY columns may be inaccurate.")

        df = build_drawdown_table(active_series, spy_norm)

        if df.empty:
            st.info("No corrections >5% found in this period.")
        else:
            from app.core.calculations.stats import fmt_duration

            n_total = len(df)
            median_decline = df["% Decline"].median()
            n_severe = (df["_severity"] == "Severe").sum()
            n_moderate = (df["_severity"] == "Moderate").sum()
            n_ongoing = df["_ongoing"].sum()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Corrections", n_total)
            c2.metric("Median Decline", f"{median_decline:.1f}%")
            c3.metric("Severe (>25%)", n_severe)
            c4.metric("Moderate (15-25%)", n_moderate)
            c5.metric("Ongoing", n_ongoing)

            n_mild = (df["_severity"] == "Mild").sum()
            n_minor = (df["_severity"] == "Minor").sum()
            filter_options = [
                f"All ({n_total})",
                f"Severe >25% ({n_severe})",
                f"Moderate 15-25% ({n_moderate})",
                f"Mild 10-15% ({n_mild})",
                f"Minor 5-10% ({n_minor})",
            ]
            selected = st.radio(
                "Filter by severity",
                filter_options,
                horizontal=True,
                label_visibility="collapsed",
                key=f"dd_filter_{unique_id}",
            )

            filtered = df
            if "Severe" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Severe"]
            elif "Moderate" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Moderate"]
            elif "Mild" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Mild"]
            elif "Minor" in selected and "All" not in selected:
                filtered = df[df["_severity"] == "Minor"]

            display_df = filtered[[c for c in filtered.columns if not c.startswith("_")]].copy()

            def fmt_duration_col(val):
                """Format day counts: negative = ongoing, None = N/A."""
                if val is None or pd.isna(val):
                    return "N/A"
                val = int(val)
                if val < 0:
                    return f"ongoing ({fmt_duration(-val)})"
                return fmt_duration(val)

            def fmt_pct(val):
                if pd.isna(val):
                    return ""
                return f"{val:.1f}%"

            def fmt_ratio(val):
                if pd.isna(val):
                    return ""
                return f"{val:.1f}x"

            def style_drawdowns(styler):
                def color_decline(val):
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        return ""
                    if abs(v) >= 25:
                        return "color: #ef4444; font-weight: bold"
                    if abs(v) >= 15:
                        return "color: #f97316; font-weight: bold"
                    if abs(v) >= 10:
                        return "color: #eab308"
                    return "color: #94a3b8"

                def color_ratio(val):
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        return ""
                    if v < 1.5:
                        return "color: #34d399"
                    if v > 3.0:
                        return "color: #f97316"
                    return ""

                def color_duration(val):
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        return ""
                    if v < 0:
                        return "color: #ef4444; font-weight: bold"
                    return ""

                def color_spy_duration(val):
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return ""
                    try:
                        v = float(val)
                    except (ValueError, TypeError):
                        return ""
                    if v < 0:
                        return "color: #ef4444; font-weight: bold"
                    return "color: #94a3b8"

                def color_spy_pct(val):
                    return "color: #94a3b8"

                styler.map(color_decline, subset=["% Decline"])
                styler.map(color_ratio, subset=["Ratio"])
                styler.map(color_duration, subset=["Recovery from Bottom", "Decline + Recovery Time"])
                styler.map(color_spy_pct, subset=["SPY DD"])
                styler.map(color_spy_duration, subset=["SPY Recovery from Bottom", "SPY Decline + Recovery Time"])
                styler.format(fmt_pct, subset=["% Decline", "SPY DD"])
                styler.format(fmt_ratio, subset=["Ratio"])
                styler.format(fmt_duration_col, subset=["Recovery from Bottom", "Decline + Recovery Time", "SPY Recovery from Bottom", "SPY Decline + Recovery Time"])
                return styler

            st.dataframe(
                display_df.style.pipe(style_drawdowns),
                use_container_width=True,
                hide_index=True,
                height=min(800, 35 * (len(display_df) + 1) + 38),
            )
