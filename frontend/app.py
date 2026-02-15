"""
Streamlit dashboard for the Adaptive Portfolio Engine - REDESIGNED VERSION
Professional robo-advisor UI with clear portfolio summary, risk status, and allocation breakdown.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Adaptive Portfolio Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .big-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .positive {
        color: #2ca02c;
    }
    .negative {
        color: #d62728;
    }
    .neutral {
        color: #1f77b4;
    }
    .regime-bull {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        display: inline-block;
    }
    .regime-volatile {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        display: inline-block;
    }
    .regime-crash {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        display: inline-block;
    }
    .event-timeline {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .event-date {
        font-weight: 600;
        color: #1f77b4;
    }
    .divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============= Helper Functions =============

def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def run_backtest_api(start_year, end_year, with_risk_engine):
    """Call backend API to run backtest."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/run_backtest",
            json={
                "start_year": start_year,
                "end_year": end_year,
                "with_risk_engine": with_risk_engine
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling backend: {str(e)}")
        return None

def run_stress_test_api(stress_type, with_risk_engine, start_year, end_year):
    """Call backend API to run stress test."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/run_stress_test",
            json={
                "stress_type": stress_type,
                "with_risk_engine": with_risk_engine,
                "start_year": start_year,
                "end_year": end_year
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling backend: {str(e)}")
        return None

def compare_risk_engine_api(start_year, end_year):
    """Call backend API to compare scenarios."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/compare_risk_engine",
            params={"start_year": start_year, "end_year": end_year},
            timeout=600
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling backend: {str(e)}")
        return None

def parse_timeseries(data):
    """Parse timeseries data from API response."""
    dates = [item['date'] for item in data]
    values = [float(item['value']) for item in data]
    return pd.Series(values, index=pd.to_datetime(dates))

def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage."""
    return f"{value*100:.2f}%"

def get_regime_class(regime):
    """Get CSS class for regime."""
    regime = regime.upper()
    if 'BULL' in regime:
        return 'regime-bull'
    elif 'CRASH' in regime:
        return 'regime-crash'
    else:
        return 'regime-volatile'

def humanize_event(event):
    """Convert technical event log to human-readable format."""
    event_type = event.get('event_type', '')
    details = event.get('details', '')
    date = event.get('date', '')
    
    # Format date
    try:
        dt = pd.to_datetime(date)
        date_str = dt.strftime('%b %Y')
    except:
        date_str = date
    
    # Choose emoji and rewrite details
    if 'REGIME_CHANGE' in event_type:
        emoji = '🔄'
        from_regime = event.get('from_regime', '')
        to_regime = event.get('to_regime', '')
        message = f"Market regime changed from {from_regime} to {to_regime}"
    elif 'DRAWDOWN_20' in event_type:
        emoji = '🚨'
        message = "Emergency: Drawdown exceeded 20%. Portfolio moved to 100% cash for protection."
    elif 'DRAWDOWN_10' in event_type:
        emoji = '📉'
        message = "Drawdown exceeded 10%. Portfolio exposure reduced by 50%."
    elif 'VOL_BREACH' in event_type:
        emoji = '📊'
        message = "Portfolio volatility exceeded target. Exposure scaled down."
    elif 'STOP_LOSS' in event_type:
        emoji = '🛑'
        tickers = event.get('stopped_tickers', [])
        if tickers:
            message = f"Stop-loss triggered for {', '.join(tickers)}. Positions closed."
        else:
            message = "Stop-loss triggered. Position(s) closed."
    elif 'POSITION_CAP' in event_type:
        emoji = '⚖️'
        message = "Position weights exceeded limit. Capped at 10% per stock."
    else:
        emoji = 'ℹ️'
        message = details
    
    return emoji, date_str, message

# ============= Chart Functions =============

def plot_equity_curve_with_regime(equity_data, regime_data):
    """Plot equity curve with regime shading."""
    equity_series = parse_timeseries(equity_data)
    regime_series = pd.Series(
        [item['value'] for item in regime_data],
        index=pd.to_datetime([item['date'] for item in regime_data])
    )
    
    fig = go.Figure()
    
    # Regime colors
    regime_colors = {
        'BULL': 'rgba(76, 175, 80, 0.15)',
        'VOLATILE': 'rgba(255, 193, 7, 0.15)',
        'CRASH': 'rgba(244, 67, 54, 0.15)'
    }
    
    # Add regime backgrounds
    current_regime = None
    start_idx = None
    
    for i, (date, regime) in enumerate(regime_series.items()):
        if regime != current_regime:
            if current_regime is not None:
                fig.add_vrect(
                    x0=regime_series.index[start_idx],
                    x1=date,
                    fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.1)'),
                    layer="below",
                    line_width=0,
                )
            current_regime = regime
            start_idx = i
    
    # Add last regime
    if current_regime is not None:
        fig.add_vrect(
            x0=regime_series.index[start_idx],
            x1=regime_series.index[-1],
            fillcolor=regime_colors.get(current_regime, 'rgba(128, 128, 128, 0.1)'),
            layer="below",
            line_width=0,
        )
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=equity_series.index,
        y=equity_series.values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        showlegend=False
    )
    
    return fig

def plot_drawdown(drawdown_data):
    """Plot drawdown curve."""
    drawdown_series = parse_timeseries(drawdown_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values * 100,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#d62728', width=2),
        fillcolor='rgba(214, 39, 40, 0.3)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_exposure(exposure_data):
    """Plot exposure timeline."""
    exposure_series = parse_timeseries(exposure_data)
    
    fig = go.Figure()
    
    # Stock exposure
    fig.add_trace(go.Scatter(
        x=exposure_series.index,
        y=exposure_series.values * 100,
        mode='lines',
        name='Stock Exposure',
        fill='tozeroy',
        line=dict(color='#2ca02c', width=2),
        fillcolor='rgba(44, 160, 44, 0.3)',
        stackgroup='one'
    ))
    
    # Cash
    fig.add_trace(go.Scatter(
        x=exposure_series.index,
        y=(1 - exposure_series.values) * 100,
        mode='lines',
        name='Cash',
        fill='tonexty',
        line=dict(color='#ff7f0e', width=2),
        fillcolor='rgba(255, 127, 14, 0.3)',
        stackgroup='one'
    ))
    
    fig.update_layout(
        title="Portfolio Exposure Over Time",
        xaxis_title="Date",
        yaxis_title="Allocation (%)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def plot_regime_timeline(regime_data):
    """Plot regime timeline as color-coded bars."""
    regime_series = pd.Series(
        [item['value'] for item in regime_data],
        index=pd.to_datetime([item['date'] for item in regime_data])
    )
    
    # Convert to numeric for plotting
    regime_map = {'BULL': 3, 'VOLATILE': 2, 'CRASH': 1}
    regime_numeric = regime_series.map(regime_map)
    
    colors = {'BULL': '#4CAF50', 'VOLATILE': '#FFC107', 'CRASH': '#F44336'}
    color_list = [colors[r] for r in regime_series]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=regime_series.index,
        y=regime_numeric,
        marker_color=color_list,
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Regime: %{text}<extra></extra>',
        text=regime_series.values
    ))
    
    fig.update_layout(
        title="Market Regime Timeline",
        xaxis_title="Date",
        yaxis_title="Regime",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['CRASH', 'VOLATILE', 'BULL']
        ),
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_allocation_pie(weights_dict):
    """Plot current allocation as pie chart."""
    if not weights_dict:
        return None
    
    df = pd.DataFrame(list(weights_dict.items()), columns=['Asset', 'Weight'])
    df = df[df['Weight'] > 0.001]  # Filter out tiny weights
    
    fig = px.pie(
        df,
        values='Weight',
        names='Asset',
        title='Current Portfolio Allocation',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

# ============= Main App =============

def main():
    # Header
    st.markdown('<div class="main-header">📈 Adaptive Portfolio Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Risk-Managed Portfolio Allocation System</div>', unsafe_allow_html=True)
    
    # Check backend health
    if not check_backend_health():
        st.error(f"⚠️ Backend server is not running! Please start it at {BACKEND_URL}")
        st.info("Run: `start_backend.bat` or `python backend/main.py`")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=2010, max_value=2023, value=2015, step=1)
    with col2:
        end_year = st.number_input("End Year", min_value=2011, max_value=2024, value=2024, step=1)
    
    if end_year <= start_year:
        st.sidebar.error("End year must be after start year!")
        return
    
    # Risk engine toggle
    with_risk_engine = st.sidebar.checkbox("🛡️ Enable Risk Engine", value=True, help="Enable multi-layer risk management")
    
    st.sidebar.markdown("---")
    
    # Actions
    st.sidebar.subheader("🚀 Actions")
    
    run_backtest_btn = st.sidebar.button("▶️ Run Backtest", type="primary", use_container_width=True)
    
    st.sidebar.markdown("### ⚡ Stress Testing")
    stress_type = st.sidebar.selectbox(
        "Scenario",
        ["market_shock", "volatility_spike", "correlation_spike"],
        format_func=lambda x: {
            "market_shock": "📉 Market Shock (-5%)",
            "volatility_spike": "📊 Volatility Spike (2x)",
            "correlation_spike": "🔗 Correlation Spike"
        }[x]
    )
    
    run_stress_btn = st.sidebar.button("⚡ Run Stress Test", use_container_width=True)
    
    st.sidebar.markdown("---")
    compare_btn = st.sidebar.button("📊 Compare All Scenarios", use_container_width=True)
    
    # Main content
    if run_backtest_btn:
        with st.spinner("🔄 Running backtest... This may take a few minutes."):
            results = run_backtest_api(start_year, end_year, with_risk_engine)
        
        if results:
            st.session_state['results'] = results
            st.session_state['is_stress'] = False
            st.success("✅ Backtest completed successfully!")
    
    if run_stress_btn:
        with st.spinner(f"⚡ Running stress test ({stress_type})..."):
            results = run_stress_test_api(stress_type, with_risk_engine, start_year, end_year)
        
        if results:
            st.session_state['results'] = results
            st.session_state['is_stress'] = True
            st.session_state['stress_type'] = stress_type
            st.success(f"✅ Stress test completed: {results.get('scenario_description', '')}")
    
    if compare_btn:
        with st.spinner("📊 Running all scenarios... This will take several minutes."):
            comparison = compare_risk_engine_api(start_year, end_year)
        
        if comparison:
            st.session_state['comparison'] = comparison
            st.success("✅ Comparison completed!")
    
    # Display results
    if 'results' in st.session_state:
        display_results(st.session_state['results'], st.session_state.get('is_stress', False))
    
    # Display comparison
    if 'comparison' in st.session_state:
        display_comparison(st.session_state['comparison'])

def display_results(results, is_stress=False):
    """Display backtest results with professional layout."""
    
    metrics = results['metrics']
    equity_data = results['equity_curve']
    
    # Parse data
    equity_series = parse_timeseries(equity_data)
    initial_capital = 100000  # As per backend
    current_value = equity_series.iloc[-1]
    peak_value = equity_series.max()
    total_profit = current_value - initial_capital
    total_return = total_profit / initial_capital
    current_dd = metrics['Max Drawdown']
    
    # Get current status from last data points
    exposure_series = parse_timeseries(results['exposure_timeline'])
    regime_series = pd.Series(
        [item['value'] for item in results['regime_timeline']],
        index=pd.to_datetime([item['date'] for item in results['regime_timeline']])
    )
    
    current_exposure = exposure_series.iloc[-1]
    current_regime = regime_series.iloc[-1]
    current_cash = 1 - current_exposure
    
    # ========== SECTION 1: Portfolio Summary ==========
    st.markdown("## 💼 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Initial Capital</div>
            <div class="metric-value neutral">{format_currency(initial_capital)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Value</div>
            <div class="metric-value" style="font-size: 2.5rem; color: #1f77b4;">{format_currency(current_value)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        profit_class = "positive" if total_profit >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Profit</div>
            <div class="metric-value {profit_class}">{format_currency(total_profit)}</div>
            <div class="{profit_class}">{format_percentage(total_return)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dd_class = "negative" if current_dd < -0.05 else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Peak Value</div>
            <div class="metric-value neutral">{format_currency(peak_value)}</div>
            <div class="{dd_class}">Drawdown: {format_percentage(current_dd)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== SECTION 2: Current Risk Status ==========
    st.markdown("## 🎯 Current Risk Status")
    
    regime_class = get_regime_class(current_regime)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <h4>Market Regime</h4>
            <div class="{regime_class}" style="margin-top: 1rem; text-align: center;">{current_regime}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="status-card">
            <h4>Portfolio Allocation</h4>
            <div style="margin-top: 1rem;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #2ca02c;">Stocks: {format_percentage(current_exposure)}</div>
                <div style="font-size: 1.2rem; color: #ff7f0e;">Cash: {format_percentage(current_cash)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        vol_class = "negative" if metrics['Volatility'] > 0.25 else "positive"
        risk_status = "✅ ENABLED" if st.session_state.get('results', {}) else "❌ DISABLED"
        
        st.markdown(f"""
        <div class="status-card">
            <h4>Risk Controls</h4>
            <div style="margin-top: 1rem;">
                <div style="font-size: 1.2rem; font-weight: 600;">Status: {risk_status}</div>
                <div class="{vol_class}">Volatility: {format_percentage(metrics['Volatility'])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== SECTION 3: Performance Metrics Grid ==========
    st.markdown("## 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CAGR", format_percentage(metrics['CAGR']))
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}")
    with col3:
        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.3f}")
    with col4:
        st.metric("Volatility", format_percentage(metrics['Volatility']))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Drawdown", format_percentage(metrics['Max Drawdown']), delta=None, delta_color="inverse")
    with col2:
        st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.3f}")
    with col3:
        st.metric("Time in Cash", f"{metrics['Time in Cash (%)']:.1f}%")
    with col4:
        st.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f}%")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== SECTION 4: Allocation Breakdown ==========
    st.markdown("## 📑 Current Allocation Breakdown")
    
    # For now, show exposure breakdown (stocks vs cash)
    # In a real scenario, you'd get individual stock weights from the last rebalance
    allocation_data = {
        'Stocks': current_exposure,
        'Cash': current_cash
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if allocation_data:
            fig_pie = plot_allocation_pie(allocation_data)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        allocation_df = pd.DataFrame([
            {'Asset': 'Stock Portfolio', 'Weight (%)': f"{current_exposure*100:.2f}%", 'Dollar Value': format_currency(current_value * current_exposure)},
            {'Asset': 'Cash', 'Weight (%)': f"{current_cash*100:.2f}%", 'Dollar Value': format_currency(current_value * current_cash)}
        ])
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== SECTION 5: Charts in Tabs ==========
    st.markdown("## 📈 Performance Charts")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💹 Equity Curve", "📉 Drawdown", "⚖️ Exposure", "🎯 Regime Timeline"])
    
    with tab1:
        fig_equity = plot_equity_curve_with_regime(results['equity_curve'], results['regime_timeline'])
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with tab2:
        fig_dd = plot_drawdown(results['drawdown_curve'])
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with tab3:
        fig_exp = plot_exposure(results['exposure_timeline'])
        st.plotly_chart(fig_exp, use_container_width=True)
    
    with tab4:
        fig_regime = plot_regime_timeline(results['regime_timeline'])
        st.plotly_chart(fig_regime, use_container_width=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ========== SECTION 6: Risk Event Timeline ==========
    st.markdown("## 📋 Risk Event Timeline")
    
    if results['risk_logs']:
        st.info(f"📌 Total Events: {len(results['risk_logs'])}")
        
        # Humanize and display
        for event in reversed(results['risk_logs'][-20:]):  # Show last 20 events
            emoji, date_str, message = humanize_event(event)
            
            st.markdown(f"""
            <div class="event-timeline">
                <div class="event-date">{emoji} {date_str}</div>
                <div style="margin-top: 0.5rem; color: #333;">{message}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ No risk events triggered during this period. Portfolio operated within normal parameters.")

def display_comparison(comparison):
    """Display scenario comparison."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("## 🆚 Scenario Comparison")
    
    st.info(f"💡 {comparison['summary']['conclusion']}")
    
    # Create comparison DataFrame
    comp_df = pd.DataFrame(comparison['comparison']).T
    
    # Display metrics side by side
    scenarios = comp_df.index.tolist()
    
    tabs = st.tabs(scenarios)
    
    for i, scenario in enumerate(scenarios):
        with tabs[i]:
            cols = st.columns(4)
            
            row_data = comp_df.loc[scenario]
            
            with cols[0]:
                st.metric("CAGR", format_percentage(row_data['CAGR']))
            with cols[1]:
                st.metric("Sharpe Ratio", f"{row_data['Sharpe Ratio']:.3f}")
            with cols[2]:
                st.metric("Max Drawdown", format_percentage(row_data['Max Drawdown']))
            with cols[3]:
                st.metric("Calmar Ratio", f"{row_data['Calmar Ratio']:.3f}")
    
    # Comparison table
    st.markdown("### 📋 Full Comparison Table")
    st.dataframe(comp_df.style.format({
        'CAGR': '{:.2%}',
        'Total Return': '{:.2%}',
        'Volatility': '{:.2%}',
        'Sharpe Ratio': '{:.3f}',
        'Sortino Ratio': '{:.3f}',
        'Max Drawdown': '{:.2%}',
        'Calmar Ratio': '{:.3f}',
        'Time in Cash (%)': '{:.2f}%',
        'Win Rate (%)': '{:.2f}%'
    }), use_container_width=True)

if __name__ == "__main__":
    main()
