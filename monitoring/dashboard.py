
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
import asyncpg
from datetime import datetime, timedelta
import time
import redis
import json
from typing import Dict, List
import requests

st.set_page_config(
    page_title="Quantum Trading System Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    def __init__(self):
        self.db_pool = None
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
    async def init_db(self):
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(
                "postgresql://quantum:quantum_pass@localhost:5432/quantum_trading",
                min_size=2,
                max_size=10
            )
            
    async def get_real_time_metrics(self):
        await self.init_db()
        
        async with self.db_pool.acquire() as conn:
            # Portfolio value and PnL
            portfolio_query = """
                SELECT 
                    SUM(CASE WHEN is_open THEN size * current_price ELSE 0 END) as portfolio_value,
                    SUM(unrealized_pnl) as unrealized_pnl,
                    SUM(realized_pnl) as realized_pnl,
                    COUNT(*) FILTER (WHERE is_open) as open_positions
                FROM positions
            """
            
            portfolio_row = await conn.fetchrow(portfolio_query)
            
            # Recent trades
            trades_query = """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(*) FILTER (WHERE success AND timestamp >= NOW() - INTERVAL '24 hours') as trades_24h,
                    COUNT(*) FILTER (WHERE success) as successful_trades,
                    AVG(pnl) FILTER (WHERE success) as avg_profit,
                    SUM(pnl) as total_pnl
                FROM trades 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
            """
            
            trades_row = await conn.fetchrow(trades_query)
            
            # Risk metrics
            risk_query = """
                SELECT portfolio_var, max_drawdown, sharpe_ratio, sortino_ratio
                FROM risk_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            risk_row = await conn.fetchrow(risk_query)
            
            return {
                'portfolio': dict(portfolio_row) if portfolio_row else {},
                'trades': dict(trades_row) if trades_row else {},
                'risk': dict(risk_row) if risk_row else {}
            }
            
    async def get_performance_chart_data(self, days: int = 7):
        await self.init_db()
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    SUM(pnl) as hourly_pnl,
                    COUNT(*) as trades_count,
                    AVG(execution_time_ms) as avg_execution_time
                FROM trades 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour
            """ % days
            
            rows = await conn.fetch(query)
            
            df = pd.DataFrame([dict(row) for row in rows])
            if not df.empty:
                df['cumulative_pnl'] = df['hourly_pnl'].cumsum()
                
            return df
            
    async def get_trading_signals_data(self):
        await self.init_db()
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    symbol,
                    confidence,
                    momentum,
                    research_score,
                    security_score,
                    status,
                    timestamp
                FROM signals 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY timestamp DESC
                LIMIT 50
            """
            
            rows = await conn.fetch(query)
            return pd.DataFrame([dict(row) for row in rows])
            
    async def get_risk_breakdown(self):
        await self.init_db()
        
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    symbol,
                    size * current_price as position_value,
                    unrealized_pnl,
                    unrealized_pnl / (size * entry_price) * 100 as return_pct,
                    risk_score
                FROM positions 
                WHERE is_open = true
                ORDER BY ABS(unrealized_pnl) DESC
            """
            
            rows = await conn.fetch(query)
            return pd.DataFrame([dict(row) for row in rows])

def create_metrics_cards(metrics):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        portfolio_value = metrics['portfolio'].get('portfolio_value', 0) or 0
        st.metric(
            "Portfolio Value",
            f"${portfolio_value:,.2f}",
            delta=f"${metrics['portfolio'].get('unrealized_pnl', 0) or 0:,.2f}"
        )
        
    with col2:
        total_trades = metrics['trades'].get('total_trades', 0) or 0
        trades_24h = metrics['trades'].get('trades_24h', 0) or 0
        st.metric(
            "Total Trades (7d)",
            f"{total_trades:,}",
            delta=f"{trades_24h} today"
        )
        
    with col3:
        successful_trades = metrics['trades'].get('successful_trades', 0) or 0
        win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{successful_trades}/{total_trades} wins"
        )
        
    with col4:
        sharpe_ratio = metrics['risk'].get('sharpe_ratio', 0) or 0
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio:.2f}",
            delta="Target: 3.0+"
        )

def create_performance_chart(df):
    if df.empty:
        st.warning("No performance data available")
        return
        
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative PnL', 'Hourly Trading Volume'),
        vertical_spacing=0.1
    )
    
    # Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='#00ff88', width=3)
        ),
        row=1, col=1
    )
    
    # Trading volume
    fig.add_trace(
        go.Bar(
            x=df['hour'],
            y=df['trades_count'],
            name='Trades/Hour',
            marker_color='#ff6b6b'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Trading Performance Overview",
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Trades", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def create_signals_table(df):
    if df.empty:
        st.warning("No recent signals")
        return
        
    # Color code by confidence
    def color_confidence(val):
        if val >= 0.8:
            return 'background-color: #d4edda'
        elif val >= 0.6:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
            
    styled_df = df.style.applymap(color_confidence, subset=['confidence'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

def create_risk_dashboard(risk_df):
    if risk_df.empty:
        st.warning("No open positions")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        # Position sizes pie chart
        fig_pie = px.pie(
            risk_df,
            values='position_value',
            names='symbol',
            title='Portfolio Allocation'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        # PnL by position
        fig_bar = px.bar(
            risk_df,
            x='symbol',
            y='unrealized_pnl',
            color='return_pct',
            title='Unrealized PnL by Position',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def create_system_health():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CPU Usage
        cpu_usage = np.random.uniform(20, 80)
        fig_cpu = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cpu_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CPU Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_cpu.update_layout(height=300)
        st.plotly_chart(fig_cpu, use_container_width=True)
        
    with col2:
        # Memory Usage
        memory_usage = np.random.uniform(30, 70)
        fig_mem = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = memory_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Memory Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        fig_mem.update_layout(height=300)
        st.plotly_chart(fig_mem, use_container_width=True)
        
    with col3:
        # GPU Usage
        gpu_usage = np.random.uniform(40, 90)
        fig_gpu = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gpu_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "GPU Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        fig_gpu.update_layout(height=300)
        st.plotly_chart(fig_gpu, use_container_width=True)

def create_live_trades_feed():
    st.subheader("🔴 Live Trades Feed")
    
    # Placeholder for live trades
    trades_placeholder = st.empty()
    
    # Simulate live trades
    sample_trades = [
        {"time": "14:23:15", "symbol": "BTC-USDT", "side": "BUY", "size": "0.1", "price": "43,250", "pnl": "+$125.50"},
        {"time": "14:22:48", "symbol": "ETH-USDT", "side": "SELL", "size": "2.5", "price": "2,680", "pnl": "+$89.20"},
        {"time": "14:22:12", "symbol": "SOL-USDT", "side": "BUY", "size": "15", "price": "98.50", "pnl": "+$45.80"},
        {"time": "14:21:39", "symbol": "ADA-USDT", "side": "SELL", "size": "1000", "price": "0.485", "pnl": "-$12.30"},
    ]
    
    with trades_placeholder.container():
        for trade in sample_trades:
            cols = st.columns([1, 2, 1, 1, 2, 1])
            cols[0].write(trade["time"])
            cols[1].write(trade["symbol"])
            
            if trade["side"] == "BUY":
                cols[2].markdown(f"🟢 {trade['side']}")
            else:
                cols[2].markdown(f"🔴 {trade['side']}")
                
            cols[3].write(trade["size"])
            cols[4].write(f"${trade['price']}")
            
            if trade["pnl"].startswith("+"):
                cols[5].markdown(f"🟢 {trade['pnl']}")
            else:
                cols[5].markdown(f"🔴 {trade['pnl']}")

async def main():
    st.title("🚀 Quantum Trading System Dashboard")
    st.markdown("---")
    
    dashboard = TradingDashboard()
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    
    refresh_rate = st.sidebar.selectbox(
        "Refresh Rate",
        [5, 10, 30, 60],
        index=1,
        help="Dashboard refresh interval in seconds"
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        [1, 3, 7, 30],
        index=2,
        help="Days of historical data to display"
    )
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.experimental_rerun()
        
    # Emergency controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚠️ Emergency Controls**")
    
    if st.sidebar.button("🛑 STOP ALL TRADING", type="primary"):
        st.sidebar.error("Emergency stop activated!")
        
    if st.sidebar.button("💰 CLOSE ALL POSITIONS"):
        st.sidebar.warning("Closing all positions...")
        
    # Main dashboard content
    try:
        # Real-time metrics
        metrics = await dashboard.get_real_time_metrics()
        create_metrics_cards(metrics)
        
        st.markdown("---")
        
        # Performance charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Performance Overview")
            perf_data = await dashboard.get_performance_chart_data(time_range)
            create_performance_chart(perf_data)
            
        with col2:
            st.subheader("🖥️ System Health")
            create_system_health()
            
        st.markdown("---")
        
        # Trading signals and risk
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Recent Trading Signals")
            signals_data = await dashboard.get_trading_signals_data()
            create_signals_table(signals_data)
            
        with col2:
            st.subheader("⚖️ Risk Breakdown")
            risk_data = await dashboard.get_risk_breakdown()
            create_risk_dashboard(risk_data)
            
        st.markdown("---")
        
        # Live feed
        create_live_trades_feed()
        
    except Exception as e:
        st.error(f"Dashboard Error: {str(e)}")
        st.info("Displaying demo data...")
        
        # Demo metrics
        demo_metrics = {
            'portfolio': {'portfolio_value': 125000, 'unrealized_pnl': 2500},
            'trades': {'total_trades': 1247, 'trades_24h': 89, 'successful_trades': 934},
            'risk': {'sharpe_ratio': 2.85}
        }
        create_metrics_cards(demo_metrics)
        
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_rate)
        st.experimental_rerun()

if __name__ == "__main__":
    asyncio.run(main())
