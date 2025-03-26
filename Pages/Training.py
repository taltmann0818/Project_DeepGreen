import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

# Custom libraries
from Components.ModelDataset import DataModule
from Components.TrainModel import LSTMClassifierModel
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

st.set_page_config(
    page_title="Model Training",
    layout="wide",
)


# ------------------------------
# Sample Data Creation Functions
# ------------------------------
def create_sample_ohlc_data():
    """Creates a sample OHLC dataset."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    open_prices = [100 + i for i in range(30)]
    high_prices = [o + 5 for o in open_prices]
    low_prices = [o - 5 for o in open_prices]
    close_prices = [o + 2 for o in open_prices]
    df = pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": [1000 + i*10 for i in range(30)]
    })
    return df

def create_sample_portfolio_data():
    """Creates sample portfolio data (portfolio value and some benchmark)."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    portfolio_value = [10000 + (i * 150) for i in range(30)]
    benchmark_value = [10000 + (i * 100) for i in range(30)]
    df = pd.DataFrame({
        "Date": dates,
        "Portfolio": portfolio_value,
        "Benchmark": benchmark_value
    })
    return df

def create_sample_stats():
    """Creates sample stats to be displayed in a table."""
    data = {
        "Metric": [
            "Total Trades", 
            "Win Rate", 
            "Average Win", 
            "Average Loss", 
            "Net Profit", 
            "Max Drawdown"
        ],
        "Value": [
            25, 
            "60%", 
            "2.5%", 
            "-1.2%", 
            "15%", 
            "-5%"
        ]
    }
    df = pd.DataFrame(data)
    return df

def create_sample_metric_stats():
    """Creates sample data for a radar/spider chart or other metric visualization."""
    metrics = ["Profit Factor", "Sharpe Ratio", "Sortino Ratio", "Volatility", "Exposure"]
    values = [1.5, 1.2, 1.4, 0.8, 0.9]
    return metrics, values

# ---------------
# Streamlit Layout
# ---------------

# Sidebar: Settings
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Symbol", ["AAPL", "TSLA", "GOOG", "BTC-USD"], index=0)
date_range = st.sidebar.date_input("Date Range", [date(2023,1,1), date(2023,1,30)])
moving_averages = st.sidebar.multiselect("Moving Averages", ["MA20", "MA50", "MA200"], ["MA20", "MA50"])
show_signals = st.sidebar.checkbox("Show Buy/Sell Signals", value=True)
show_volume = st.sidebar.checkbox("Show Volume", value=True)
st.sidebar.markdown("---")
trade_mode = st.sidebar.radio("Trade Mode", ["Paper Trading", "Live Trading"])
st.sidebar.button("Refresh")

# 1) OHLC and Signals
st.subheader("OHLC and Signals")
ohlc_data = create_sample_ohlc_data()

fig_ohlc = go.Figure(data=[go.Candlestick(
    x=ohlc_data["Date"],
    open=ohlc_data["Open"],
    high=ohlc_data["High"],
    low=ohlc_data["Low"],
    close=ohlc_data["Close"],
    name="OHLC"
)])
fig_ohlc.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))

# Optionally add volume bars
if show_volume:
    fig_ohlc.add_trace(
        go.Bar(
            x=ohlc_data["Date"],
            y=ohlc_data["Volume"],
            marker_color="gray",
            name="Volume",
            yaxis="y2"
        )
    )
    fig_ohlc.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            position=1.0,
            range=[0, max(ohlc_data["Volume"])*1.2]
        )
    )

# Add moving averages if selected
for ma in moving_averages:
    window = int(ma.replace("MA", ""))
    ohlc_data[ma] = ohlc_data["Close"].rolling(window=window).mean()
    fig_ohlc.add_trace(
        go.Scatter(
            x=ohlc_data["Date"],
            y=ohlc_data[ma],
            mode="lines",
            name=ma
        )
    )

# Add buy/sell signals if toggled
if show_signals:
    # Dummy signals at random points
    buy_signals_x = [ohlc_data["Date"].iloc[5], ohlc_data["Date"].iloc[15]]
    buy_signals_y = [ohlc_data["Close"].iloc[5], ohlc_data["Close"].iloc[15]]
    sell_signals_x = [ohlc_data["Date"].iloc[10], ohlc_data["Date"].iloc[20]]
    sell_signals_y = [ohlc_data["Close"].iloc[10], ohlc_data["Close"].iloc[20]]

    fig_ohlc.add_trace(
        go.Scatter(
            x=buy_signals_x,
            y=buy_signals_y,
            mode="markers",
            marker_symbol="triangle-up",
            marker_color="green",
            marker_size=10,
            name="Buy Signal"
        )
    )
    fig_ohlc.add_trace(
        go.Scatter(
            x=sell_signals_x,
            y=sell_signals_y,
            mode="markers",
            marker_symbol="triangle-down",
            marker_color="red",
            marker_size=10,
            name="Sell Signal"
        )
    )

st.plotly_chart(fig_ohlc, use_container_width=True)

# 2) Portfolio
st.subheader("Portfolio")
portfolio_data = create_sample_portfolio_data()

fig_portfolio = go.Figure()
fig_portfolio.add_trace(
    go.Scatter(
        x=portfolio_data["Date"],
        y=portfolio_data["Portfolio"],
        mode="lines+markers",
        name="Portfolio Value"
    )
)
fig_portfolio.add_trace(
    go.Scatter(
        x=portfolio_data["Date"],
        y=portfolio_data["Benchmark"],
        mode="lines+markers",
        name="Benchmark"
    )
)
fig_portfolio.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_portfolio, use_container_width=True)

# 3) Stats
st.subheader("Stats")
stats_df = create_sample_stats()
st.table(stats_df)

# 4) Metric Stats (Radar Chart Example)
st.subheader("Metric Stats")
metrics, values = create_sample_metric_stats()

# Radar chart in Plotly
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=metrics,
    fill='toself',
    name='Metrics'
))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(values)*1.2]
        )
    ),
    showlegend=False,
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig_radar, use_container_width=True)

# ---------------
# End of the App
# ---------------
st.write("App Layout Inspired by Dash Screenshot")
