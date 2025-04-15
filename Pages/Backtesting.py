import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import numpy as np
import quantstats as qs
from pathlib import Path
import random
import time

# Custom libraries
from Components.TrainModel import torchscript_predict
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

### Page parameters --------------------------------------------------------------------------------

# Retrieve authenticator class with YAML credentials from SL session_state for logout widget on this page
if not st.experimental_user.is_logged_in:
    st.warning("Go to the Dashboard Page to Get Started")

### Backend functions ------------------------------------------------------------------------------  

def get_index_tickers(index):
    if index == 'NASDAQ':
        tickers = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        tickers = tickers.iloc[:, [1]].to_numpy().flatten() # Clean up the dataframe
    elif index == 'S&P500':
        tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = tickers.iloc[:, [0]].to_numpy().flatten() # Clean up the dataframe
    elif index == 'RUSSELL1000':
        tickers = pd.read_html("https://en.wikipedia.org/wiki/Russell_1000_Index")[3]
        tickers = tickers.iloc[:, [1]].to_numpy().flatten() # Clean up the dataframe
    elif index == 'DOWJONES':
        tickers = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")[2]
        tickers = tickers.iloc[:, [2]].to_numpy().flatten() # Clean up the dataframe
    else:
        tickers = []

    return tickers

def make_predictions(model, ticker, data_window, prediction_window, model_window_size):
    # Get stock data
    if model == 'Tempus_v2.1.pt':
        indicators = ['ema_20', 'ema_50', 'ema_200', 'stoch_rsi', 'macd', 'b_percent', 'keltner_lower', 'keltner_upper','adx','Close']
    elif model == 'Tempus_v2.2.pt':
        indicators = ['ema_20', 'ema_50', 'ema_200', 'stoch_rsi', 'macd', 'b_percent', 'keltner_lower', 'keltner_upper','adx']
    else:
        indicators = ['ema_20', 'ema_50', 'ema_100', 'stoch_rsi', 'macd', 'State', 'Close']

    out_of_sample_data, raw_stock_data = TickerData(ticker, indicators, years=1, prediction_window=prediction_window,start_date=data_window[0],end_date=data_window[1], prediction_mode=True).process_all()
    if out_of_sample_data is None:
        raise ValueError("No data retrieved!")
    
    # Load the model and make predictions
    preds_df = torchscript_predict(
        model_path=f"Models/{model}",
        input_df=out_of_sample_data,
        device="cpu",
        window_size=model_window_size,
        prediction_mode=True
    )
    
    preds_df = pd.merge(preds_df, raw_stock_data[['Open', 'High', 'Low', 'Volume','Close']], left_index=True, right_index=True, how='left')

    return preds_df


def backtesting(input_data, ticker, initial_capital, pct_change_entry, pct_change_exit, benchmark_toggle, rfr=0.25):
    
    backtester = BackTesting(input_data, ticker, initial_capital, pct_change_entry=pct_change_entry,pct_change_exit=pct_change_exit)
    backtester.run_simulation()
    trades_fig, value_fig, _ = backtester.plot_performance()

    if benchmark_toggle:
        metrics = np.array(qs.reports.metrics(backtester.pf.returns(), 'NDAQ', mode='full', rf=rfr / 100, display=False))
    else:
        metrics = np.array(qs.reports.metrics(backtester.pf.returns(), ticker, mode='full', rf=rfr / 100, display=False))

    return trades_fig, value_fig, metrics

def multi_backtesting(tickers, initial_capital, model, data_window, prediction_window, model_window_size, pct_change_entry, pct_change_exit):

    returns = []
    sharpe_ratios = []
    VaRs = []
    spinner_strings = ["Running the Bulls...", "Poking the Bear..."]
    progress_text = f"{np.random.choice(spinner_strings)} Please wait."
    my_bar = st.progress(0, text=progress_text)
    total_tickers = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            preds_df = make_predictions(model, ticker, data_window, prediction_window, model_window_size)

            backtester = BackTesting(preds_df, ticker, initial_capital, pct_change_entry=pct_change_entry,pct_change_exit=pct_change_exit)
            backtester.run_simulation()
            bt_results = pd.DataFrame(backtester.pf.returns())
            bt_results['cumulative_return'] = np.array(((1 + bt_results[0]).cumprod() - 1) * 100)
            bt_results['ticker'] = ticker
            returns.append(bt_results)

            # Other portfolio metrics
            sharpe_ratios.append(backtester.pf.sharpe_ratio())
            VaRs.append(backtester.pf.value_at_risk())

        except ValueError as e:
            if str(e) == "No data retrieved!":
                warning_placeholder = st.empty()
                warning_placeholder.warning(f"Unable to backtest {ticker}") # Display the warning message in the placeholder
                time.sleep(2) # Wait for 2 seconds
                warning_placeholder.empty() # Clear the container
                continue
            else:
                raise # Re-raise other ValueError exceptions

        per_done = np.round((idx / total_tickers) * 100, 2)
        my_bar.progress(idx / total_tickers, text=f"{per_done}% of tickers backtested")

    my_bar.empty()
    if not returns:
        raise ValueError("No valid data retrieved for any of the tickers!")
    returns = pd.concat(returns, ignore_index=False)

    # Create an interactive plot using Plotly
    fig = px.line(
        returns.reset_index(),
        x='index',
        y='cumulative_return',
        color='ticker',
        title='Cumulative Returns by Ticker',
        labels={'index': 'Date', 'cumulative_return': 'Cumulative Return'}
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        showlegend=False,
        height=600,
        template='ggplot2',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
    )

    # Count positive and negative returns
    last_returns = returns.groupby('ticker')['cumulative_return'].last()
    positive_count = sum(last_returns > 0)
    negative_count = sum(last_returns <= 0)

    # Convert to DataFrame for visualization
    last_returns_df = pd.DataFrame(last_returns).reset_index()
    last_returns_df.columns = ['Ticker', 'Final Return']
    last_returns_df.sort_values('Final Return', ascending=False, inplace=True)

    # Create a simple pie chart showing the proportion
    fig_pie = px.pie(
        values=[positive_count, negative_count],
        names=['Positive', 'Negative'],
        title='Proportion of Tickers with Positive vs Negative Returns',
        color_discrete_sequence=['green', 'red'],
        template='ggplot2',
    )

    fig_pie.update_traces(textinfo='percent+label').update_layout(showlegend=False)

    # Build a  metrics df
    metrics_df = pd.DataFrame({'Metric Name': ['Cumulative Return (%)','Sharpe Ratio','Value-at-Risk (%)'],
                               'Average': [np.average(last_returns),np.average(sharpe_ratios),np.average(VaRs)],
                               'Minimum': [np.min(last_returns),np.min(sharpe_ratios),np.min(VaRs)],
                               'Maximum': [np.max(last_returns),np.max(sharpe_ratios),np.max(VaRs)],
                               'Std. Dev': [np.std(last_returns),np.std(sharpe_ratios),np.std(VaRs)],
                               }).set_index('Metric Name')

    return fig, fig_pie, metrics_df

# ---------------
# Streamlit Layout
# ---------------

## Frontend ui -----------------------------------------------------------------------------------------------------------------------------
if st.experimental_user.is_logged_in:

    col1, col2 = st.columns([3, 1])
        
    with col2:
        with st.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            submit = st.button("Backtest",icon=":material/query_stats:")

            # Segmented control to toggle showing the ticker input
            modes = ["Single", "Multi"]
            mode_selection = st.segmented_control(
                "Mode", modes, selection_mode="single", help="This is the mode used for backtesting."
            )

            # Display ticker input conditionally based on selection
            if mode_selection == "Single":
                ticker_select = st.text_input("Ticker")
                benchmark_toggle = st.toggle("Use index benchmark?")
            elif mode_selection == "Multi":
                ticker_select = st.selectbox("Index",['NASDAQ','S&P500','RUSSELL1000','DOWJONES'])
                if ticker_select is not None:
                    tickers = get_index_tickers(ticker_select)
                median_ticker = int(np.round(np.median([index for index, _ in enumerate(tickers)]))+1)
                sample_size = st.slider("Sample Size", 1, len(tickers), median_ticker)
            else:
                ticker_select = None

            st.subheader("Settings")
            today = datetime.datetime.now() 
            data_range = st.slider("Data Range",value=(datetime.date(2000, 1, 1), datetime.date(today.year, today.month, today.day)),format="MM/DD/YYYY") 
            
            directory_path = 'Models/'
            models = [file.name for file in Path(directory_path).glob('*.pt') if file.is_file()]
            model_select = st.selectbox('Select Model', models,help="This is the model used for price prediction.")
            prediction_window = st.slider("Prediction Window", 1, 10, 5)
            sequence_window = st.slider("LTSM Sequence Window", 1, 200, 50)
            
            st.write("Backtesting Parameters")
            initial_capital = st.number_input("Initial Capital",value=1000)
            pct_change_entry = st.number_input("% Change for BUY",help="This measures the relative gain between an equity's actual price and the predicted price for a BUY to be signalled.",value=5.00)
            pct_change_exit = st.number_input("% Change for SELL",help="This measures the relative decrease between an equity's actual price and the predicted price for a SELL to be signalled.",value=2.00)
            risk_free_rate = st.number_input("Risk Free Rate (%)",value=0.25)

            #cuda_toggle = st.toggle("Use CUDA cores?")
    
    with col1:
        st.subheader("Backtesting")
        if submit:
            missing_fields = []
            if mode_selection is None:
                missing_fields.append("Mode")
            if (mode_selection == "Scan" or mode_selection == "Single") and ticker_select is None:
                missing_fields.append("Ticker")
            if model_select is None:
                missing_fields.append("Model")
            if prediction_window is None:
                missing_fields.append("Prediction Window")
            if sequence_window is None:
                missing_fields.append("LTSM Sequence Window")
            if initial_capital is None:
                missing_fields.append("Initial Capital")
            if pct_change_entry is None:
                missing_fields.append("% Change for BUY")
            if pct_change_exit is None:
                missing_fields.append("% Change for SELL")

            # Check if any fields are missing
            if missing_fields:
                st.error(f"Please fill out the following fields: {', '.join(missing_fields)}")

            else:
                if mode_selection == "Multi":
                    sampled_tickers = random.sample(list(tickers), sample_size)
                    fig, fig_pie, metrics_df = multi_backtesting(sampled_tickers,initial_capital,model_select,data_range,prediction_window,sequence_window,pct_change_entry,pct_change_exit)
                    avg_cumreturn = np.round(np.array(metrics_df)[0][0],2)
                    avg_sharpe = np.round(np.array(metrics_df)[1][0],2)
                    avg_VaR = np.round(np.array(metrics_df)[2][0],2)

                elif mode_selection == "Single":
                    spinner_strings = ["Running the Bulls...","Poking the Bear..."]
                    with st.spinner(np.random.choice(spinner_strings)):
                        predictions_df = make_predictions(model_select, ticker_select, data_range, prediction_window, sequence_window)
                        trades_fig, value_fig, metrics = backtesting(predictions_df, ticker_select, initial_capital, pct_change_entry, pct_change_exit, benchmark_toggle, rfr=risk_free_rate)

                with st.container(border=True):
                    st.subheader("Portfolio")

                    if mode_selection == "Multi":
                        st.plotly_chart(fig)

                        st.write("Placeholder for chart that shows improvement over baseline (selected index) by ticker")
                        st.bar_chart(np.random.randn(50, 3))

                        st.plotly_chart(fig_pie)

                    elif mode_selection == "Single":
                        st.plotly_chart(trades_fig)
                        st.plotly_chart(value_fig)

                with st.container(border=True):
                    st.subheader("Metrics")

                    if mode_selection == "Multi":
                        mcol1, mcol2, mcol3 = st.columns(3)
                        mcol1.metric("Avg. Return (%)", avg_cumreturn)
                        mcol2.metric("Avg. Sharpe Ratio", avg_sharpe)
                        mcol3.metric("Avg. Value-at-Risk (%)", avg_VaR)

                        st.table(metrics_df)

                        st.write("Note: Cumulative returns are calculated at the last day in the data range.")

                    elif mode_selection == "Single":

                        strat_cagr, bm_cagr = metrics[5][1], metrics[5][0]*100
                        strat_return, bm_return = metrics[5][1]*100, metrics[5][0]*100

                        #Risk metrics
                        strat_sharpe, bm_sharpe = metrics[6][1], metrics[6][0]
                        strat_vol, bm_vol = metrics[16][1]*100, metrics[16][0]*100
                        strat_serenity, bm_serenity = metrics[59][1], metrics[59][0]
                        strat_sortino, bm_sortino = metrics[10][1], metrics[10][0]

                        #VaR
                        strat_dVaR, bm_dVaR= metrics[27][1]*100, metrics[27][0]*100

                        #Extreme Risk Metrics
                        strat_avgdrawdwn, bm_avgdrawdwn = metrics[55][1]*100, metrics[55][0]*100
                        strat_maxdrawdwn, bm_maxdrawdwn = metrics[14][1]*100, metrics[14][0]*100
                        strat_drawdwndays, bm_drawdwndays = metrics[15][1], metrics[15][0]
                        # Building metrics df
                        metrics_df = pd.DataFrame({'Metric Name': ['Ann. Return (CAGR) %','Cumulative Return %',
                                                      'Ann. Volatility %',f'Sharpe Ratio (Rf= {risk_free_rate}%)','Serenity Ratio','Sortino Ratio',
                                                     'Daily Value-at-Risk %',
                                                     'Avg Drawdown %','Max Drawdown %','Max Time-under-water (days)'],
                                      f'Strategy ({ticker_select})': [strat_cagr, strat_return, strat_vol,strat_sharpe,strat_serenity,strat_sortino,strat_dVaR,strat_avgdrawdwn,strat_maxdrawdwn,strat_drawdwndays],
                                      f'Benchmark ({ticker_select})': [bm_cagr, bm_return, bm_vol, bm_sharpe, bm_serenity,bm_sortino,bm_dVaR,bm_avgdrawdwn,bm_maxdrawdwn,bm_drawdwndays]}).set_index('Metric Name')


                        mcol1, mcol2, mcol3 = st.columns(3)
                        cagr_delta = np.round(strat_cagr - bm_cagr,2)
                        sharpe_delta = np.round(strat_sharpe - bm_sharpe, 2)
                        vol_delta = np.round(strat_vol - bm_vol, 2)
                        mcol1.metric("CAGR", f"{np.round(strat_cagr,2)} %", cagr_delta)
                        mcol2.metric("Sharpe Ratio", strat_sharpe, sharpe_delta)
                        mcol3.metric("Volatility (ann.)", f"{np.round(strat_vol,2)} %", vol_delta, delta_color="inverse")

                        st.table(metrics_df)

        else:
            with st.container(border=True):
                st.write("Enter the params and click the button to get results!")

expander = st.expander("See metric definitions")
expander.write('''
    - <b>CAGR:</b> The compound annual growth rate is the rate of return that an investment would need to have every year in order to grow from its beginning balance to its ending balance, over a given time interval. The CAGR assumes that any profits were reinvested at the end of each period of the investment’s life span. The compound annual growth rate isn’t a true return rate, but rather a representational figure. It is essentially a number that describes the rate at which an investment would have grown if it had grown at the same rate every year and the profits were reinvested at the end of each year. For stock market investors, this can be particularly useful in comparing the performance of different stocks.

    - <b>Sharpe Ratio:</b> The Sharpe ratio compares the return of an investment with its risk. It's a mathematical expression of the insight that excess returns over a period of time may signify more volatility and risk, rather than investing skill. The Sharpe ratio is one of the most widely used methods for measuring risk-adjusted relative returns. It compares a fund's historical or projected returns relative to an investment benchmark with the historical or expected variability of such returns.

    - <b>Serentiy Ratio:</b> The Serenity Ratio is an alternative measure to the Sharpe Ratio that accounts for extreme risk. While the latter only divides return premium by the annualized volatility, the Serenity Ratio uses the Ulcer Index and a Pitfall Indicator (PI) as risk measures to quantify the tendency of a fund to be “stuck” in drawdown.

    - <b>Sortino Ratio:</b> The Sortino ratio is a variation of the Sharpe ratio that differentiates harmful volatility from total overall volatility by using the asset's standard deviation of negative portfolio returns—downside deviation—instead of the total standard deviation of portfolio returns. The Sortino ratio is a useful way for investors, analysts, and portfolio managers to evaluate an investment's return for a given level of bad risk. Since this ratio uses only the downside deviation as its risk measure, it addresses the problem of using total risk, or standard deviation, which is important because upside volatility is beneficial to investors and isn't a factor most investors worry about.

    - <b>Daily Value-at-Risk:</b> Value at Risk (VaR) has been called the "new science of risk management," and is a statistic that is used to predict the greatest possible losses over a specific time frame.

    - <b>Drawdown:</b> A drawdown is the peak-to-trough decline of an investment, trading account, or fund during a specific period. It can be used to measure an investment's historical risk, compare the performance of different funds, or monitor a portfolio's performance.
''', unsafe_allow_html=True)
# ---------------
# End of the App
# ---------------
st.write("No analysis provided on this page or site should constitute any professional investment advice or recommendations to buy, sell, or hold any investments or investment products of any kind, and should be treated as information for educational purposes to learn more about the stock market.")