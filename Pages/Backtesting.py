import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import numpy as np
import quantstats as qs
from pathlib import Path

# Custom libraries
from Components.TrainModel import torchscript_predict
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

### Page parameters --------------------------------------------------------------------------------

# Retrieve authenticator class with YAML credentials from SL session_state for logout widget on this page
if st.session_state.get("authentication_status") == None:
    st.warning("Go to the Dashboard Page to Get Started")

### Backend functions ------------------------------------------------------------------------------  


def make_predictions(model, ticker, data_window, prediction_window, model_window_size):
    # Get stock data
    
    out_of_sample_data, raw_stock_data = TickerData(ticker, years=1, prediction_window=prediction_window,
                                                    start_date=data_window[0],
                                                    end_date=data_window[1]).process_all()
    if out_of_sample_data is None:
        raise ValueError("No data retrieved!")
    
    # Load the model and make predictions
    preds_df = torchscript_predict(
        model_path=f"Models/{model}",
        input_df=out_of_sample_data,
        device="cpu",
        window_size=model_window_size,
        target_col="shifted_prices"
    )
    
    preds_df = pd.merge(preds_df, raw_stock_data[['Open', 'High', 'Low', 'Volume','Close']], left_index=True, right_index=True, how='left')

    return preds_df



def backtesting(input_data, ticker, initial_capital, pct_change_entry, pct_change_exit, benchmark_ticker='NDAQ', rfr=0.25):
    
    backtester = BackTesting(input_data, ticker, initial_capital, pct_change_entry=pct_change_entry,pct_change_exit=pct_change_exit)
    backtester.run_simulation()
    trades_fig, value_fig, _ = backtester.plot_performance()

    metrics = np.array(qs.reports.metrics(backtester.pf.returns(), benchmark_ticker ,mode='full', rf=rfr/100, display=False))

    return trades_fig, value_fig, metrics

# ---------------
# Streamlit Layout
# ---------------

## Frontend ui -----------------------------------------------------------------------------------------------------------------------------
if st.session_state.get("authentication_status"):

    
    col1, col2 = st.columns([3, 1])
        
    with col2:
        st.link_button("View Github", 'https://github.com/taltmann0818/Project_DeepGreen')
        with st.form('backtest_form'):
            st.subheader("Settings")
            ticker_select = st.text_input("Ticker")
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
    
            # Submit ST form button and run model training
            submit = st.form_submit_button("Backtest")
    
    with col1:
        st.subheader("Backtesting")
        if submit:
            spinner_strings = ["Running the Bulls...","Poking the Bear..."]
            with st.spinner(np.random.choice(spinner_strings)):
                predictions_df = make_predictions(model_select, ticker_select, data_range, prediction_window, sequence_window)
                trades_fig, value_fig, metrics = backtesting(predictions_df, ticker_select, initial_capital, pct_change_entry, pct_change_exit, benchmark_ticker='NDAQ', rfr=risk_free_rate)
            
            with st.container(border=True):
                st.subheader("Portfolio")
                
                st.plotly_chart(trades_fig)
                st.plotly_chart(value_fig)
            
            with st.container(border=True):
                st.subheader("Metrics")

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
                              'Benchmark (NDAQ)': [bm_cagr, bm_return, bm_vol, bm_sharpe, bm_serenity,bm_sortino,bm_dVaR,bm_avgdrawdwn,bm_maxdrawdwn,bm_drawdwndays]}).set_index('Metric Name')


                mcol1, mcol2, mcol3 = st.columns(3)
                cagr_delta = np.round(strat_cagr - bm_cagr,2)
                sharpe_delta = np.round(strat_sharpe - bm_sharpe, 2)
                vol_delta = np.round(strat_vol - bm_vol, 2)
                mcol1.metric("CAGR", f"{np.round(strat_cagr,2)} %", cagr_delta)
                mcol2.metric("Sharpe Ratio", strat_sharpe, strat_sharpe-bm_sharpe)
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