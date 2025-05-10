import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import numpy as np
from pathlib import Path

# Custom libraries
from Components.TrainModel import TEMPUS, DataModule
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

### Page parameters --------------------------------------------------------------------------------

# Retrieve authenticator class with YAML credentials from SL session_state for logout widget on this page
if not st.user.is_logged_in:
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
    elif model == 'Tempus_v2.3.pt':
        indicators = ['ema_20', 'ema_50', 'ema_200', 'stoch_rsi', 'macd', 'b_percent', 'keltner_lower', 'keltner_upper','adx','Close']
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

def multi_forecasting(tickers, initial_capital, model, data_window, prediction_window, model_window_size, pct_change_entry, pct_change_exit,spinner_string):

    forecast_results = {
        'Ticker': [], 'Name': [], 'Sector': [], 'Per. Change': [],'52WeekHigh': [], '52WeekLow': [],
        'eps': [], 'deRatio': [], 'roe': [], 'roa': [],'peRatio': [], 'psRatio': [], 'evEBITDA': []
    }

    my_bar = st.progress(0, text=spinner_string)
    total_tickers = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            preds_df = make_predictions(model, ticker, data_window, prediction_window, model_window_size)

  

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

# ---------------
# Streamlit Layout
# ---------------

## Frontend ui -----------------------------------------------------------------------------------------------------------------------------
if st.user.is_logged_in:

    col1, col2 = st.columns([3, 1])
        
    spinner_strings = ["Running the Bulls...","Poking the Bear...","Buying the Dip...","Chasing the Rally...","Playing the Spread...","Hunting the Bubble...","Fighting the Fed..."]
        
    with col2:
        with st.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            submit = st.button("Backtest",icon=":material/query_stats:")

            # Segmented control to toggle showing the ticker input
            modes = ["Single", "Scan"]
            mode_selection = st.segmented_control(
                "Mode", modes, selection_mode="single", help="This is the mode used for forecasting."
            )

            # Display ticker input conditionally based on selection
            if mode_selection == "Single":
                ticker_select = st.text_input("U.S. Equity")
                benchmark_toggle = st.toggle("Use index benchmark?")
                if benchmark_toggle:
                    index_benchmark = st.selectbox("Index",['NDAQ','INX','RUI','DJI'])
            elif mode_selection == "Scan":
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
            
            model_select = st.selectbox('Select Model', [file.name for file in Path('Models/').glob('*.pt') if file.is_file()], help="This is the model used for price prediction.")
            prediction_window = st.slider("Prediction Window", 1, 10, 5)
            sequence_window = st.slider("LTSM Sequence Window", 1, 200, 50)
            
            st.write("Backtesting Parameters")
            initial_capital = st.number_input("Initial Capital",value=1000)
            pct_change_entry = st.number_input("% Change for BUY",help="This measures the relative gain between an equity's actual price and the predicted price for a BUY to be signalled.",value=5.00)
            pct_change_exit = st.number_input("% Change for SELL",help="This measures the relative decrease between an equity's actual price and the predicted price for a SELL to be signalled.",value=2.00)
            risk_free_rate = st.number_input("Risk Free Rate (%)",value=0.25)

    with col1:
        st.subheader("Forecasting")
        if submit:
            missing_fields = []
            if mode_selection is None:
                missing_fields.append("Mode")
            if ticker_select is None:
                missing_fields.append("Equity")
            if model_select is None:
                missing_fields.append("Model")
            if prediction_window is None:
                missing_fields.append("Prediction Window")
            if sequence_window is None:
                missing_fields.append("LTSM Sequence Window")
            if pct_change_entry is None:
                missing_fields.append("% Change for BUY")
            if pct_change_exit is None:
                missing_fields.append("% Change for SELL")

            # Check if any fields are missing
            if missing_fields:
                st.error(f"Please fill out the following fields: {', '.join(missing_fields)}")

            else:
                with st.container(border=True):
                    st.subheader("Train History")

                    st.bar_chart(np.random.randn(50, 3))

                with st.container(border=True):
                    st.subheader("Predictions")

                    st.bar_chart(np.random.randn(50, 3))

                with st.container(border=True):
                    st.subheader("Metrics")

                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Loss", "70")
                    mcol2.metric("MSE", "9")
                    mcol3.metric("MAPE", "5%")

        else:
            with st.container(border=True):
                st.write("Enter the params and click the button to get results!")

       
# Sidebar: Settings


# ---------------
# End of the App
# ---------------
st.write("No analysis provided on this page or site should constitute any professional investment advice or recommendations to buy, sell, or hold any investments or investment products of any kind, and should be treated as information for educational purposes to learn more about the stock market.")