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

# ---------------
# Streamlit Layout
# ---------------

## Frontend ui -----------------------------------------------------------------------------------------------------------------------------
if st.experimental_user.is_logged_in:

    col1, col2 = st.columns([3, 1])
        
    with col2:
        st.write("")
        with st.container(border=True):
            submit = st.button("Forecast",icon=":material/query_stats:")

            # Segmented control to toggle showing the ticker input
            modes = ["Scan", "Single"]
            mode_selection = st.segmented_control(
                "Mode", modes, selection_mode="single", help="This is the mode used for price forecasting."
            )

            # Display ticker input conditionally based on selection
            if mode_selection == "Single":
                ticker_select = st.text_input("Ticker")
            if mode_selection == "Scan":
                ticker_select = st.selectbox("Index",['NASDAQ','S&P500','RUSSELL2000','DOWJONES'])
                if ticker_select is not None:
                    tickers = get_index_tickers(ticker_select)
                median_ticker = int(np.round(np.median([index for index, _ in enumerate(tickers)])) + 1)
                scan_size = st.slider("Scan Size", 1, len(tickers), median_ticker)
            else:
                ticker_select = None

            options_toggle = st.toggle("Include options?")

            st.subheader("Settings")
            models = [file.name for file in Path('Models/').glob('*.pt') if file.is_file()]
            model_select = st.selectbox('Select Model', models, help="This is the model used for price prediction.")
            prediction_window = st.slider("Prediction Window", 1, 10, 5)
            sequence_window = st.slider("LTSM Sequence Window", 1, 200, 50)

            st.write("Backtesting Parameters")
            pct_change_entry = st.number_input("% Change for BUY",
                                               help="This measures the relative gain between an equity's actual price and the predicted price for a BUY to be signalled.",
                                               value=5.00)
            pct_change_exit = st.number_input("% Change for SELL",
                                              help="This measures the relative decrease between an equity's actual price and the predicted price for a SELL to be signalled.",
                                              value=2.00)

    with col1:
        st.subheader("Forecasting")
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