import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import numpy as np

# Custom libraries
from Components.TrainModel import TEMPUS, DataModule
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

### Page parameters --------------------------------------------------------------------------------

# Retrieve authenticator class with YAML credentials from SL session_state for logout widget on this page
if not st.experimental_user.is_logged_in:
    st.warning("Go to the Dashboard Page to Get Started")

### Backend functions ------------------------------------------------------------------------------  


# ---------------
# Streamlit Layout
# ---------------

## Frontend ui -----------------------------------------------------------------------------------------------------------------------------
if st.experimental_user.is_logged_in:

    col1, col2 = st.columns([3, 1])
        
    with col2:
        st.link_button("View Github", 'https://github.com/taltmann0818/Project_DeepGreen')
        with st.container(border=True):
            submit = st.button("Forecast")

            # Segmented control to toggle showing the ticker input
            selection = st.segmented_control("Select Mode", options=["val_1", "val_2"], index=0)

            # Display ticker input conditionally based on selection
            if selection == "val_1":
                ticker_select = st.text_input("Ticker")
            else:
                ticker_select = None

            st.subheader("Settings")
            today = datetime.datetime.now()
            data_range = st.slider(
                "Data Range",
                value=(datetime.date(2000, 1, 1), today.date()),
                format="MM/DD/YYYY"
            )

            directory_path = 'Models/'
            models = [file.name for file in Path(directory_path).glob('*.pt') if file.is_file()]
            model_select = st.selectbox('Select Model', models, help="This is the model used for price prediction.")

            prediction_window = st.slider("Prediction Window", 1, 10, 5)
            sequence_window = st.slider("LSTM Sequence Window", 1, 200, 50)

            st.write("Backtesting Parameters")
            initial_capital = st.number_input("Initial Capital", value=1000)
            pct_change_entry = st.number_input(
                "% Change for BUY",
                help="This measures the relative gain between an equity's actual price and the predicted price for a BUY to be signaled.",
                value=5.00
            )
            pct_change_exit = st.number_input(
                "% Change for SELL",
                help="This measures the relative decrease between an equity's actual price and the predicted price for a SELL to be signaled.",
                value=2.00
            )
            risk_free_rate = st.number_input("Risk Free Rate (%)", value=0.25)

    with col1:
        st.subheader("Forecasting")
        if submit:
            missing_fields = []

            if selection == "val_1" and not ticker_select:
                missing_fields.append("Ticker")

            if not models:
                missing_fields.append("Model")

            # Check if any fields are missing
            if missing_fields:
                st.error(f"Please fill out the following fields: {', '.join(missing_fields)}")
            else:
                # Place your backtesting function call or logic here
                st.success("Running backtest with the provided inputs...")


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

       
# Sidebar: Settings


# ---------------
# End of the App
# ---------------
