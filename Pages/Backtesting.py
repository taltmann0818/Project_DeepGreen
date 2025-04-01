import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import numpy as np
import quantstats as qs

# Custom libraries
from Components.TrainModel import torchscript_predict
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

### Page parameters --------------------------------------------------------------------------------

# Retrieve authenticator class with YAML credentials from SL session_state for logout widget on this page
if st.session_state.get("authentication_status") == None:
    st.warning("Go to the Dashboard Page to Get Started")

### Backend functions ------------------------------------------------------------------------------  


def make_predictions(input_data, model, ticker, data_window, prediction_window, model_window_size):
    # Get stock data
    out_of_sample_data, raw_stock_data = TickerData(ticker, years=1, prediction_window=prediction_window).process_all()
    # Load the model and make predictions
    preds_df = torchscript_predict(
        model_path=f"Models/{model}",
        input_df=input_data,
        device="cpu",
        window_size=model_window_size,
        target_col="shifted_prices"
    )
    
    preds_df = pd.merge(preds_df, raw_stock_data[['Open', 'High', 'Low', 'Volume','Close']], left_index=True, right_index=True, how='left')

    return preds_df



def backtesting(input_data, ticker, initial_capital, pct_change_entry, pct_change_exit, benchmark_ticker='NDAQ', rfr=0.0025):
    
    backtester = BackTesting(input_data, ticker, initial_capital, pct_change_entry=pct_change_entry,pct_change_exit=pct_change_exit)
    results, _ = backtester.run_simulation()
    trades_fig, value_fig, _ = backtester.plot_performance()

    metrics = qs.reports.metric(backtester.pf.returns(), benchmark_ticker ,mode='full', rf=rfr, display=False)

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
            tickers = st.text_input("Ticker")
            today = datetime.datetime.now() 
            data_range = st.slider("Data Range",value=(datetime.date(2000, 1, 1), datetime.date(today.year, today.month, today.day)),
                    format="MM/DD/YYYY") 
            prediction_window = st.slider("Prediction Window", 1, 10, 5)
            
            feature_select = st.multiselect('Select Model', feature_list,default=feature_list,help="This is the model used for price prediction.")
            sequence_window = st.slider("Sequence Window", 1, 200, 50)
            
            st.write("Model Hyperparameters")
            learning_rate = st.number_input("Learning Rate")
            weight_decay = st.number_input("Weight Decay")
            dropout = st.number_input("Dropout")
            epochs = st.number_input("Epochs")
            clip_size = st.number_input("Gradient Clipping Size")
            layers = st.selectbox("Batch Size", (1, 4, 8, 16, 32, 64))
            layers = st.selectbox("Hidden Size", (16, 32, 64, 256, 512))
    
            cuda_toggle = st.toggle("Use CUDA cores?")
    
            # Submit ST form button and run model training
            submit = st.form_submit_button("Train Model")

    
    with col1:
        st.subheader("Backtesting")
        if submit:
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


                
# ---------------
# End of the App
# ---------------
st.write("No analysis provided on this page or site should constitute any professional investment advice or recommendations to buy, sell, or hold any investments or investment products of any kind, and should be treated as information for educational purposes to learn more about the stock market.")
