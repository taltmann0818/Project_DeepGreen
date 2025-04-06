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
        with st.form('train_form'):
            st.subheader("Settings")
            tickers = st.text_input("Ticker(s)",help="To enter multiple tickers, seperate with a comma like ',' You can also enter 'NASDAQ' or 'S&P' for all the stocks in the index.")
            today = datetime.datetime.now() 
            data_range = st.slider("Data Range",value=(datetime.date(2000, 1, 1), datetime.date(today.year, today.month, today.day)),
                    format="MM/DD/YYYY") 
            prediction_window = st.slider("Prediction Window", 1, 10, 5)
            
            
            feature_list = ['EMA20','EMA50','EMA100','EMA200','RSI','MACD','REGIME']
            feature_select = st.multiselect('Select Features', feature_list,default=feature_list,help="These are the predictor variables included in the model.")
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
        st.subheader("TEMPUS Training")
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



        
# Sidebar: Settings



# ---------------
# End of the App
# ---------------
