import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def prepare_data_for_model_input(prices, ma):
   prices[f'close_ma'] = prices.rolling(ma).mean()
   prices[f'close_log_return'] = np.log(prices[f'close_ma']/prices[f'close_ma'].shift(1)).dropna()
   prices.dropna(inplace = True)
   prices_array = np.array([[q] for q in prices[f'close_log_return'].values])

   return prices, prices_array

class RegimeDetection:
   def get_regimes_hmm(self, input_data, params):
       hmm_model = self.initialise_model(GaussianHMM(), params).fit(input_data)
       return hmm_model

   def initialise_model(self, model, params):
       for parameter, value in params.items():
           setattr(model, parameter, value)
       return model

def feed_forward_training(model, params, prices, split_index, retrain_step):
   # train/test split and initial model training
   init_train_data = prices[:split_index]
   test_data = prices[split_index:]
   rd_model = model(init_train_data, params)

   # predict the state of the next observation
   states_pred = []
   for i in range(math.ceil(len(test_data))):
       split_index += 1
       preds = rd_model.predict(prices[:split_index]).tolist()
       states_pred.append(preds[-1])

       # retrain the existing model
       if i % retrain_step == 0:
           rd_model = model(prices[:split_index], params)

   return states_pred

def plot_hidden_states(hidden_states, prices_df):
   colors = ['blue', 'green', 'orange']
   n_components = len(np.unique(hidden_states))
   fig = go.Figure()
   for i in range(n_components):
       mask = hidden_states == i
       print('Number of observations for State ', i,":", len(prices_df.index[mask]))

       fig.add_trace(go.Scatter(x=prices_df.index[mask], y=prices_df["close"][mask],
                   mode='markers',  name='Hidden State ' + str(i), marker=dict(size=4,color=colors[i])))

   fig.update_layout(template="ggplot2", height=400, width=900, legend=dict(
           yanchor="top", y=0.99, xanchor="left",x=0.01), margin=dict(l=20, r=20, t=20, b=20)).show()

   training_data, raw_stock_data = TickerData(['PLTR'], years=5, prediction_window=5,
                                              indicator_list=indicators).process_all()

   prices, prices_array = prepare_data_for_model_input(raw_stock_data[['close']], 5)

   split_index = np.where(prices.index > '2023-01-01')[0][0]
   regime_detection = RegimeDetection()
   model_hmm = regime_detection.get_regimes_hmm
   params = {'n_components': 2, 'covariance_type': 'full', 'random_state': 100}
   states_pred_hmm, bic = feed_forward_training(model_hmm, params, prices_array, split_index, 20)
   plot_hidden_states(np.array(states_pred_hmm), prices[['close']][split_index:])
   print(bic)