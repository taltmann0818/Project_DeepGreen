import warnings
from hmmlearn import hmm
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MarketRegimes:
    def __init__(self, data, model_pickle=None, regime_number=3, **kwargs):
        self.model_pickle = model_pickle
        self.regime_number = regime_number
        self.data = data

        self.save_model = kwargs.get('save_model', False)
        self.getMetrics = kwargs.get('getMetrics', False)
        if self.getMetrics and self.model_pickle is not None:
            warnings.warn("getMetrics is set to True but a model_pickle was provided. Training a new model will not be performed.")
        self.covariance_type = kwargs.get('covariance_type', "full")
        self.n_iter = kwargs.get('n_iter', 100)
        self.random_state = kwargs.get('random_state', 5)

        if self.data is not None:
            self.data['Return'] = self.data["Close"].pct_change()
            self.data['Range'] = self.data['High'] - data['Low']
            self.data.dropna(inplace=True)
            # Prepare the feature matrix for HMM (e.g., returns, range, volume as features)
            self.features = self.data[['Return', 'Range', 'Volume']].values

    def train_hmm_model(self):
        model = hmm.GaussianHMM(n_components=self.regime_number, covariance_type=self.covariance_type, n_iter=self.n_iter, random_state=self.random_state)
        model.fit(self.features)  # Train the HMM on the features

        # Get model training metrics
        converged = model.monitor_.converged
        log_likelihood, bic, aic = model.score(self.features), model.bic(self.features), model.aic(self.features)

        return model, converged, log_likelihood, bic, aic

    def run_regime_detection(self):
        if self.model_pickle is not None:
            with open(f"Models/{self.model_pickle}", "rb") as file:
                self.model = pickle.load(file)
        else:
            if self.getMetrics:
                self.model, converged, log_likelihood, bic, aic = self.train_hmm_model()
            else:
                self.model, _, _, _, _ = self.train_hmm_model()
            if self.save_model:
                with open("Models/hmm_model.pkl", "wb") as file:
                    pickle.dump(self.model, file)

        hidden_states = self.model.predict(self.features)
        self.data['State'] = hidden_states

        # Map state index to regime name
        state_order = np.argsort(self.model.means_[:, 0])  # indices of states sorted by mean return
        state_labels = {}
        state_labels[state_order[0]] = "Bearish"  # lowest mean return
        state_labels[state_order[1]] = "Neutral"  # middle
        state_labels[state_order[2]] = "Bullish"  # highest mean return
        # Create a new column with the regime label for each day
        self.data['Regime'] = self.data['State'].map(state_labels)

        if self.getMetrics and self.model_pickle is None:
            return self.data, converged, log_likelihood, bic, aic
        else:
            return self.data

    def visualize_regime_detection(self, data, ticker):
        #ticker = np.random.choice(stock_data['Ticker'].unique()) # Randomly sample a ticker
        if ticker is not None:
            fig_data = self.data[data['Ticker'] == ticker]

            # Define colors for each regime
            fig = make_subplots(rows=3, cols=1,
                                subplot_titles=('Bearish Hidden State', 'Neutral Hidden State', 'Bullish Hidden State'))

            bearish_mask = fig_data['Regime'] == "Bearish"
            fig.add_trace(go.Scatter(x=fig_data.index[bearish_mask], y=fig_data['Close'][bearish_mask], mode="markers",
                                     name="Bearish", line=dict(color="red")), row=1, col=1)
            neutral_mask = fig_data['Regime'] == "Neutral"
            fig.add_trace(go.Scatter(x=fig_data.index[neutral_mask], y=fig_data['Close'][neutral_mask], mode="markers",
                                     name="Neutral", line=dict(color="orange")), row=2, col=1)
            bullish_mask = fig_data['Regime'] == "Bullish"
            fig.add_trace(go.Scatter(x=fig_data.index[bullish_mask], y=fig_data['Close'][bullish_mask], mode="markers",
                                     name="Bullish", line=dict(color="green")), row=3, col=1)

            fig.update_layout(
                title=f'Market Regimes Identified by HMM for {ticker}',
                xaxis_title="Date",
                height=800,
                yaxis_title="Stock Price (USD)",
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            return fig