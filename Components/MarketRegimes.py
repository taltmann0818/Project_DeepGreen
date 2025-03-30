import warnings
from hmmlearn import hmm
import pickle

class MarketRegimes:
    def __init__(self, data, ticker, model_pickle=None, regimes=3, **kwargs):
        self.ticker = ticker
        self.model_pickle = model_pickle
        self.regimes = regimes

        self.save_model = kwargs.get('save_model', False)
        self.getMetrics = kwargs.get('getMetrics', False)
        if self.getMetrics and self.model_pickle is not None:
            warnings.warn("getMetrics is set to True but a model_pickle was provided. Training a new model will not be performed.")
        self.covariance_type = kwargs.get('covariance_type', "full")
        self.n_iter = kwargs.get('n_iter', 100)
        self.random_state = kwargs.get('random_state', 5)

        if data is not None:
            data['Return'] = data["Close"].pct_change()
            data['Range'] = data['High'] - data['Low']
            data.dropna(inplace=True)
            # Prepare the feature matrix for HMM (e.g., returns, range, volume as features)
            self.features = data[['Return', 'Range', 'Volume']].values

    def train_hmm_model(self):
        model = hmm.GaussianHMM(n_components=self.regimes, covariance_type=self.covariance_type, n_iter=self.n_iter, random_state=self.random_state)
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

        if self.getMetrics and self.model_pickle is None:
            return hidden_states, converged, log_likelihood, bic, aic
        else:
            return hidden_states