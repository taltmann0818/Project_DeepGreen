import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MarketRegimes:
    def __init__(
        self,
        data: pd.DataFrame,
        feature_window: int = 20,
        ma_short: int = 50,
        ma_long: int = 200,
        min_regime_length: int = 5,
        macro_data: pd.DataFrame = None,
        model_pickle: str = None,
        regime_number: int = 3,
        n_iter: int = 100,
        covariance_type: str = "full",
        random_state: int = 5,
        n_starts: int = 5,
        save_model: bool = False,
        getMetrics: bool = False,
    ):
        """
        data: must contain columns ['Date', 'Close', 'High', 'Low', 'Volume']
        macro_data: optional DataFrame indexed by Date with columns like 'VIX', 'YieldCurve'
        feature_window: lookback for rolling volatility
        ma_short, ma_long: for trend feature
        min_regime_length: minimum days to enforce regime persistence
        n_starts: number of random initializations to pick best HMM
        """
        self.model_pickle = model_pickle
        self.save_model = save_model
        self.getMetrics = getMetrics
        self.regime_number = regime_number
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_starts = n_starts
        self.min_regime_length = min_regime_length

        # 1) Prepare data and merge macros
        df = data.copy()
        df['Return'] = np.log(df['Close']).diff()
        df['Volatility'] = df['Return'].rolling(window=feature_window).std()
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        df['MA_diff'] = (
            df['Close'].rolling(window=ma_short).mean() -
            df['Close'].rolling(window=ma_long).mean()
        ) / df['Close']
        if macro_data is not None:
            df = df.join(macro_data, how='left')
        df.dropna(inplace=True)

        # 2) Feature scaling
        features = df[['Return', 'Volatility', 'Range', 'MA_diff', 'Volume']].values
        self.scaler = StandardScaler().fit(features)
        self.features = self.scaler.transform(features)
        self.data = df.reset_index(drop=True)

    def _smooth_states(self, states: pd.Series) -> pd.Series:
        """Enforce a minimum regime length by merging short runs into the previous regime."""
        sm = states.copy()
        i = 0
        while i < len(sm):
            j = i + 1
            while j < len(sm) and sm.iloc[j] == sm.iloc[i]:
                j += 1
            run_len = j - i
            if run_len < self.min_regime_length and i > 0:
                # merge into previous regime
                sm.iloc[i:j] = sm.iloc[i - 1]
            i = j
        return sm

    def train_hmm_model(self):
        """Fit multiple random-start HMMs and keep the best by log-likelihood."""
        best_score = -np.inf
        best_model = None

        for seed in range(self.random_state, self.random_state + self.n_starts):
            model = hmm.GaussianHMM(
                n_components=self.regime_number,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=seed,
                verbose=False
            )
            model.fit(self.features)
            score = model.score(self.features)
            if score > best_score:
                best_score = score
                best_model = model

        # compute information criteria on best model
        ll = best_model.score(self.features)
        bic = best_model.bic(self.features)
        aic = best_model.aic(self.features)
        conv = best_model.monitor_.converged

        return best_model, conv, ll, bic, aic

    def run_regime_detection(self):
        # load or train
        if self.model_pickle:
            with open(self.model_pickle, "rb") as f:
                model = pickle.load(f)
            conv = ll = bic = aic = None
        else:
            model, conv, ll, bic, aic = self.train_hmm_model()
            if self.save_model:
                with open("hmm_model.pkl", "wb") as f:
                    pickle.dump(model, f)

        # predict hidden states
        raw_states = model.predict(self.features)
        states = pd.Series(raw_states)
        # smooth out very short regimes
        states = self._smooth_states(states)
        self.data['State'] = states

        # map states to regimes by sorted mean return
        means = [model.means_[i, 0] for i in range(self.regime_number)]
        order = np.argsort(means)
        labels = {}
        if self.regime_number == 2:
            labels[order[0]] = "Bear"
            labels[order[1]] = "Bull"
        else:
            labels[order[0]] = "Bear"
            labels[order[1]] = "Neutral"
            labels[order[2]] = "Bull"
        self.data['Regime'] = self.data['State'].map(labels)
        self.labels = labels

        if self.getMetrics and not self.model_pickle:
            return self.data, conv, ll, bic, aic
        return self.data

    def visualize_regime_detection(self, ticker: str = None):
        df = self.data.copy()
        if ticker is not None and 'Ticker' in df:
            df = df[df['Ticker'] == ticker]

        # build subplots titles from self.labels
        subplot_titles = [f"{lab} Regime" for lab in self.labels.values()]
        fig = make_subplots(rows=len(self.labels), cols=1, subplot_titles=subplot_titles)

        # iterate over the stored mapping
        for i, (state_idx, lab) in enumerate(self.labels.items(), start=1):
            mask = df['Regime'] == lab
            fig.add_trace(
                go.Scatter(
                    x=df.index[mask],
                    y=df['Close'][mask],
                    mode="markers",
                    name=lab
                ),
                row=i, col=1
            )

        fig.update_layout(
            title=f"HMM Market Regimes{' for ' + ticker if ticker else ''}",
            height=300 * len(self.labels),
            template="plotly_white"
        )
        return fig

    def rolling_regime_prediction(self, train_window: int = 252):
        """
        Feed-forward using a rolling window:
        - Train on the first `train_window` days
        - Predict the next day
        - Expand window by 1 day and repeat
        Returns a Series of predicted states.
        """
        n = len(self.features)
        pred_states = []
        for end in range(train_window, n):
            model, *_ = MarketRegimes(
                data=self.data.iloc[:end],
                regime_number=self.regime_number,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                n_starts=self.n_starts
            ).train_hmm_model()
            s = model.predict(self.features[end : end+1])
            pred_states.append(s[0])
        return pd.Series([np.nan]*train_window + pred_states, index=self.data.index)
