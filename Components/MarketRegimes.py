import warnings
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
from typing import List

warnings.filterwarnings("ignore")


class RegimeDetector:
    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = None,
    ):
        """
        A pooled‐ticker HMM regime detector with feed-forward retraining.

        Parameters
        ----------
        n_components : int
            Number of hidden states (e.g. 2 for bull/bear).
        covariance_type : str
            HMM covariance type.
        n_iter : int
            Maximum EM iterations.
        random_state : int
            RNG seed.
        """
        self.n_components  = n_components
        self.hmm_kwargs    = dict(
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.model         = None
        self.tickers       = None
        self.lengths       = None
        self.idx_maps      = None
        self.states_all    = None
        self.hidden_states = None
        self.df            = None
        self.ma            = None

    @staticmethod
    def _prepare(df: pd.DataFrame, ma: int):
        """
        Compute MA‐smoothed log‐returns per ticker and stack them.

        Returns X_all, tickers, lengths, idx_maps.
        """
        features, lengths, tickers, idx_maps = [], [], [], {}
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        for tk, grp in df.groupby("Ticker"):
            grp = grp.sort_values("Date").set_index("Date")
            grp["close_ma"] = grp["Close"].rolling(ma).mean()
            grp["lr"]       = np.log(grp["close_ma"] / grp["close_ma"].shift(1))
            grp = grp.dropna(subset=["lr"])

            X = grp[["lr"]].values
            features.append(X)
            lengths.append(len(X))
            tickers.append(tk)
            idx_maps[tk] = grp.index

        X_all = np.vstack(features)
        return X_all, tickers, lengths, idx_maps

    @staticmethod
    def _truncated_lengths(lengths: List[int], upto: int):
        """
        For a cut‐off index `upto`, compute how many obs per ticker lie in [0, upto).
        """
        out, rem = [], upto
        for L in lengths:
            if rem <= 0:
                break
            out.append(min(L, rem))
            rem -= L
        return out

    @staticmethod
    def _split_states(states_all: np.ndarray, tickers: List[str], lengths: List[int]):
        """
        Un‐stack a flat state array back into per‐ticker sequences.
        """
        res, idx = {}, 0
        for tk, L in zip(tickers, lengths):
            res[tk] = states_all[idx : idx + L]
            idx += L
        return res

    def _new_model(self):
        return GaussianHMM(n_components=self.n_components, **self.hmm_kwargs)

    def train_feedforward(
        self,
        df: pd.DataFrame,
        ma: int,
        initial_train_size: int,
        retrain_step: int,
    ):
        """
        Walk‐forward training on the pooled data.

        After calling this, `self.model` is the final HMM,
        and `self.hidden_states` is a dict[ticker → state array].
        """
        # store for plotting later
        self.df = df.copy()
        self.ma = ma

        X_all, tickers, lengths, idx_maps = self._prepare(df, ma)
        N = X_all.shape[0]
        states_all = np.zeros(N, dtype=int)

        # initial fit
        init_lens       = self._truncated_lengths(lengths, initial_train_size)
        self.model      = self._new_model().fit(
            X_all[:initial_train_size], lengths=init_lens
        )

        # walk‐forward
        for t in range(initial_train_size, N):
            curr_lens      = self._truncated_lengths(lengths, t + 1)
            st             = self.model.predict(
                X_all[: t + 1], lengths=curr_lens
            )[-1]
            states_all[t]  = st

            # retrain periodically
            if (t - initial_train_size + 1) % retrain_step == 0:
                self.model = self._new_model().fit(
                    X_all[: t + 1], lengths=curr_lens
                )

        # stash everything
        self.states_all    = states_all
        self.tickers       = tickers
        self.lengths       = lengths
        self.idx_maps      = idx_maps
        self.hidden_states = self._split_states(states_all, tickers, lengths)

        return self.hidden_states

    def predict(self, df: pd.DataFrame, ma: int):
        """
        Out‐of‐sample prediction using a previously fitted model.
        """
        # re‐store for plotting
        
        df = df.reset_index()
        self.df = df.copy()
        df['Date'] = pd.to_datetime(self.df['date'])
        self.ma = ma

        if 'Ticker' in df.columns:
            df['Ticker'] = self.df['Ticker'].astype(str)

        X_all, tickers, lengths, idx_maps = self._prepare(df, ma)
        states_all = self.model.predict(X_all, lengths=lengths)

        self.tickers       = tickers
        self.lengths       = lengths
        self.idx_maps      = idx_maps
        self.states_all    = states_all
        self.hidden_states = self._split_states(states_all, tickers, lengths)

        # flatten data for merging back in
        df['State'] = np.nan
        df = df.set_index(['Ticker', 'Date'])
        for ticker, states in self.hidden_states.items():
            dates = self.idx_maps[ticker]  # DatetimeIndex of points we predicted
            df.loc[(ticker, dates), 'State'] = states  # vectorized assignment

        df = df.reset_index()
        combined_states = df['State'].to_numpy()

        return self.hidden_states, combined_states

    def save(self, path: str):
        """Pickle this entire detector (including trained HMM)."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load a previously saved RegimeDetector."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def plot_states(self, ticker: str):
        """
        Scatter‐plot the hidden states for one ticker.
        """
        # reindex df by Date for easy lookups
        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        ts_idx   = self.idx_maps[ticker]
        ts_close = df.loc[ts_idx, "Close"].values
        ts_states = self.hidden_states[ticker]

        fig = go.Figure()
        palette = ["blue", "green", "orange", "red", "purple"]
        for s in np.unique(ts_states):
            mask = ts_states == s
            fig.add_trace(
                go.Scatter(
                    x=ts_idx[mask],
                    y=ts_close[mask],
                    mode="markers",
                    name=f"State {s}",
                    marker=dict(size=3, color=palette[s % len(palette)]),
                )
            )

        fig.update_layout(
            template="ggplot2",
            title=f"Hidden States for {ticker}",
            yaxis_title="Close",
            margin=dict(l=20, r=20, t=50, b=20),
        ).show()
        
"""
Example usage for training:

raw_stock_data["Date"] = pd.to_datetime(raw_stock_data.index)
raw_stock_data = raw_stock_data.rename(columns={"close": "Close"})
det = RegimeDetector(
    n_components=2,
    covariance_type="full",
    n_iter=100,
    random_state=42,
)
det.train_feedforward(
    df=raw_stock_data,
    ma=5,
    initial_train_size=100,
    retrain_step=20,
)
det.save("Models/hmm_temp.pkl")
"""