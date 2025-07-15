# datamodule_cs.py
from __future__ import annotations

import pandas as pd
import numpy as np
from polygon import RESTClient

from Components.polygon_client_patch import patch_polygon_client
patch_polygon_client(max_pool_size=50)
from pandas.tseries.offsets import BDay
from typing import Callable, Dict, Iterable, List, Mapping, Sequence
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Components.DataModules.data_fetcher import DataFetcher
from Components.DataModules.sector_analysis import SectorAnalysis
from Components.DataModules.fundementals import FundementalData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_rank(series: pd.Series) -> pd.Series:
    """[-0.5, 0.5] cross-section rank (lower = −0.5, higher = +0.5)."""
    return series.rank(method="average", pct=True).sub(0.5)

class CrossSectional():
    """Static helper for building cross‑sectional (date, ticker) feature frames."""

    # Columns on which we compute daily cross‑sectional ranks
    rank_cols: List[str] = [
        "ret_1m",   # 21‑day momentum
        #"pe_ratio",
        #"ps_ratio",
        #"pb_ratio"
    ]

    # ---------------------------------------------------------------------
    # vectorised frame builder
    # ---------------------------------------------------------------------

    @classmethod
    def build_frame_range(
        cls,
        days: int = 1260,
        sample_size: int = 4000,
        prediction_window: int = 3,
        **kwargs,
    ) -> pd.DataFrame:
        """Build full feature/label table for *start_date → end_date* (inclusive).

        Parameters
        ----------
        start_date, end_date
            Boundaries of the sample (YYYY‑MM‑DD strings).
        loaders
            Mapping that should return **long** DataFrames with at least
            ``["date", "ticker"]``.  Expected keys (optional except *prices*):

            - ``"prices"`` – OHLCV & *adv20*.
            - ``"fundamentals"`` – point‑in‑time fields (e.g. *pe_ttm*).
            - ``"iv_surface"`` – implied‑vol surface metrics.
            - ``"alt"`` – any alternative feed(s).

        days
            Look‑back window used for rolling stats (default ≈ 1 yr).
        loader_kwargs
            Forwarded verbatim to each loader.
            :param days:
            :param sample_size:
            :param prediction_window:
        """
        # 1) ───────────────────────── fetch all slices in one go
        api_key = 'XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2'
        start_date = kwargs.get('start_date',None)
        end_date = kwargs.get('end_date', None)
        client = RESTClient(api_key)
        data_fetcher = DataFetcher(
            client=client,
            start_date=start_date,
            end_date=end_date,
            days=days,
            sample_size=sample_size,
        )
        px = data_fetcher.fetch_stock_data()
        print("Finished adding OHLCV")
        data_fetcher.client.client.clear()
        px = SectorAnalysis.add_sector_indicators(px, data_fetcher, ['sic_sector'])
        print("Finished adding sector indicators")
        #fnd = FundementalData(tickers=px['Ticker'].unique(), days=days, workers=20).fetch()
        #print("Finished adding fundementals")

        #fnd.to_csv('fund_temp.csv')

        # 2) ───────────────────────── merge on (date, ticker)
        px["date"] = pd.to_datetime(px["date"], errors="coerce", utc=False)
        #fnd["date"] = pd.to_datetime(fnd["date"], errors="coerce", utc=False)
        #df = (
        #    px.merge(fnd, on=["date", "Ticker"], how="left", suffixes=("", "_fnd"))
        #)

        # 3) ───────────────────────── price‑derived features (vectorised)
        df = px.sort_values(["Ticker", "date"])

        # daily returns
        df["ret_1d"] = df.groupby("Ticker")["Close"].pct_change()

        # 21‑day momentum (∑ daily returns)
        df["ret_1m"] = (
            df.groupby("Ticker")["ret_1d"]
              .rolling(window=21, min_periods=21)
              .sum()
              .reset_index(level=0, drop=True)
        )

        # 21‑day realised volatility (σ)
        df["vol_21d"] = (
            df.groupby("Ticker")["ret_1d"]
              .rolling(window=21, min_periods=21)
              .std()
              .reset_index(level=0, drop=True)
        )

        # peer residuals: return minus sector mean (requires *sector* column)
        if "sic_sector" in df.columns:
            sector_mean = df.groupby(["date", "sic_sector"])["ret_1d"].transform("mean")
            df["peer_resid"] = df["ret_1d"] - sector_mean

        # 4) ───────────────────────── cross‑section percentile ranks
        for col in cls.rank_cols:
            if col in df.columns:
                df[f"{col}_rank"] = (
                    df.groupby("date")[col]
                      .transform(lambda s: pct_rank(s.fillna(s.median())))
                )

        # 6) ───────────────────────── forward‑looking label (3‑BD return)
        df["fwd_3d_close"] = df.groupby("Ticker")["Close"].shift(-abs(prediction_window))
        df["fwd_3d_return"] = df["fwd_3d_close"] / df["Close"] - 1.0

        # 7) ───────────────────────── final clean‑up
        drop_cols = ["fwd_3d_close"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Drop rows with NaN values in critical columns including target
        critical_cols = ["ret_1m", "fwd_3d_return"] + [f"{c}_rank" for c in cls.rank_cols]
        df = df.dropna(subset=critical_cols)

        # Remove infinite values in target
        df = df[~np.isinf(df["fwd_3d_return"])]

        return df.reset_index(drop=True)

class CrossSectionDS(Dataset):
    """
    Wraps a Pandas frame that is already in (date, ticker) long format.

    Each __getitem__ returns (X, y) where:
      X : FloatTensor [n_features]
      y : FloatTensor []   (scalar target)
    """
    def __init__(self, frame: pd.DataFrame,
                 feature_cols: Sequence[str],
                 target_col: str):
        # Get feature values and normalize them
        feature_values = frame[feature_cols].values

        # Replace any remaining NaN or inf values with 0
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features to prevent gradient explosion
        # Use robust scaling (subtract median, divide by IQR)
        for i in range(feature_values.shape[1]):
            col_values = feature_values[:, i]
            if len(col_values) > 0:
                q25, q50, q75 = np.percentile(col_values, [25, 50, 75])
                iqr = q75 - q25
                if iqr > 1e-8:  # Avoid division by zero
                    feature_values[:, i] = (col_values - q50) / iqr
                else:
                    feature_values[:, i] = col_values - q50

        # Get target values and scale them appropriately
        target_values = frame[target_col].values
        target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale targets to have similar magnitude as features for better training stability
        # Use standard scaling for targets (mean=0, std=1)
        target_mean = np.mean(target_values)
        target_std = np.std(target_values)
        if target_std > 1e-8:
            target_values = (target_values - target_mean) / target_std
        else:
            target_values = target_values - target_mean

        # Store scaling parameters for potential inverse transform
        self.target_mean = target_mean
        self.target_std = target_std

        self.X = torch.tensor(feature_values, dtype=torch.float32)
        self.y = torch.tensor(target_values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------------
# LightningDataModule wrapper (thin convenience layer)
# ---------------------------------------------------------------------------

class DataModule(pl.LightningDataModule):
    """Simple PyTorch Lightning *DataModule* that delegates feature builds
    to :class:`CrossSectional` and emits ready‑to‑use *pandas* frames.
    """
    def __init__(
        self,
        start_date: str | None = None,
        end_date: str  | None = None,
        days: int = 1260,
        sample_size: int = 4000,
        feature_cols: Sequence[str] | None = None,   # autodetect if None
        target_col: str = "fwd_3d_return",
        val_ratio: float = 0.2,                      # last 20 % of days = validation
        prediction_window: int = 3,
        batch_size: int = 512,
        num_workers: int = 4,
        **build_kwargs,                              # forwarded to build_frame_range
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["feature_cols"])  # <- logs everything else
        self.feature_cols = list(feature_cols) if feature_cols is not None else None

        self._full_df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None

    # ------------------------------------------------------------
    def prepare_data(self):  # type: ignore[override]
        # Heavy lifting happens here (CPU‑bound); run only once in DDP.
        self._full_df = CrossSectional.build_frame_range(
            days=self.hparams.days,
            sample_size=self.hparams.sample_size,
            start_date=self.hparams.start_date,
            end_date=self.hparams.end_date,
            prediction_window=self.hparams.prediction_window
        )

    # ---------------------------------------------------------------------
    # Light splitting / Dataset creation
    # ---------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        if self._full_df is None:
            raise RuntimeError("prepare_data() must run before setup().")

        df = self._full_df.sort_values("date").reset_index(drop=True)

        # infer features if caller did not pass an explicit list
        if self.feature_cols is None:
            drop = {"date", "Ticker", self.hparams.target_col}
            self.feature_cols = [c for c in df.columns if c not in drop]

        # --------------- time-based split ---------------------------------
        unique_days = np.sort(df["date"].unique())
        cut_idx = int((1 - self.hparams.val_ratio) * len(unique_days))
        cut_day = unique_days[cut_idx]

        self.train_df = df[df["date"] <= cut_day].reset_index(drop=True)
        self.val_df   = df[df["date"] >  cut_day].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Dataloaders
    # ---------------------------------------------------------------------
    def _make_loader(self, frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = CrossSectionDS(frame, self.feature_cols, self.hparams.target_col)
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            #pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_loader(self.train_df, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_df, shuffle=False)
