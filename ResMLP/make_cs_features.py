import pandas as pd
import numpy as np
from typing import Tuple, Optional
import torch
import hashlib
from pathlib import Path
import json
import os
from Components.TickerData import TickerData
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Components.TickerData import TickerData

class DataModule:
    """
    Data module for cross-sectional ResNLP model training that handles data loading, preprocessing, and creation of PyTorch Lightning dataloaders.
    """

    def __init__(
        self,
        config: dict = None,
        batch_size: int = None,
        max_prediction_length: int = None,
        max_encoder_length: int = None,
        days: int = 252,
        prediction_window: int = 3,
        num_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: str = "data_cache",
        sample_size: int = 100,
    ):
        self.config = config

        # Use config values with fallback to parameters
        self.batch_size = batch_size or config.get("BATCH_SIZE", 256)
        self.sample_size = sample_size
        self.max_prediction_length = max_prediction_length or config.get("DECODER_LEN", 3)
        self.max_encoder_length = max_encoder_length or config.get("ENCODER_LEN", 30)
        self.days = days
        self.prediction_window = prediction_window
        self.num_workers = num_workers or max(1, os.cpu_count() // 2)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Get feature columns from config
        self.static_categoricals = config.get("STATIC_CATS", [])
        self.time_varying_known_reals = config.get("TV_KNOWN_REAL", [])
        self.time_varying_unknown_reals = config.get("TV_UNKNOWN_REAL", [])

        # Model path from config
        self.model_path = config.get("ONNX_MODEL_PATH", "")

        # Additional columns needed for processing (target and identifier)
        self.indicator_list = ['Close'] + self.static_categoricals + self.time_varying_known_reals + self.time_varying_unknown_reals



# ---------- 1.  helpers --------------------------------------------------
def pct_rank(s: pd.Series) -> pd.Series:
    """[-0.5, 0.5] cross-section rank (lower = −0.5, higher = +0.5)."""
    return s.rank(pct=True, method="average").sub(0.5)

def write_partition(df: pd.DataFrame, d: date):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        tbl,
        root_path=ROOT,
        partition_cols=["year", "month", "day"],
        basename_template="part-{}-{}.parquet".format(d, np.random.randint(1e9))
    )

# ---------- 2.  main loop ------------------------------------------------
for d in pd.bdate_range(START, END):
    print("Building", d.date())
    # 2.1 gather raw slices ------------------------------------------------
    px = load_prices(d)              # -> DataFrame[ticker, close, volume]
    fnd = load_fundamentals(d)       # -> DataFrame with point-in-time fields
    iv  = load_iv_surface(d)         # -> DataFrame[ticker, iv_atm_1d, iv_25d]
    alt = load_alt_data(d)           # -> DataFrame any alt-feeds

    # 2.2 merge on ticker --------------------------------------------------
    df = px.merge(fnd, on="ticker", how="left")\
           .merge(iv , on="ticker", how="left")\
           .merge(alt, on="ticker", how="left")
    df["date"] = d.date()

    # 2.3 engineer price & momentum features ------------------------------
    px_hist = load_price_window(d, window=63)      # 3-month window for ranks
    rets    = px_hist.pivot(index="date", columns="ticker", values="close")\
                     .pct_change().dropna()
    mom_1m  = rets.rolling(21).sum().loc[d]        # 1-month momentum
    df = df.set_index("ticker")
    df["ret_1m"] = mom_1m
    # … add 1-w, 3-m, volatility, RSI, etc.

    #Peer residuals	Return_i − sector_mean_return at t-1

    # 2.4 cross-sectional ranks (do *after* filling NaNs) ------------------
    rank_cols = ["ret_1m", "pe_ttm", "adv20", "iv_atm_1d"]
    for c in rank_cols:
        df[f"{c}_rank"] = pct_rank(df[c].fillna(df[c].median()))

    # 2.5 interaction terms ------------------------------------------------
    df["mom_iv_inter"] = df["ret_1m_rank"] * df["iv_atm_1d_rank"]

    # 2.6 future 3-day return label ---------------------------------------
    fwd_px = load_close(d + pd.tseries.offsets.BDay(3))
    df["fwd_3d_return"] = (
        fwd_px.set_index("ticker")["close"] / df["close"] - 1.0
    )

    # 2.7 tidy, reset, write ----------------------------------------------
    df = df.reset_index()
    df["year"], df["month"], df["day"] = d.year, d.month, d.day
    write_partition(df.astype("float32", errors="ignore"), d.date())