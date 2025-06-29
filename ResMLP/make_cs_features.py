import pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from datetime import date, timedelta

ROOT = Path("features_cs")           # partition root
START = date(2022, 1, 3)             # first trading day to (re)build
END   = date.today() - timedelta(days=1)

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