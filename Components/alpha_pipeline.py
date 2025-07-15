# alpha_pipeline.py
"""
Alpha Vector Pipeline
=====================
A comprehensive alpha generation pipeline that takes raw TFT predictions and transforms them
into pure, risk-controlled alpha signals ready for portfolio optimization.

Features:
- Processes raw prediction files with market data enrichment
- Computes market beta, sector exposures, and size factors
- Applies factor neutralization and risk scaling
- Outputs alpha signals ready for portfolio manager

The pipeline performs:
1. Raw prediction data loading and validation
2. Market data enrichment (benchmark prices, beta calculation)
3. Sector classification and dummy variable creation
4. Market cap and SMB exposure calculation
5. Alpha vector generation with factor neutralization
6. Risk scaling and confidence weighting
7. Output formatting for portfolio optimization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Tuple
import logging
from Components.TickerData import TickerData
from Components.DataModules.sector_analysis import SectorAnalysis
from polygon import RESTClient

VOL_EPS = 1e-4       # protects against divide‑by‑zero in σ‑scaling
OUTLIER_CLIP = 10.0  # final z‑scores clipped to ±10 σ

class AlphaVectorPipeline:
    """Comprehensive alpha generation pipeline with data enrichment capabilities."""

    def __init__(
            self,
            volume_threshold: float = 1_000_000.0,
            factor_cols: List[str] | None = None,
            sigma_col: str = "sigma_daily",
            polygon_api_key: str = 'XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2',
    ):
        """Parameters
        ----------
        volume_threshold
            level at which no liquidity haircut is applied.
        factor_cols
            Names of exposure columns to neutralise.
        sigma_col
            Column containing ex‑ante daily volatility.
        return_col
            Column containing predicted returns.
        polygon_api_key
            API key for Polygon.io market cap data.
        """
        self.volume_threshold = volume_threshold
        self.factor_cols = factor_cols or []
        self.sigma_col = sigma_col
        self.polygon_api_key = polygon_api_key

        # Initialize data retriever for benchmark data
        self.data_retriever = None

        # Initialize Polygon client for market cap data
        if polygon_api_key:
            self.polygon_client = RESTClient(polygon_api_key, num_pools=50)
        else:
            self.polygon_client = None

        logging.basicConfig(level=logging.INFO)

    def compute_market_beta_robust(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute market beta for each ticker, handling missing data gracefully.

        Parameters
        ----------
        df : DataFrame with columns ['date', 'Ticker', 'asset_ret', 'mkt_ret']

        Returns
        -------
        pd.Series indexed by ticker with beta values
        """
        # Ensure data is sorted
        df = df.sort_values(['date', 'Ticker']).copy()

        # Get unique tickers 
        tickers = df['Ticker'].unique()
        betas = {}

        # Get market returns as a series indexed by date
        market_returns = df[['date', 'mkt_ret']].drop_duplicates('date').set_index('date')['mkt_ret']

        for ticker in tickers:
            # Get this ticker's returns
            ticker_data = df[df['Ticker'] == ticker].set_index('date')['asset_ret']

            # Find overlapping dates (automatically handles NaNs)
            common_dates = ticker_data.index.intersection(market_returns.index)

            # Get aligned returns (no NaNs)
            ticker_rets = ticker_data.loc[common_dates].dropna()
            market_rets = market_returns.loc[ticker_rets.index]

            # Compute beta using covariance method
            try:
                covariance = np.cov(ticker_rets, market_rets)[0, 1]
                market_var = np.var(market_rets, ddof=1)
                betas[ticker] = covariance / market_var
            except:
                betas[ticker] = np.nan

        return pd.Series(betas, name='beta')

    def get_market_cap_for_ticker(self, ticker: str) -> Optional[float]:
        """Get market cap for a single ticker using Polygon API"""
        if not self.polygon_client:
            return None

        try:
            details = self.polygon_client.get_ticker_details(ticker)
            return getattr(details, 'market_cap', None)
        except Exception as e:
            #logging.warning(f"Error fetching market cap for {ticker}: {e}")
            return None

    def zscore(self, series: pd.Series) -> pd.Series:
        """Compute z-score normalization"""
        return (series - series.mean()) / series.std(ddof=0)

    def alpha_signals(self, pred_df: pd.DataFrame) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], pd.DataFrame]:
        """Execute the full pipeline.

        Parameters
        ----------
        pred_df
            TFT forecast DataFrame (see *Assumptions* above).
        exposure_df
            Factor / sector exposure DataFrame.

        Returns
        -------
        dict
            ``{date: {ticker: alpha_value}}`` nested mapping suitable for the
            optimiser.
        """
        # Validate required columns exist in pred_df
        required_pred_cols = ["date", "Ticker", "pred_return", "q_low", "q_high", self.sigma_col]
        missing_cols = [col for col in required_pred_cols if col not in pred_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in pred_df: {missing_cols}")

        # Ensure date columns are datetime
        df = pred_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # ---------------- 1. Raw expected return -------------------------
        df["mu"] = df["pred_return"].astype(float)

        # ---------------- 2. Tradability haircut ------------------------
        #df["haircut"] = np.minimum(1.0, df[self.adv_col] / self.volume_threshold)
        #df["mu_adj"] = df["mu"] * df["haircut"]

        # ---------------- 3. Risk‑scale to IR units ----------------------
        # Avoid division by zero
        sigma = df[self.sigma_col].replace(0.0, VOL_EPS).astype(float)
        df["ir"] = df["mu"] / sigma

        # ---------------- 4. X‑section normalisation --------------------
        df["ir_norm"] = df.groupby("date")["ir"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
        df["ir_norm"] = df["ir_norm"].clip(-OUTLIER_CLIP, OUTLIER_CLIP)

        # ---------------- 5. Factor / sector neutralisation -------------
        def _neutralise(group: pd.DataFrame) -> pd.Series:
            y = group["ir_norm"].astype(float)
            X = group[self.factor_cols].astype(float)
            X = sm.add_constant(X, has_constant="add")
            beta = np.linalg.pinv(X.values) @ y.values
            resid = y - X @ beta
            return resid

        df["alpha_pure"] = df.replace(np.nan, 0.0).groupby("date", group_keys=False).apply(_neutralise)

        # ---------------- 6. Shrinkage / confidence weighting ------------
        width = (df["q_high"] - df["q_low"]).replace(0.0, np.nan)
        confidence = (1.0 / width).groupby(df["date"]).transform(lambda x: x / x.max())
        df["alpha_shrunk"] = df["alpha_pure"] * confidence

        # 7) Final mean‑zero, unit‑σ standardisation ---------------------------
        df["alpha_final"] = df.groupby("date")["alpha_shrunk"].transform(
            lambda x: ((x - x.mean()) / x.std(ddof=0)).clip(-OUTLIER_CLIP, OUTLIER_CLIP)
        )       

        # ---------------- 7. Pack for the optimiser ----------------------
        if "date" in df.columns and not pd.api.types.is_datetime64_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
            df = df["date"].sort_values(ascending=True)

        # Pack for optimiser ----------------------------------------------------
        packed: Dict[pd.Timestamp, Dict[str, float]] = {
            d: g.set_index("Ticker")["alpha_shrunk"].to_dict()
            for d, g in df.groupby("date", sort=False)
        }

        return packed, df

    def run(self, predictions: pd.DataFrame) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], pd.DataFrame]:
        """
        Process raw prediction file and generate alpha signals with full data enrichment.

        This method replicates the complete workflow from the notebook:
        1. Load raw predictions
        2. Add benchmark data and compute market beta
        3. Add sector information
        4. Add market cap and SMB exposure
        5. Generate alpha signals

        Parameters
        ----------
        predictions_file : str
            Path to the raw predictions parquet file

        Returns
        -------
        Dict[pd.Timestamp, Dict[str, float]]
            Alpha signals ready for portfolio optimization
            :param predictions:
        """
        logging.info("Starting raw prediction processing...")

        # 1. Load raw predictions
        logging.info(f"Loaded {len(predictions)} prediction records")

        if 'date' in predictions.index.names or predictions.index.name == 'date':
            predictions = predictions.reset_index()

        # 2. Initialize data retriever and get benchmark data
        if not self.data_retriever:
            self.data_retriever = TickerData(indicator_list=None, days=252, prediction_mode=True)

        benchmark_prices = self.data_retriever.get_ohlc_for_ticker('I:NDX').reset_index().rename(columns={"Close": "spy_close"})

        # Ensure timezone consistency
        benchmark_prices['date'] = benchmark_prices['date'].dt.tz_localize(None)
        predictions['date'] = predictions['date'].dt.tz_localize(None)

        # Merge with benchmark data
        predictions = predictions.merge(
            benchmark_prices[['date', 'spy_close']],
            on=['date'],
            how='inner'
        )
        logging.info("Added benchmark data")

        # 3. Compute market beta
        df = predictions.copy()
        df['asset_ret'] = df.groupby('Ticker')['Close'].pct_change()
        df['mkt_ret'] = df['spy_close'].pct_change()

        # Drop rows with missing returns
        df = df.dropna(subset=['asset_ret', 'mkt_ret'])

        # Compute betas
        betas = self.compute_market_beta_robust(df)
        predictions['MktBeta'] = predictions['Ticker'].map(betas)
        logging.info("Computed market betas")

        # 4. Add sector information
        unique_tickers = predictions['Ticker'].unique()
        ticker_sectors = SectorAnalysis.create_sector_indicator(
            unique_tickers,
            self.data_retriever.data_fetcher,
            max_workers=10
        )
        predictions['sector'] = predictions['Ticker'].map(ticker_sectors)

        # Create sector dummy variables
        sector_dummies = pd.get_dummies(predictions['sector'], prefix='sector', dtype=float)
        predictions = pd.concat([predictions, sector_dummies], axis=1)
        sector_columns = [col for col in predictions.columns if col.startswith('sector_')]
        logging.info(f"Added sector information with {len(sector_columns)} sectors")

        # 5. Add market cap and SMB exposure
        if self.polygon_client:
            market_caps = {}
            for ticker in unique_tickers:
                market_cap = self.get_market_cap_for_ticker(ticker)
                market_caps[ticker] = market_cap

            predictions['mkt_cap'] = predictions['Ticker'].map(market_caps)
            predictions['log_mktcap'] = np.log(predictions['mkt_cap'])
            predictions['smb_exposure'] = predictions.groupby('date')['log_mktcap'].transform(self.zscore) * -1
            logging.info("Added market cap and SMB exposure")
        else:
            logging.warning("No Polygon API key provided, skipping market cap data")
            predictions['smb_exposure'] = 0.0

        # 6. Update factor columns to include all enrichment factors
        if not self.factor_cols:
            self.factor_cols = sector_columns + ['MktBeta'] + ['smb_exposure']

        # 7. Generate alpha signals using the existing run method
        return self.alpha_signals(predictions)


# -------------------------------------------------------------------------
# Example usage (remove if importing as a library)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Skeleton demo only – fill with real data
    demo_pred = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01"] * 3),
            "Ticker": ["AAPL", "MSFT", "META"],
            "pred_return": [0.012, 0.010, 0.008],
            "q_low": [0.005, 0.004, 0.003],
            "q_high": [0.020, 0.018, 0.015],
            "sigma_daily": [0.018, 0.017, 0.022],
            "adv20": [2_500_000, 3_200_000, 1_100_000],
        }
    )

    demo_expo = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01"] * 3),
            "ticker": ["AAPL", "MSFT", "META"],
            "MktBeta": [1.05, 0.98, 1.10],
            "SMB": [-0.2, -0.1, 0.3],
            "TechSector": [1, 1, 1],
        }
    )

    pipe = AlphaVectorPipeline()
    alpha_dict = pipe.alpha_signals(demo_pred, demo_expo)
    # Pretty‑print result
    import pprint

    pprint.pp(alpha_dict, sort_dicts=False)