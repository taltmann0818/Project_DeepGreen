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
from scipy import linalg
from scipy.stats import linregress
from typing import Dict, List, Optional
import logging
from Components.TickerData import TickerData
from Components.DataModules.sector_analysis import SectorAnalysis
from polygon import RESTClient


class AlphaVectorPipeline:
    """Comprehensive alpha generation pipeline with data enrichment capabilities."""

    def __init__(
            self,
            volume_threshold: float = 1_000_000.0,
            factor_cols: List[str] | None = None,
            sigma_col: str = "sigma_daily",
            return_col: str = "pred_return",
            polygon_api_key: str = 'XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2',
            min_observations: int = 30,
    ) -> None:
        """Parameters
        ----------
        volume_threshold
            \$ADV level at which no liquidity haircut is applied.
        factor_cols
            Names of exposure columns to neutralise.
        sigma_col
            Column containing ex‑ante daily volatility.
        return_col
            Column containing predicted returns.
        polygon_api_key
            API key for Polygon.io market cap data.
        min_observations
            Minimum observations required for beta calculation.
        """
        self.volume_threshold = volume_threshold
        self.factor_cols = factor_cols or []
        self.sigma_col = sigma_col
        self.return_col = return_col
        self.polygon_api_key = polygon_api_key
        self.min_observations = min_observations

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

            if len(common_dates) < self.min_observations:
                betas[ticker] = np.nan
                continue

            # Get aligned returns (no NaNs)
            ticker_rets = ticker_data.loc[common_dates].dropna()
            market_rets = market_returns.loc[ticker_rets.index]

            # Final check after dropping NaNs
            if len(ticker_rets) < self.min_observations:
                betas[ticker] = np.nan
                continue

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

    def _add_constant(self, X: np.ndarray) -> np.ndarray:
        """Add constant column to design matrix (replaces statsmodels.api.add_constant)"""
        ones = np.ones((X.shape[0], 1))
        return np.column_stack([ones, X])

    def _ols_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform OLS regression and return residuals.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_samples, n_features)
        y : np.ndarray
            Target vector (n_samples,)

        Returns
        -------
        np.ndarray
            Residuals from the regression
        """
        try:
            # Use scipy's lstsq for more robust computation
            coeffs, residuals, rank, s = linalg.lstsq(X, y)
            fitted_values = X @ coeffs
            return y - fitted_values
        except Exception as e:
            # If regression fails, return original y values
            print(f"Regression failed: {e}")
            return y

    def _process_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Process a single date group for factor neutralization.

        This helper method encapsulates the logic for processing each date group,
        making the code more modular and easier to maintain.

        Parameters
        ----------
        group : pd.DataFrame
            DataFrame containing data for a single date

        Returns
        -------
        pd.DataFrame
            Processed group with alpha_pure column added
        """
        if len(group) < 2:
            group["alpha_pure"] = group["ir_norm"]
            return group

        # Handle NaN values
        group = group.copy()
        group["ir_norm"] = group["ir_norm"].fillna(0.0)
        for col in self.factor_cols:
            group[col] = group[col].fillna(0.0)

        y = group["ir_norm"].astype(float).values

        # Check if we have any factor columns
        if not self.factor_cols:
            # If no factors, alpha_pure = ir_norm (no neutralization)
            group["alpha_pure"] = y
            return group

        X = group[self.factor_cols].astype(float).values
        X = self._add_constant(X)

        try:
            residuals = self._ols_regression(X, y)
            group["alpha_pure"] = residuals
            return group
        except Exception as e:
            # Log error and return original group with ir_norm as alpha_pure
            print(f"Error in regression: {e}")
            group["alpha_pure"] = group["ir_norm"]
            return group

    def alpha_signals(self, pred_df: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
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
        required_pred_cols = ["date", "Ticker", self.return_col, "q_low", "q_high", self.sigma_col]
        missing_cols = [col for col in required_pred_cols if col not in pred_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in pred_df: {missing_cols}")

        # Ensure date columns are datetime
        df = pred_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # ---------------- 1. Raw expected return -------------------------
        df["mu"] = df[self.return_col]

        # ---------------- 2. Tradability haircut ------------------------
        #df["haircut"] = np.minimum(1.0, df[self.adv_col] / self.volume_threshold)
        #df["mu_adj"] = df["mu"] * df["haircut"]

        # ---------------- 3. Risk‑scale to IR units ----------------------
        # Avoid division by zero
        df["ir"] = df["mu"] / df[self.sigma_col].replace(0.0, np.nan)

        # ---------------- 4. X‑section normalisation --------------------
        df["ir_norm"] = df.groupby("date")["ir"].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )

        # ---------------- 5. Factor / sector neutralisation -------------
        pure_alpha_frames = []
        for date, grp in df.groupby("date", sort=False):
            # Skip empty groups or groups with insufficient data
            if len(grp) < 2:
                continue

            # Handle NaN values
            grp = grp.copy()
            grp["ir_norm"] = grp["ir_norm"].fillna(0.0)
            for col in self.factor_cols:
                grp[col] = grp[col].fillna(0.0)

            y = grp["ir_norm"].astype(float).values

            # Check if we have any factor columns
            if not self.factor_cols:
                # If no factors, alpha_pure = ir_norm (no neutralization)
                tmp = grp.copy()
                tmp["alpha_pure"] = y
                pure_alpha_frames.append(tmp)
                print(f"No beta factor(s) neutralization for alpha vector")
                continue

            X = grp[self.factor_cols].astype(float).values
            X = self._add_constant(X)

            try:
                residuals = self._ols_regression(X, y)
                tmp = grp.copy()
                tmp["alpha_pure"] = residuals
                pure_alpha_frames.append(tmp)
            except Exception as e:
                # Log error and skip this group
                print(f"Error processing group for date {date}: {e}")
                continue

        if not pure_alpha_frames:
            raise ValueError("No data groups were successfully processed")

        df = pd.concat(pure_alpha_frames, ignore_index=True)

        # ---------------- 6. Shrinkage / confidence weighting ------------
        width = df["q_high"] - df["q_low"]
        width = width.clip(lower=1e-8)  # Ensure positive width with small minimum to avoid division issues
        confidence = 1.0 / width
        confidence = confidence.replace([np.inf, -np.inf], np.nan)
        confidence = confidence.fillna(confidence.median())
        confidence = confidence.groupby(df["date"]).transform(
            lambda x: x / x.max() if x.max() > 0 else x
        )
        df["alpha_shrunk"] = df["alpha_pure"] * confidence

        # ---------------- 7. Pack for the optimiser ----------------------
        if "date" in df.columns and not pd.api.types.is_datetime64_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # More memory-efficient packing - process one group at a time
        packed: Dict[pd.Timestamp, Dict[str, float]] = {}
        for date, group in df.groupby("date", sort=False):
            date_key = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
            ticker_values = dict(zip(group["Ticker"], group["alpha_shrunk"]))
            packed[date_key] = ticker_values

        return packed

    def run(self, predictions: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
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
            self.factor_cols = sector_columns + ['smb_exposure', 'MktBeta']

        # 7. Generate alpha signals using the existing run method
        alpha_dict = self.alpha_signals(predictions)
        logging.info(f"Generated alpha signals for {len(alpha_dict)} dates")

        return alpha_dict


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
