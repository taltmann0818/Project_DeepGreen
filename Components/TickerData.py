"""
Refactored TickerData class that uses modular components.
This is the main orchestration class that brings together all the separated modules.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from polygon import RESTClient
from Components.polygon_client_patch import patch_polygon_client
patch_polygon_client(max_pool_size=50)

from Components.DataModules.data_fetcher import DataFetcher
from Components.DataModules.technical_indicators import TechnicalIndicators
from Components.DataModules.sector_analysis import SectorAnalysis
from Components.DataModules.calendar_earnings import CalendarEarnings
from Components.DataModules.market_news import MarketNews
from Components.MarketRegimes import RegimeDetector

from pathlib import Path
def _find_model(pickle_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]   # ../..
    for path in project_root.rglob(pickle_name):
        return path
    raise FileNotFoundError(f"{pickle_name} not found anywhere under {project_root}")
    
class TickerData:
    """
    Refactored TickerData class with modular architecture.

    This class orchestrates data fetching, preprocessing, and indicator calculation
    using separate modules for different responsibilities.
    """

    def __init__(self, indicator_list, days=1, prediction_window=3, **kwargs):
        """
        Initialize the TickerData with a ticker symbol and configuration.

        Parameters:
        -----------
        indicator_list : list
            List of technical indicators to calculate
        years : int, default=1
            Number of years of historical data to fetch
        prediction_window : int, default=5
            Window size for prediction
        **kwargs : dict
            Additional keyword arguments:
            - start_date : str, optional
                Start date for data fetching (format: 'YYYY-MM-DD')
            - end_date : str, optional
                End date for data fetching (format: 'YYYY-MM-DD')
            - prediction_mode : bool, default=False
                Whether to run in prediction mode
            - max_workers : int, default=None
                Maximum number of worker threads for parallel processing
        """
        # Configuration
        if indicator_list is not None:
            self.indicator_list = set(indicator_list)
        self.prediction_window = -abs(prediction_window)
        self.days = days

        self.start_date = kwargs.get('start_date',None)
        self.end_date = kwargs.get('end_date', None)
        self.prediction_mode = kwargs.get('prediction_mode', False)
        self.max_workers = kwargs.get('max_workers', None)
        self.sample_size = kwargs.get('sample_size', None)

        # Initialize data fetcher
        api_key = 'XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2'
        self.client = RESTClient(api_key, num_pools=50)
        self.data_fetcher = DataFetcher(
            client=self.client,
            start_date=self.start_date,
            end_date=self.end_date,
            days=days,
            sample_size=self.sample_size,
        )

        # Data storage
        self.dataset_ex_df = None
        self.final_df = None

    def fetch_stock_data(self, workers=20):
        """
        Fetch stock data for the configured tickers.

        Parameters:
        -----------
        workers : int, default=20
            Number of worker threads for parallel processing

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all ticker data
        """
        if self.max_workers:
            workers = max(workers, self.max_workers)
            
        return self.data_fetcher.fetch_stock_data(workers)

    def preprocess_data(self):
        """Preprocess the fetched data"""
        self.dataset_ex_df = self.fetch_stock_data()
        print("Finished fetching OHLCV data")

        if self.dataset_ex_df.empty:
            raise ValueError("No data available for processing")

        # Ensure proper column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume', 'ticker': 'Ticker'
        }
        self.dataset_ex_df = self.dataset_ex_df.rename(columns=column_mapping)

        self.stock_data = self.dataset_ex_df.copy()

        # Add shifted prices for prediction if not in prediction mode
        if not self.prediction_mode:
            grouped = self.dataset_ex_df.groupby('Ticker')
            self.dataset_ex_df['shifted_prices'] = grouped['Close'].shift(self.prediction_window)

        return self.dataset_ex_df

    def add_features(self, df=None):
        """Add all requested features to the dataset"""
        if self.dataset_ex_df is None and df is None:
            raise ValueError("Data must be preprocessed before adding features")

        # Ensure proper datetime index
        if df is None:
            df = self.dataset_ex_df.copy()

        # Check if we already have a datetime index or need to create one
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                # Create a multi-index with Ticker and date to avoid duplicate labels
                df = df.set_index(["Ticker", "date"]).sort_index()
            else:
                # If no date column, assume index is already the date
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)
        else:
            # If we already have a DatetimeIndex, ensure it's timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        # Group by ticker for indicator calculations
        grouped = df.groupby('Ticker')

        # Add basic technical indicators
        df = TechnicalIndicators.add_technical_indicators(df=df,grouped=grouped,indicator_list=self.indicator_list,nasdaq_data=None)
        print("Finished adding technical indicators")

        # Add cross-asset indicators
        #df = self._add_cross_asset_indicators(df)

        # Add sector indicators
        self.data_fetcher.client.client.clear()
        df = SectorAnalysis.add_sector_indicators(df, self.data_fetcher, self.indicator_list)
        print("Finished adding sector indicators")

        # Add news indicators
        self.data_fetcher.client.client.clear()
        df = MarketNews.add_news_indicators(df, self.data_fetcher, self.indicator_list)
        print("Finished adding news indicators")

        # Add calendar and earnings indicators
        df = CalendarEarnings.add_calendar_earnings_indicators(df, self.data_fetcher, self.indicator_list)
        print("Finished adding calendar indicators")

        # Add Hidden Markov Model market regimes
        if 'hmm_state' in self.indicator_list:
            hmm = _find_model("hmm_v2.pkl")
            _, df['hmm_state']  = RegimeDetector.load(hmm).predict(df, ma=5)

        self.dataset_ex_df = df
        return df

    def _add_cross_asset_indicators(self, df):
        """Add cross-asset indicators like SPY, VIX, sector ETFs"""
        cross_asset_indicators = ['spy_ret_1', 'spy_ret_3', 'sector_etf_ret_1', 'vix_delta_1', 'yc_2y10y_delta']
        needed_cross_asset = [ind for ind in cross_asset_indicators if ind in self.indicator_list]

        if needed_cross_asset:
            # SPY returns
            if 'spy_ret_1' in self.indicator_list or 'spy_ret_3' in self.indicator_list:
                spy_data = self.data_fetcher.get_spy_data()
                if spy_data is not None:
                    spy_returns = TechnicalIndicators.calculate_returns(spy_data['close'], [1, 3])
                    if 'spy_ret_1' in self.indicator_list:
                        spy_ret_1 = spy_returns['ret_1'].rename('spy_ret_1')
                        df = df.merge(spy_ret_1.to_frame(), left_index=True, right_index=True, how='left')
                    if 'spy_ret_3' in self.indicator_list:
                        spy_ret_3 = spy_returns['ret_3'].rename('spy_ret_3')
                        df = df.merge(spy_ret_3.to_frame(), left_index=True, right_index=True, how='left')
                else:
                    # Create placeholder columns if SPY data fetch fails
                    if 'spy_ret_1' in self.indicator_list:
                        df['spy_ret_1'] = np.nan
                    if 'spy_ret_3' in self.indicator_list:
                        df['spy_ret_3'] = np.nan

            # Sector ETF returns (using XLK as representative)
            if 'sector_etf_ret_1' in self.indicator_list:
                sector_data = self.data_fetcher.get_sector_etf_data(['XLK'])
                if 'XLK' in sector_data:
                    xlk_ret_1 = sector_data['XLK']['close'].pct_change().rename('sector_etf_ret_1')
                    df = df.merge(xlk_ret_1.to_frame(), left_index=True, right_index=True, how='left')
                else:
                    df['sector_etf_ret_1'] = np.nan

            # VIX delta
            if 'vix_delta_1' in self.indicator_list:
                vix_data = self.data_fetcher.get_vix_data()
                if vix_data is not None:
                    vix_delta_1 = vix_data['close'].diff().rename('vix_delta_1')
                    df = df.merge(vix_delta_1.to_frame(), left_index=True, right_index=True, how='left')
                else:
                    df['vix_delta_1'] = np.nan

            # Yield curve (2y-10y) - placeholder implementation
            if 'yc_2y10y_delta' in self.indicator_list:
                df['yc_2y10y_delta'] = 0.0  # Placeholder

        return df

    def merge_data(self, df=None):
        """Merge and finalize the data based on prediction mode"""
        if self.dataset_ex_df is None and df is None:
            raise ValueError("No data available for merging")
            
        if df is None:
            df = self.dataset_ex_df.copy()

        cols = ['Ticker']
        if self.prediction_mode:
            cols += list(self.indicator_list)
        else:
            cols += ['shifted_prices'] + list(self.indicator_list)

        # Filter columns that actually exist in the dataframe
        available_cols = [col for col in cols if col in df.columns]

        self.final_df = df[available_cols].replace(
            [float('inf'), float('-inf')], float('nan')
        ).dropna()

        return self.final_df

    def process_all(self):
        """
        Main method to process all data and indicators.

        Returns:
        --------
        pd.DataFrame
            Final processed DataFrame with all indicators
        """
        self.preprocess_data()
        self.add_features()
        return self.merge_data()

    def get_data_summary(self):
        """Get a summary of the processed data"""
        if self.final_df is None:
            return "No data processed yet. Call process_all() first."

        summary = {
            'shape': self.final_df.shape,
            'tickers': self.final_df['Ticker'].unique().tolist() if 'Ticker' in self.final_df.columns else [],
            'date_range': (self.final_df.index.min(), self.final_df.index.max()),
            'indicators': [col for col in self.final_df.columns if col not in ['Ticker', 'shifted_prices']],
            'missing_values': self.final_df.isnull().sum().sum()
        }

        return summary

    def get_ohlc_for_ticker(self, ticker, multiplier=1, timespan="day", limit=50000):
        """
        Fetch daily OHLC bars for `ticker` in one paginated call and
        align to full_dates, filling zeros on missing days.
        """
        if not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")

        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        aggs_iter = self.client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=self.start_date,
            to=self.end_date,
            limit=limit
        )  # returns an iterator over all pages :contentReference[oaicite:4]{index=4}

        aggs = pd.DataFrame(aggs_iter)
        if aggs.empty:
            # No data: return zero-filled template
            df_empty = pd.DataFrame(0, columns=["open", "high", "low", "close", "volume"])
            df_empty.index.name = "date"
            df_empty["ticker"] = ticker
            return df_empty

        # Convert timestamp → NY date
        dt_utc = pd.to_datetime(aggs['timestamp'], unit="ms", utc=True) \
            .dt.tz_convert('America/New_York')  # convert TZ :contentReference[oaicite:5]{index=5}
        aggs['date'] = dt_utc.dt.normalize()  # strip time → midnight
        aggs['Ticker'] = ticker

        daily = (
            aggs
            .loc[:, ["Ticker", "date", "open", "high", "low", "close", "volume"]]
            .set_index("date")
            .sort_index()
        )

        # 8) Reindex to full_dates, filling missing days with zeros
        daily.index.name = "date"
        daily["Ticker"] = ticker
        daily = daily.rename(columns={'ticker': 'Ticker','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})

        return daily
