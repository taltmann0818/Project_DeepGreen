"""
Data fetching module for TickerData.
Handles all API interactions with Polygon for fetching stock data, news, and market data.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pandas.tseries.holiday import USFederalHolidayCalendar
from dateutil.easter import easter


class DataFetcher:
    """Handles all data fetching operations from Polygon API"""

    def __init__(self, client, start_date=None, end_date=None, sample_size=None, days=1):
        """
        Initialize the DataFetcher with API credentials and date range.

        Parameters:
        -----------
        api_key : str
            Polygon API key
        start_date : str, optional
            Start date for data fetching (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for data fetching (format: 'YYYY-MM-DD')
        years : int, default=1
            Number of years of historical data to fetch if dates not provided
        """
        self.client = client
        self.days = days
        self.sample_size = sample_size

        if not start_date:
            self.start_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")
        else:
            self.start_date = start_date

        if not end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
        else:
            self.end_date = end_date

        # Cache for SIC codes to avoid repeated API calls
        self.sic_code_cache = {}
        self.last_api_call_time = 0
        self.api_call_delay = 0.2  # 200ms delay between API calls to avoid rate limiting

    def get_news_for_ticker(self, ticker, start_date, end_date, full_dates, limit=1000):
        """
        Fetch news sentiment data for a ticker.

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date for news data
        end_date : str
            End date for news data
        full_dates : list
            List of all dates to include in output
        limit : int, default=1000
            Maximum number of articles to fetch

        Returns:
        --------
        pd.DataFrame
            DataFrame with news sentiment data indexed by date
        """
        # 1) Fetch all articles in one paginated iterator
        articles = self.client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=start_date,
            published_utc_lte=end_date,
            limit=limit,
            sort="published_utc",
            order="asc"
        )
        # 2) Flatten into rows of (ticker, date, sentiment)
        rows = [
            (ticker, art.published_utc.split("T")[0], ins.sentiment)
            for art in articles
            for ins in (art.insights or [])
        ]
        # If no news at all, return zeros for every date
        if not rows:
            df_empty = pd.DataFrame(0,
                                    index=full_dates,
                                    columns=["positive", "neutral", "negative"]
                                    )
            df_empty.index.name = "date"
            df_empty["Ticker"] = ticker
            return df_empty

        df = pd.DataFrame(rows, columns=["Ticker", "date", "sentiment"])
        # 3) Pivot daily counts
        daily = (
            df.groupby(["Ticker", "date", "sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for col in ['bearish', 'bullish', 'hold', 'mixed', 'negative', 'positive','neutral']:
            if col not in daily.columns:
                daily[col] = 0

        daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize('America/New_York')

        return daily.set_index(["date"])

    def get_dail_aggs(self, date):
        return pd.DataFrame(
            self.client.get_grouped_daily_aggs(
                date,
                adjusted="true",
            )
        )

    def get_spy_data(self):
        """Get SPY data for cross-asset indicators"""
        try:
            # Use the same data fetching mechanism as for other tickers
            spy_data = self.get_ohlc_for_ticker('SPY', multiplier=1, timespan="day", limit=50000)
            if spy_data is not None and not spy_data.empty:
                spy_data.index = pd.to_datetime(spy_data.index).tz_localize(None)
                return spy_data
        except Exception as e:
            print(f"Warning: Could not fetch SPY data: {e}")
        return None

    def get_vix_data(self):
        """Get VIX data for volatility indicators"""
        try:
            # VIX data from Polygon
            vix_data = self.get_ohlc_for_ticker('VIX', multiplier=1, timespan="day", limit=50000)
            if vix_data is not None and not vix_data.empty:
                vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
                return vix_data
        except Exception as e:
            print(f"Warning: Could not fetch VIX data: {e}")
        return None

    def get_sector_etf_data(self, sector_etfs=['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU']):
        """Get sector ETF data"""
        sector_data = {}
        for etf in sector_etfs:
            try:
                etf_data = self.get_ohlc_for_ticker(etf, multiplier=1, timespan="day", limit=50000)
                if etf_data is not None and not etf_data.empty:
                    etf_data.index = pd.to_datetime(etf_data.index).tz_localize(None)
                    sector_data[etf] = etf_data
            except Exception as e:
                print(f"Warning: Could not fetch {etf} data: {e}")
        return sector_data

    def get_sic_code_for_ticker(self, ticker):
        """Get SIC code for a ticker using Polygon API with rate limiting and caching"""
        # Check cache first
        if ticker in self.sic_code_cache:
            return self.sic_code_cache[ticker]

        try:
            details = self.client.get_ticker_details(ticker)
            sic_code = getattr(details, 'sic_code', None)

            # Cache the result (even if None)
            self.sic_code_cache[ticker] = sic_code
            return sic_code

        except Exception as e:
            error_str = str(e).lower()
            # For non-rate-limiting errors, don't retry
            #print(f"Warning: Could not fetch SIC code for {ticker}: {e}")
            # Cache the failure
            self.sic_code_cache[ticker] = None
            return None

    def generate_us_market_holidays(self, start_year: int, end_year: int) -> np.ndarray:
        # 1) federal holidays
        cal = USFederalHolidayCalendar()
        fed = cal.holidays(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31"
        )
        # 2) Good Fridays by computing Easter Sunday then subtracting 2 days
        good_fridays = [
            easter(yr) - timedelta(days=2)
            for yr in range(start_year, end_year + 1)
        ]
        # combine and return as np.datetime64[D]
        all_hols = pd.DatetimeIndex(fed).union(pd.DatetimeIndex(good_fridays))
        return all_hols.values.astype("datetime64[D]")

    def fetch_stock_data(self, workers=20):
        """
        Fetch stock data for multiple tickers in parallel.

        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        workers : int, default=20
            Number of worker threads for parallel processing

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with all ticker data
        """

        full_dates = np.array(
            pd.date_range(start=self.start_date, end=self.end_date, freq="D", tz="America/New_York")
            .strftime("%Y-%m-%d")
        ).astype("datetime64[D]")

        holidays = self.generate_us_market_holidays(
            start_year=int(self.start_date[:4]),
            end_year=int(self.end_date[:4])
        )
        extra = np.array(['2025-01-09'], dtype='datetime64[D]')
        holidays = np.union1d(holidays, extra)
        business_mask = np.is_busday(full_dates, holidays=holidays)
        business_days = full_dates[business_mask].astype("datetime64[D]")
        business_days = business_days.astype(str)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            all_results = ex.map(lambda d: self.get_dail_aggs(d), business_days)
        dfs = [df for df in all_results if df is not None]
        all_data = pd.concat(dfs, axis=0, ignore_index=True)

        dt_utc = pd.to_datetime(all_data['timestamp'], unit="ms", utc=True).dt.tz_convert('America/New_York')
        all_data['date'] = dt_utc.dt.normalize()  # strip time → midnight
        all_data['date'] = all_data['date'].dt.tz_localize(None)
        # Keep date as a column instead of setting it as index to avoid duplicate labels
        all_data = all_data.loc[:, ["ticker", "date", "open", "high", "low", "close", "volume"]].sort_values(['date', 'ticker'])

        # Remove duplicate rows (same ticker and date) to avoid duplicate index issues
        all_data = all_data.drop_duplicates(subset=['ticker', 'date'], keep='last')

        stock_data = all_data.rename(columns={'ticker': 'Ticker','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})

        if self.sample_size is not None:
            tickers = stock_data['Ticker'].unique()
            tickers = np.random.choice(tickers, self.sample_size)
            stock_data = stock_data[stock_data['Ticker'].isin(tickers)]

        return stock_data