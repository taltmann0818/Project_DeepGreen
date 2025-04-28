from concurrent.futures import ThreadPoolExecutor
from polygon import RESTClient
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Components.MarketRegimes import MarketRegimes

try:
    from sqlalchemy import create_engine, MetaData, Table, select
except ImportError:
    create_engine = None
    MetaData = None
    Table = None
    select = None

import pandas as pd
import urllib.parse
import os

class TickerData:
    def __init__(self, tickers, indicator_list, years=1, prediction_window=5,**kwargs):
        """
        Initialize the StockAnalyzer with a ticker symbol and number of past days to fetch.
        """
        self.client = RESTClient(os.environ["POLYGON_API_KEY"])
        self.tickers = tickers
        self.indicator_list = set(indicator_list)
        self.prediction_window = abs(prediction_window)
        if years > 5:
            raise ValueError("Max years is 5 due to API limits.")
        self.start_date = kwargs.get('start_date')
        self.end_date   = kwargs.get('end_date')
        self.prediction_mode = kwargs.get('prediction_mode', False)
        self.days = years * 365

    def get_news_for_ticker(self, ticker, start_date, end_date, full_dates, limit=1000):
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

        for col in ['bearish', 'bullish', 'hold', 'mixed']:
            if col not in daily.columns:
                daily[col] = 0

        daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize('America/New_York')

        return daily.set_index(["date"])

    def get_ohlc_for_ticker(self, ticker, start_date, end_date, multiplier=1, timespan="day", limit=50000):
        """
        Fetch daily OHLC bars for `ticker` in one paginated call and
        align to full_dates, filling zeros on missing days.
        """
        aggs_iter = self.client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
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

        return daily[["Ticker", "open", "high", "low", "close", "volume"]]

    def get_index_vol(self, period=14, multiplier=1, timespan="day", limit=50000):
        aggs_iter = self.client.list_aggs(
            ticker='I:NDX',
            multiplier=multiplier,
            timespan=timespan,
            from_=self.start_date,
            to=self.end_date,
            limit=limit
        )  # returns an iterator over all pages :contentReference[oaicite:4]{index=4}
        aggs = pd.DataFrame(aggs_iter)
        dt_utc = pd.to_datetime(aggs['timestamp'], unit="ms", utc=True) \
            .dt.tz_convert('America/New_York')  # convert TZ :contentReference[oaicite:5]{index=5}
        aggs['date'] = dt_utc.dt.normalize()
        aggs.set_index("date")

        return self.stochastic_rsi(aggs['close'], rsi_period=period, stoch_period=period)

    def fetch_stock_data(self, workers=20):
        if not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        full_dates = pd.date_range(start=self.start_date,end=self.end_date,freq="D",tz="America/New_York")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            stock_dfs = ex.map(lambda t: self.get_ohlc_for_ticker(t, self.start_date, self.end_date), self.tickers)
        self.stock_data = pd.concat(stock_dfs, axis=0)

        if 'positive' in self.indicator_list:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                news_dfs = ex.map(lambda t: self.get_news_for_ticker(t, self.start_date, self.end_date, full_dates), self.tickers)
            self.news_data = pd.concat(news_dfs, axis=0)
        else:
            self.news_data = None

        return self.stock_data, self.news_data

    def get_fundamentals(self):
        pass

    # ——— Preprocessing (unchanged) ———
    def preprocess_data(self):
        self.dataset_ex_df = (
            self.stock_data
            .rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        )
        # Merge in MarketRegimes
        self.dataset_ex_df = MarketRegimes(self.dataset_ex_df, "hmm_model.pkl").run_regime_detection()
        if not self.prediction_mode:
            self.dataset_ex_df['shifted_prices'] = self.dataset_ex_df['Close'].shift(self.prediction_window)

        # Merge in news data if requested
        if self.news_data is not None:
            # bring 'date' back as a column in both
            df_stocks = self.dataset_ex_df.reset_index()  # date→ column
            df_news = self.news_data.reset_index()  # date→ column

            # perform the merge on date + Ticker
            merged = pd.merge(
                df_stocks,
                df_news,
                on=['date', 'Ticker'],
                how='left'
            ).fillna(0)
            # restore date as the index
            self.dataset_ex_df = merged.set_index('date')

        return self.dataset_ex_df

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    def trend(self, series, period1=8, period2=21, period3=55):
        ema_8 = self.ema(series, period1)
        ema_21 = self.ema(series, period2)
        ema_55 = self.ema(series, period3)
        return np.where(ema_8  > ema_21, 1, 0), np.where(ema_21 > ema_55, 1, 0)

    @staticmethod
    def z_score(series, period=50):
        ma_50 = series.rolling(window=period).mean()
        std_50 = series.rolling(window=period).std()
        return (series - ma_50) / std_50

    @staticmethod
    def stochastic_rsi(series, rsi_period=14, stoch_period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(rsi_period).mean() / loss.rolling(rsi_period).mean()
        rsi = 100 - (100 / (1 + rs))
        return (rsi - rsi.rolling(stoch_period).min()) / (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())

    @staticmethod
    def macd(series, fast_period=12, slow_period=26, signal_period=9):
        fast = series.ewm(span=fast_period, adjust=False).mean()
        slow = series.ewm(span=slow_period, adjust=False).mean()
        return fast - slow

    @staticmethod
    def atr(close, high, low, period=14):
        true_range = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def momentum_signals(close, volume):
        returns = close.pct_change()
        mom_1m = returns.rolling(21).sum()
        mom_3m = returns.rolling(63).sum()
        mom_6m = returns.rolling(126).sum()
        price_momentum = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m)
        volume_momentum = volume / volume.rolling(21).mean()

        return price_momentum, volume_momentum

    @staticmethod     
    def bollinger_percent_b(series, period=20, std_dev=2):
        sma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return (series - lower) / (upper - lower)
        
    @staticmethod
    def keltner_channel(high, low, close, ema_period=20, atr_period=10, multiplier=2):
        center = close.ewm(span=ema_period).mean()
        atr = (high - low).rolling(atr_period).mean()
        return center + multiplier * atr, center - multiplier * atr

    @staticmethod
    def adx(high, low, close, period=14, smoothing=14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(smoothing).mean()
        up = high.diff().clip(lower=0)
        down = (-low.diff()).clip(lower=0)
        pos = 100 * up.rolling(smoothing).mean() / atr
        neg = 100 * down.rolling(smoothing).mean() / atr
        dx = 100 * (pos - neg).abs() / (pos + neg)
        return dx.rolling(period).mean()

    @staticmethod
    def engulfing_patterns(open_, close):
        cur_b = close > open_
        prev_b = close.shift() > open_.shift()
        cur_body = (close - open_).abs()
        prev_body = (close.shift() - open_.shift()).abs()
        bull = ( cur_b & ~prev_b & (open_ <= close.shift()) & (close >= open_.shift()) & (cur_body > prev_body) )
        bear = (~cur_b &  prev_b & (open_ >= close.shift()) & (close <= open_.shift()) & (cur_body > prev_body) )
        return bull.astype(int), bear.astype(int)

    @staticmethod
    def williams_r(high, low, close, period=14):
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return (hh - close) / (hh - ll) * -1

    # ——— Core Refactored Indicator Loop ———
    def add_technical_indicators(self):
        df = self.dataset_ex_df.copy()

        # — EMA (single-series via transform) —
        for period in (20, 50, 200):
            name = f'ema_{period}'
            if name in self.indicator_list:
                df[name] = (
                    df
                    .groupby('Ticker')['Close']
                    .transform(lambda s: self.ema(s, period))
                )
        # — Stochastic RSI (single-series via transform) —
        for period in (14, 28):
            name = f'stoch_rsi{period}'
            if name in self.indicator_list:
                df[name] = (
                    df
                    .groupby('Ticker')['Close']
                    .transform(lambda s: self.stochastic_rsi(s, period))
                )

        # — Single-series indicators —
        if 'macd' in self.indicator_list:
            df['macd'] = df.groupby('Ticker')['Close'].transform(self.macd)

        if 'b_percent' in self.indicator_list:
            df['b_percent'] = df.groupby('Ticker')['Close'].transform(self.bollinger_percent_b)

        if 'z_score' in self.indicator_list:
            df['z_score'] = df.groupby('Ticker')['Close'].transform(self.z_score)

        # — Single-series but need multiple of OHLC —
        if 'adx' in self.indicator_list:
            adx_series = (
                df
                .groupby('Ticker')
                .apply(lambda g: self.adx(g['High'], g['Low'], g['Close']))
                .reset_index(level=0, drop=True)
            )
            df['adx'] = adx_series

        if 'williams_r' in self.indicator_list:
            wr_series = (
                df
                .groupby('Ticker')
                .apply(lambda g: self.williams_r(g['High'], g['Low'], g['Close']))
                .reset_index(level=0, drop=True)
            )
            df['williams_r'] = wr_series

        if 'atr' in self.indicator_list:
            atr_series = (
                df
                .groupby('Ticker')
                .apply(lambda g: self.atr(g['Close'], g['High'], g['Low']))
                .reset_index(level=0, drop=True)
            )
            df['atr'] = atr_series

        # — Engulfing patterns (two outputs: bullish & bearish) —
        if {'bullish_engulfing', 'bearish_engulfing'} & set(self.indicator_list):
            eng = (
                df
                .groupby('Ticker')
                .apply(lambda g: pd.DataFrame({
                    'bullish_engulfing': self.engulfing_patterns(g['Open'], g['Close'])[0],
                    'bearish_engulfing': self.engulfing_patterns(g['Open'], g['Close'])[1],
                }, index=g.index))
                .reset_index(level=0, drop=True)
            )
            if 'bullish_engulfing' in self.indicator_list:
                df['bullish_engulfing'] = eng['bullish_engulfing']
            if 'bearish_engulfing' in self.indicator_list:
                df['bearish_engulfing'] = eng['bearish_engulfing']

        # — Keltner Channel (two outputs: upper & lower) —
        if {'keltner_upper', 'keltner_lower'} & set(self.indicator_list):
            kc = (
                df
                .groupby('Ticker')
                .apply(lambda g: pd.DataFrame({
                    'keltner_upper': self.keltner_channel(g['High'], g['Low'], g['Close'])[0],
                    'keltner_lower': self.keltner_channel(g['High'], g['Low'], g['Close'])[1],
                }, index=g.index))
                .reset_index(level=0, drop=True)
            )
            if 'keltner_upper' in self.indicator_list:
                df['keltner_upper'] = kc['keltner_upper']
            if 'keltner_lower' in self.indicator_list:
                df['keltner_lower'] = kc['keltner_lower']

        # — Momentum (two outputs: price & volume) —
        if {'price_momentum', 'volume_momentum'} & set(self.indicator_list):
            momentum = (
                df
                .groupby('Ticker')
                .apply(lambda g: pd.DataFrame({
                    'price_momentum': self.momentum_signals(g['Close'], g['Volume'])[0],
                    'volume_momentum': self.momentum_signals(g['Close'], g['Volume'])[1],
                }, index=g.index))
                .reset_index(level=0, drop=True)
            )
            if 'price_momentum' in self.indicator_list:
                df['price_momentum'] = momentum['price_momentum']
            if 'volume_momentum' in self.indicator_list:
                df['volume_momentum'] = momentum['volume_momentum']

        self.dataset_ex_df = df
        return df

    # ——— Final Merge Based on indicators & mode ———
    def merge_data(self):
        cols = ['Ticker']
        if self.prediction_mode:
            cols += list(self.indicator_list)
        else:
            cols += ['shifted_prices'] + list(self.indicator_list)
        self.final_df = self.dataset_ex_df[cols].dropna()
        return self.final_df

    def process_all(self):
        self.fetch_stock_data()
        self.preprocess_data()
        self.add_technical_indicators()
        return self.merge_data(), self.stock_data

def upload_data_sql(data_to_upload, table_name,chunksize=100):
    try:
        # Get connection string from environment variables
        connection_string = os.environ["AZURE_SQL_CONNECTIONSTRING"]
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}")

        data_to_upload = data_to_upload.reset_index().rename(columns={'index': 'Date'})
        data_to_upload = data_to_upload[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        # Convert Date column to a SQL-compatible format (remove timezone information)
        if pd.api.types.is_datetime64_any_dtype(data_to_upload['Date']):
            data_to_upload['Date'] = data_to_upload['Date'].dt.tz_localize(None)

        print("Sample data to be uploaded:")
        print(data_to_upload.head()) # Print a small sample of the data for verification

        # Upload the dataframe to SQL
        data_to_upload.to_sql(
            name=table_name,
            con=engine,
            if_exists='append',  # 'replace' if you want to overwrite, 'append' to add to existing
            index=False,
            chunksize=chunksize
        )

        print(f"Successfully uploaded {len(data_to_upload)} records to {table_name} table")

    except Exception as e:
        print(f"Error uploading data: {str(e)}")

        # Provide additional debugging information
        if 'data_to_upload' in locals():
            print("Data sample at the time of error:")
            print(data_to_upload.head())


def fetch_sql_data(table_name):
    try:
        # Get connection string from environment variables
        connection_string = os.environ["AZURE_SQL_CONNECTIONSTRING"]
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}")

        # Establish connection
        with engine.connect() as connection:
            # Prepare base SELECT query
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=engine)

            # Execute query and fetch results into a pandas DataFrame
            result = connection.execute(select(table))
            data = pd.DataFrame(result.fetchall(), columns=result.keys())

        return data

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
