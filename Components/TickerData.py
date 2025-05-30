from concurrent.futures import ThreadPoolExecutor
from polygon import RESTClient
import warnings
import numpy as np
import pandas as pd
import pywt
import antropy as ant
from pyts.decomposition import SingularSpectrumAnalysis
from pydmd import DMD
from hurst import compute_Hc

from datetime import datetime, timedelta
import os

from Components.MarketRegimes import RegimeDetector

class TickerData:
    def __init__(self, tickers, indicator_list, years=1, prediction_window=5,**kwargs):
        """
        Initialize the StockAnalyzer with a ticker symbol and number of past days to fetch.
        """
        self.client = RESTClient(os.environ["POLYGON_API_KEY"])
        self.tickers = tickers
        self.indicator_list = set(indicator_list)
        self.prediction_window = -abs(prediction_window)
        self.years = years
        self.days = years * 365
        if years > 5:
            raise ValueError("Max years is 5 due to API limits.")
        self.start_date = kwargs.get('start_date')
        self.end_date   = kwargs.get('end_date')
        if not self.start_date:
            self.start_date = (datetime.now() - timedelta(days=self.days)).strftime("%Y-%m-%d")
        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.prediction_mode = kwargs.get('prediction_mode', False)

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

        for col in ['bearish', 'bullish', 'hold', 'mixed', 'negative', 'positive','neutral']:
            if col not in daily.columns:
                daily[col] = 0

        daily['date'] = pd.to_datetime(daily['date']).dt.tz_localize('America/New_York')

        return daily.set_index(["date"])

    def get_ohlc_for_ticker(self, ticker, multiplier=1, timespan="day", limit=50000):
        """
        Fetch daily OHLC bars for `ticker` in one paginated call and
        align to full_dates, filling zeros on missing days.
        """
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

        return daily[["Ticker", "open", "high", "low", "close", "volume"]]

    def get_index_data(self):
        nasdaq_close = TickerData('I:NDX', [], years=self.years).get_ohlc_for_ticker('I:NDX')

        return nasdaq_close

    def fetch_stock_data(self, workers=20):

        full_dates = pd.date_range(start=self.start_date,end=self.end_date,freq="D",tz="America/New_York")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            stock_dfs = ex.map(lambda t: self.get_ohlc_for_ticker(t), self.tickers)
        self.stock_data = pd.concat(stock_dfs, axis=0)

        if 'positive' in self.indicator_list:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                news_dfs = ex.map(lambda t: self.get_news_for_ticker(t, self.start_date, self.end_date, full_dates), self.tickers)
            self.news_data = pd.concat(news_dfs, axis=0)
        else:
            self.news_data = None

        return self.stock_data, self.news_data

    # ——— Preprocessing (unchanged) ———
    def preprocess_data(self):
        self.dataset_ex_df = (
            self.stock_data
            .rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        )
        # Merge in MarketRegimes
        if 'hmm_state' in self.indicator_list:
            _, self.dataset_ex_df['hmm_state']  = RegimeDetector.load("Models/hmm_v2.pkl").predict(self.dataset_ex_df, ma=5)
        if not self.prediction_mode:
            self.dataset_ex_df = self.dataset_ex_df.sort_values(['Ticker', 'Date'])
            self.dataset_ex_df['shifted_prices'] = (
                self.dataset_ex_df
                .groupby('Ticker')['Close']
                .shift(self.prediction_window)
            )

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
    def compute_cmf(data, period=20):
        close, low, high, volume = data.Close, data.Low, data.High, data.Volume
        mfm = ((close - low) - (high - close)) / (high - low)  # money flow multiplier
        mfm = mfm.fillna(0)
        mfv = mfm * volume  # money flow volume
        mfv_sum = mfv.rolling(period).sum()
        vol_sum = volume.rolling(period).sum()
        cmf = mfv_sum / vol_sum  # Chaikin Money Flow = sum(MFV)/sum(Volume) over period
        return cmf

    @staticmethod
    def compute_cci(data, period=20):
        close, low, high = data.Close, data.Low, data.High
        tp = (high + low + close) / 3.0  # typical price each day
        sma_tp = tp.rolling(period).mean()  # simple moving average of typical price
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)  # mean absolute deviation
        cci = (tp - sma_tp) / (0.015 * mad) # Compute CCI with the 0.015 scaling factor

        return cci

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
    def compute_parabolic_sar(data, step=0.02, max_step=0.2):
        # step: acceleration factor increment (usually 0.02), max_step: max AF (usually 0.2)
        close, low, high = data.Close, data.Low, data.High

        length = len(close)
        sar = [0.0] * length
        # Initialization: Start with first trend assumption (e.g., assume first period is an uptrend)
        # Typically initialize SAR as previous period's extreme.
        # We'll start by assuming an uptrend from first day to second for initialization:
        sar[0] = low.iloc[0]  # initial SAR at first low (for uptrend start)
        bull = True  # start as bullish trend
        af = step
        ep = high.iloc[0]  # extreme price (highest high in uptrend)
        for i in range(1, length):
            prev_sar = sar[i - 1]
            if bull:
                # Uptrend: SAR = prev SAR + AF * (EP - prev SAR)
                sar[i] = prev_sar + af * (ep - prev_sar)
                # Ensure SAR is below the last two lows (cannot rise above actual price lows in uptrend)
                sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[i])
                # Update extreme point and acceleration factor if new high made
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + step, max_step)
                # Check for trend reversal
                if close.iloc[i] < sar[i]:
                    # flip to downtrend
                    bull = False
                    sar[i] = ep  # on reversal, SAR starts at last EP (last high)
                    af = step
                    ep = low.iloc[i]  # reset extreme to current low
            else:
                # Downtrend: SAR = prev SAR - AF * (prev SAR - EP)
                sar[i] = prev_sar - af * (prev_sar - ep)
                # Ensure SAR is above the last two highs
                sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[i])
                # Update extreme point if new low made
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + step, max_step)
                # Check for reversal
                if close.iloc[i] > sar[i]:
                    bull = True
                    sar[i] = ep  # on reversal, SAR starts at last EP (last low)
                    af = step
                    ep = high.iloc[i]
        return pd.Series(sar, index=close.index)

    @staticmethod
    def dmd_prediction(series, modes=2):
        dmd = DMD(svd_rank=modes)
        dmd.fit(series.values.reshape(1, -1))
        reconstruction = dmd.reconstructed_data.real.flatten()
        return pd.Series(reconstruction, index=series.index)

    @staticmethod
    def ssa_trend(series, window_size=20):
        # groups = [[0]] means “only reconstruct component 0 (the one with the largest singular value)”
        ssa = SingularSpectrumAnalysis(window_size=window_size, groups=[[0]])
        X = series.values.reshape(1, -1)  # shape (1, n_timestamps)
        reconstructed = ssa.fit_transform(X)  # now shape (1, 1, n_timestamps)
        trend = reconstructed.squeeze()  # collapse to shape (n_timestamps,)
        return pd.Series(trend, index=series.index)

    @staticmethod
    def wavelet_denoise(series, wavelet='db4', level=1):
        # Perform discrete wavelet transform
        coeffs = pywt.wavedec(series.values, wavelet, level=level)
        # Zero-out the highest detail coefficients to remove noise
        coeffs[-1] = np.zeros_like(coeffs[-1])
        # Reconstruct the series from modified coefficients
        reconstructed = pywt.waverec(coeffs, wavelet)
        # Ensure the reconstructed series has the same length
        reconstructed = reconstructed[:len(series)]
        return pd.Series(reconstructed, index=series.index)

    @staticmethod
    def hurst_window(x):
        # x is a 1-D NumPy array of length WINDOW
        H, c, data = compute_Hc(x, kind='price', simplified=True)
        return H

    @staticmethod
    def perm_entropy_window(x):
        # x is the window array
        return ant.perm_entropy(x, order=3, normalize=True)

    @staticmethod
    def fractal(x):
        # x is a 1-D NumPy array
        # choose m (embedding dim) and r (tolerance), e.g. m=2, r=0.2*std
        return ant.higuchi_fd(x)

    @staticmethod
    def sampen_window(x):
        # x is a 1-D NumPy array
        return ant.sample_entropy(x, order=2, metric='chebyshev')

    # ——— Core Refactored Indicator Loop ———
    def add_technical_indicators(self):
        df = self.dataset_ex_df.copy()

        # — EMA (single-series via transform) —
        for period in (20, 50, 100, 200):
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
        if 'b_percent' in self.indicator_list:
            df['b_percent'] = df.groupby('Ticker')['Close'].transform(self.bollinger_percent_b)

        if 'macd' in self.indicator_list:
            df['macd'] = df.groupby('Ticker')['Close'].transform(self.macd)

        if 'ssa_trend' in self.indicator_list:
            df['ssa_trend'] = df.groupby('Ticker')['Close'].apply(self.ssa_trend).reset_index(level=0, drop=True)

        if 'dmd' in self.indicator_list:
            df['dmd'] = df.groupby('Ticker')['Close'].apply(self.dmd_prediction).reset_index(level=0, drop=True)

        if 'close_denoised_L1' in self.indicator_list:
            df['close_denoised_L1'] = df.groupby('Ticker')['Close'].apply(self.wavelet_denoise).reset_index(level=0, drop=True)

        if 'cmf' in self.indicator_list:
            df['cmf'] = df.groupby('Ticker')[['High', 'Low', 'Close', 'Volume']].apply(
                lambda x: self.compute_cmf(x)['cmf'] if isinstance(self.compute_cmf(x),
                                                                   pd.DataFrame) else self.compute_cmf(x)
            ).reset_index(level=0, drop=True)

        if 'cci' in self.indicator_list:
            df['cci'] = df.groupby('Ticker')[['High','Low','Close','Volume']].apply(self.compute_cci).reset_index(level=0, drop=True)

        if 'parabolic_sar' in self.indicator_list:
            df['parabolic_sar'] = df.groupby('Ticker')[['High','Low','Close']].apply(self.compute_parabolic_sar).reset_index(level=0, drop=True)

        if 'hurst_100' in self.indicator_list:
            df['hurst_100'] = (
                df
                .groupby('Ticker')['Close']
                .apply(lambda s: s.rolling(100)
                       .apply(self.hurst_window, raw=True))
                .reset_index(level=0, drop=True)
            )

        if 'perm_entropy_50' in self.indicator_list:
            df['perm_entropy_50'] = (
                df
                .groupby('Ticker')['Close']
                .apply(lambda s: s.rolling(50)
                       .apply(self.perm_entropy_window, raw=True))
                .reset_index(level=0, drop=True)
            )

        if 'sampen_50' in self.indicator_list:
            df['sampen_50'] = (
                df
                .groupby('Ticker')['Close']
                .apply(lambda s: s.rolling(50)
                       .apply(self.sampen_window, raw=True))
                .reset_index(level=0, drop=True)
            )

        if 'fractal_50' in self.indicator_list:
            df['fractal_50'] = (
                df
                .groupby('Ticker')['Close']
                .apply(lambda s: s.rolling(50)
                       .apply(self.fractal, raw=True))
                .reset_index(level=0, drop=True)
            )

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

        if 'nasdaq_rsi' or 'nasdaq_returns' in self.indicator_list:
            nasdaq_close = self.get_index_data()
            if 'nasdaq_rsi' in self.indicator_list:
                nasdaq_rsi = self.stochastic_rsi(nasdaq_close['close']).rename('nasdaq_rsi')
                df = df.merge(nasdaq_rsi.to_frame(), left_index=True, right_index=True)

            if 'nasdaq_returns' in self.indicator_list:
                nasdaq_returns = nasdaq_close['close'].pct_change().rename('nasdaq_returns')
                df = df.merge(nasdaq_returns.to_frame(), left_index=True, right_index=True)

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

#indicators = ['ema_20', 'ema_50', 'ema_100', 'stoch_rsi14', 'macd', 'b_percent', 'hmm_state', 'ssa_trend', 'dmd', 'hurst_100','perm_entropy_50','close_denoised_L1', 'Close','fractal_50','sampen_50','cci','cmf','parabolic_sar','keltner_upper','keltner_lower','nasdaq_rsi','nasdaq_returns']
