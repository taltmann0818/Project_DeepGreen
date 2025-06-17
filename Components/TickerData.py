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
    def __init__(self, data, indicator_list, years=1, prediction_window=5, **kwargs):
        """
        Initialize the StockAnalyzer with a ticker symbol and number of past days to fetch.

        Parameters:
        -----------
        data : str or DataFrame
            Ticker symbol or DataFrame containing stock data
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
                If None, uses min(32, os.cpu_count() + 4)
        """
        self.client = RESTClient('XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2')
        self.data = data
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
        self.max_workers = kwargs.get('max_workers', None)

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
        self.dataset_ex_df = self.data
        # Merge in MarketRegimes
        if 'hmm_state' in self.indicator_list:
            _, self.dataset_ex_df['hmm_state']  = RegimeDetector.load("Models/hmm_v2.pkl").predict(self.dataset_ex_df, ma=5)
        if not self.prediction_mode:
            self.dataset_ex_df = self.dataset_ex_df.sort_values(['Ticker', 'date'])
            self.dataset_ex_df['shifted_prices'] = (
                self.dataset_ex_df
                .groupby('Ticker')['Close']
                .shift(self.prediction_window)
            )

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

        # Vectorized calculation of mean absolute deviation
        # First calculate the rolling mean
        rolling_mean = tp.rolling(period).mean()
        # Then calculate the absolute deviation from the mean
        abs_dev = np.abs(tp - rolling_mean.shift(0))
        # Finally calculate the mean of the absolute deviations
        mad = abs_dev.rolling(period).mean()

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
        close = data.Close.values  # Convert to numpy arrays for faster access
        low = data.Low.values
        high = data.High.values
        index = data.Close.index  # Save index for later

        length = len(close)
        sar = np.zeros(length)
        # Initialization: Start with first trend assumption (e.g., assume first period is an uptrend)
        sar[0] = low[0]  # initial SAR at first low (for uptrend start)
        bull = True  # start as bullish trend
        af = step
        ep = high[0]  # extreme price (highest high in uptrend)

        # Pre-allocate arrays for trend tracking
        is_bull = np.ones(length, dtype=bool)
        is_bull[0] = bull

        for i in range(1, length):
            prev_sar = sar[i - 1]
            if bull:
                # Uptrend: SAR = prev SAR + AF * (EP - prev SAR)
                sar[i] = prev_sar + af * (ep - prev_sar)
                # Ensure SAR is below the last two lows (cannot rise above actual price lows in uptrend)
                sar[i] = min(sar[i], low[i - 1], low[i])
                # Update extreme point and acceleration factor if new high made
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
                # Check for trend reversal
                if close[i] < sar[i]:
                    # flip to downtrend
                    bull = False
                    is_bull[i] = False
                    sar[i] = ep  # on reversal, SAR starts at last EP (last high)
                    af = step
                    ep = low[i]  # reset extreme to current low
            else:
                # Downtrend: SAR = prev SAR - AF * (prev SAR - EP)
                sar[i] = prev_sar - af * (prev_sar - ep)
                # Ensure SAR is above the last two highs
                sar[i] = max(sar[i], high[i - 1], high[i])
                # Update extreme point if new low made
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
                # Check for reversal
                if close[i] > sar[i]:
                    bull = True
                    is_bull[i] = True
                    sar[i] = ep  # on reversal, SAR starts at last EP (last low)
                    af = step
                    ep = high[i]

            # Store current trend state
            is_bull[i] = bull

        return pd.Series(sar, index=index)

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
        # Convert to ndarray and drop NaNs created by the rolling window
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        # Too few points → not enough information
        if x.size < 20:
            return np.nan
        # Constant series → variance = 0 → RS statistic undefined
        if np.allclose(x, x[0]):
            return np.nan
        try:
            # Ignore benign warnings inside the called routine
            with np.errstate(all="ignore"):
                H, _, _ = compute_Hc(x, kind="price", simplified=True)
            return H if np.isfinite(H) else np.nan
        except (FloatingPointError, ValueError):
            # Any numerical issue ⇒ treat window as invalid
            return np.nan

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
    def atr(close, high, low, period=14):
        true_range = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    # ——— Optimized Core Indicator Loop ———
    def add_technical_indicators(self):
        self.dataset_ex_df.index = pd.to_datetime(self.dataset_ex_df.index).tz_localize(None)
        df = self.dataset_ex_df.copy()

        # Group tickers once to avoid redundant groupby operations
        ticker_groups = {ticker: group for ticker, group in df.groupby('Ticker')}

        # Function to process indicators for a single ticker
        def process_ticker_indicators(ticker_data):
            ticker, data = ticker_data
            result = {}

            # Process EMA indicators
            for period in (20, 50, 100, 200):
                name = f'ema_{period}'
                if name in self.indicator_list:
                    result[name] = self.ema(data['Close'], period)

            # Process Stochastic RSI indicators
            for period in (14, 28):
                name = f'stoch_rsi{period}'
                if name in self.indicator_list:
                    result[name] = self.stochastic_rsi(data['Close'], period)

            # Process single-series indicators
            if 'b_percent' in self.indicator_list:
                result['b_percent'] = self.bollinger_percent_b(data['Close'])

            if 'macd' in self.indicator_list:
                result['macd'] = self.macd(data['Close'])

            if 'ssa_trend' in self.indicator_list:
                result['ssa_trend'] = self.ssa_trend(data['Close'])

            if 'dmd' in self.indicator_list:
                result['dmd'] = self.dmd_prediction(data['Close'])

            if 'close_denoised_L1' in self.indicator_list:
                result['close_denoised_L1'] = self.wavelet_denoise(data['Close'])

            if 'cmf' in self.indicator_list:
                cmf_result = self.compute_cmf(data[['High', 'Low', 'Close', 'Volume']])
                result['cmf'] = cmf_result['cmf'] if isinstance(cmf_result, pd.DataFrame) else cmf_result

            if 'cci' in self.indicator_list:
                result['cci'] = self.compute_cci(data[['High', 'Low', 'Close', 'Volume']])

            if 'parabolic_sar' in self.indicator_list:
                result['parabolic_sar'] = self.compute_parabolic_sar(data[['High', 'Low', 'Close']])

            # Process rolling window calculations more efficiently
            close_values = data['Close'].values
            close_index = data['Close'].index

            # Pre-calculate rolling windows for better performance
            if any(indicator in self.indicator_list for indicator in ['hurst_100', 'perm_entropy_50', 'sampen_50', 'fractal_50']):
                # Create a dictionary to store rolling window results
                rolling_results = {}

                if 'hurst_100' in self.indicator_list:
                    # Process in chunks for better performance
                    hurst_values = np.full(len(close_values), np.nan)
                    valid_indices = np.arange(100, len(close_values))

                    # Process in smaller batches to avoid memory issues
                    batch_size = 1000
                    for i in range(0, len(valid_indices), batch_size):
                        batch_indices = valid_indices[i:i+batch_size]
                        for j in batch_indices:
                            window = close_values[j-100:j]
                            hurst_values[j] = self.hurst_window(window)

                    rolling_results['hurst_100'] = pd.Series(hurst_values, index=close_index)

                if 'perm_entropy_50' in self.indicator_list:
                    perm_values = np.full(len(close_values), np.nan)
                    valid_indices = np.arange(50, len(close_values))

                    # Process in smaller batches
                    batch_size = 1000
                    for i in range(0, len(valid_indices), batch_size):
                        batch_indices = valid_indices[i:i+batch_size]
                        for j in batch_indices:
                            window = close_values[j-50:j]
                            perm_values[j] = self.perm_entropy_window(window)

                    rolling_results['perm_entropy_50'] = pd.Series(perm_values, index=close_index)

                if 'sampen_50' in self.indicator_list:
                    sampen_values = np.full(len(close_values), np.nan)
                    valid_indices = np.arange(50, len(close_values))

                    # Process in smaller batches
                    batch_size = 1000
                    for i in range(0, len(valid_indices), batch_size):
                        batch_indices = valid_indices[i:i+batch_size]
                        for j in batch_indices:
                            window = close_values[j-50:j]
                            sampen_values[j] = self.sampen_window(window)

                    rolling_results['sampen_50'] = pd.Series(sampen_values, index=close_index)

                if 'fractal_50' in self.indicator_list:
                    fractal_values = np.full(len(close_values), np.nan)
                    valid_indices = np.arange(50, len(close_values))

                    # Process in smaller batches
                    batch_size = 1000
                    for i in range(0, len(valid_indices), batch_size):
                        batch_indices = valid_indices[i:i+batch_size]
                        for j in batch_indices:
                            window = close_values[j-50:j]
                            fractal_values[j] = self.fractal(window)

                    rolling_results['fractal_50'] = pd.Series(fractal_values, index=close_index)

                # Add results to the result dictionary
                result.update(rolling_results)

            # Process multi-output indicators
            if {'keltner_upper', 'keltner_lower'} & set(self.indicator_list):
                upper, lower = self.keltner_channel(data['High'], data['Low'], data['Close'])
                if 'keltner_upper' in self.indicator_list:
                    result['keltner_upper'] = upper
                if 'keltner_lower' in self.indicator_list:
                    result['keltner_lower'] = lower

            if {'price_momentum', 'volume_momentum'} & set(self.indicator_list):
                price_mom, volume_mom = self.momentum_signals(data['Close'], data['Volume'])
                if 'price_momentum' in self.indicator_list:
                    result['price_momentum'] = price_mom
                if 'volume_momentum' in self.indicator_list:
                    result['volume_momentum'] = volume_mom

            if {'bullish_engulfing', 'bearish_engulfing'} & set(self.indicator_list):
                bullish, bearish = self.engulfing_patterns(data['Open'], data['Close'])
                if 'bullish_engulfing' in self.indicator_list:
                    result['bullish_engulfing'] = bullish
                if 'bearish_engulfing' in self.indicator_list:
                    result['bearish_engulfing'] = bearish

            return ticker, result

        # Process NASDAQ data once if needed
        if {'nasdaq_rsi', 'nasdaq_returns'} & set(self.indicator_list):
            nasdaq_close = self.get_index_data()
            nasdaq_close.index = pd.to_datetime(nasdaq_close.index).tz_localize(None)
            if 'nasdaq_rsi' in self.indicator_list:
                nasdaq_rsi = self.stochastic_rsi(nasdaq_close['close']).rename('nasdaq_rsi')
                df = df.merge(nasdaq_rsi.to_frame(), left_index=True, right_index=True)
            if 'nasdaq_returns' in self.indicator_list:
                nasdaq_returns = nasdaq_close['close'].pct_change().rename('nasdaq_returns')
                df = df.merge(nasdaq_returns.to_frame(), left_index=True, right_index=True)

        # Process indicators in parallel for each ticker
        # Use self.max_workers if specified, otherwise use a sensible default
        workers = self.max_workers if self.max_workers is not None else min(os.cpu_count(), len(ticker_groups))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_ticker_indicators, ticker_groups.items()))

        self.indicator_results = results

        # Update the dataframe with the calculated indicators
        for ticker, ticker_results in self.indicator_results:
            ticker_idx = ticker_groups[ticker].index  # cache ‑ small speed gain
            for indicator, values in ticker_results.items():
                if indicator not in df.columns:
                    df[indicator] = np.nan
                df.loc[ticker_idx, indicator] = values

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
        #self.fetch_stock_data()
        self.preprocess_data()
        self.add_technical_indicators()
        return self.merge_data()

#indicators = ['ema_20', 'ema_50', 'ema_100', 'stoch_rsi14', 'macd', 'b_percent', 'hmm_state', 'ssa_trend', 'dmd', 'hurst_100','perm_entropy_50','close_denoised_L1', 'Close','fractal_50','sampen_50','cci','cmf','parabolic_sar','keltner_upper','keltner_lower','nasdaq_rsi','nasdaq_returns']
