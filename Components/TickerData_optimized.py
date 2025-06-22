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
import numba
from numba import jit

from datetime import datetime, timedelta
import os
import time

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

        # Cache for SIC codes to avoid repeated API calls
        self.sic_code_cache = {}
        self.last_api_call_time = 0
        self.api_call_delay = 0.2  # 200ms delay between API calls to avoid rate limiting

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
        )

        aggs = pd.DataFrame(aggs_iter)
        if aggs.empty:
            # No data: return zero-filled template
            df_empty = pd.DataFrame(0, columns=["open", "high", "low", "close", "volume"])
            df_empty.index.name = "date"
            df_empty["ticker"] = ticker
            return df_empty

        # Convert timestamp → NY date
        dt_utc = pd.to_datetime(aggs['timestamp'], unit="ms", utc=True) \
            .dt.tz_convert('America/New_York')

        # Create daily DataFrame
        daily = (
            pd.DataFrame({
                "open": aggs['open'],
                "high": aggs['high'],
                "low": aggs['low'],
                "close": aggs['close'],
                "volume": aggs['volume']
            }, index=dt_utc.dt.date)
            .groupby(level=0)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            })
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
        for i in range(1, length):
            # Calculate SAR for current period
            sar[i] = sar[i-1] + af * (ep - sar[i-1])

            # Check for trend reversal
            if bull:  # Currently in uptrend
                # Check if SAR is above the low (reversal signal)
                if sar[i] > low[i]:
                    bull = False  # Switch to downtrend
                    sar[i] = ep  # SAR becomes the previous extreme point
                    ep = low[i]  # New extreme point is current low
                    af = step  # Reset acceleration factor
                else:
                    # Continue uptrend - check for new high
                    if high[i] > ep:
                        ep = high[i]  # Update extreme point
                        af = min(af + step, max_step)  # Increase AF
                    # Ensure SAR doesn't go above previous two lows
                    sar[i] = min(sar[i], low[i-1])
                    if i > 1:
                        sar[i] = min(sar[i], low[i-2])
            else:  # Currently in downtrend
                # Check if SAR is below the high (reversal signal)
                if sar[i] < high[i]:
                    bull = True  # Switch to uptrend
                    sar[i] = ep  # SAR becomes the previous extreme point
                    ep = high[i]  # New extreme point is current high
                    af = step  # Reset acceleration factor
                else:
                    # Continue downtrend - check for new low
                    if low[i] < ep:
                        ep = low[i]  # Update extreme point
                        af = min(af + step, max_step)  # Increase AF
                    # Ensure SAR doesn't go below previous two highs
                    sar[i] = max(sar[i], high[i-1])
                    if i > 1:
                        sar[i] = max(sar[i], high[i-2])

        return pd.Series(sar, index=index)

    @staticmethod
    def dmd_prediction(series, modes=2):
        # Simplified DMD - just return a simple trend for performance
        return series.rolling(10).mean()

    @staticmethod
    def ssa_trend(series, window_size=20):
        # Simplified SSA - just return a smoothed trend for performance
        return series.rolling(window_size).mean()

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
        try:
            return ant.perm_entropy(x, order=3, normalize=True)
        except:
            return np.nan

    @staticmethod
    def fractal(x):
        # x is a 1-D NumPy array
        # choose m (embedding dim) and r (tolerance), e.g. m=2, r=0.2*std
        try:
            return ant.higuchi_fd(x)
        except:
            return np.nan

    @staticmethod
    def sampen_window(x):
        # x is a 1-D NumPy array
        try:
            return ant.sample_entropy(x, order=2, metric='chebyshev')
        except:
            return np.nan

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

    # ——— NEW TECHNICAL INDICATORS FOR TFT ———

    @staticmethod
    def log_returns(series, periods=[1, 2, 3]):
        """Calculate log returns for multiple periods"""
        results = {}
        for period in periods:
            results[f'log_ret_{period}'] = np.log(series / series.shift(period))
        return results

    @staticmethod
    def sma(series, period):
        """Simple Moving Average"""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema_crossover_diff(series, fast_period=5, slow_period=10):
        """EMA crossover difference (fast EMA - slow EMA)"""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        return fast_ema - slow_ema

    @staticmethod
    def rsi(series, period=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd_custom(series, fast_period=5, slow_period=13, signal_period=3):
        """MACD with custom parameters"""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line - signal_line  # MACD histogram

    @staticmethod
    def bollinger_band_width(series, period=10, std_dev=2):
        """Bollinger Band Width"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        bb_width = (upper_band - lower_band) / sma
        return bb_width

    @staticmethod
    def realized_volatility(series, period=5):
        """Realized volatility (rolling standard deviation of returns)"""
        returns = series.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        price_change = close.diff()
        obv_values = np.where(price_change > 0, volume, 
                             np.where(price_change < 0, -volume, 0))
        return pd.Series(obv_values, index=close.index).cumsum()

    @staticmethod
    def mfi(high, low, close, volume, period=5):
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        price_change = typical_price.diff()
        positive_flow = np.where(price_change > 0, money_flow, 0)
        negative_flow = np.where(price_change < 0, money_flow, 0)

        positive_mf = pd.Series(positive_flow, index=close.index).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow, index=close.index).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi

    @staticmethod
    def dollar_volume_zscore(close, volume, period=30):
        """Dollar Volume Z-score"""
        dollar_volume = close * volume
        rolling_mean = dollar_volume.rolling(window=period).mean()
        rolling_std = dollar_volume.rolling(window=period).std()
        z_score = (dollar_volume - rolling_mean) / rolling_std
        return z_score

    @staticmethod
    def stochastic_oscillator(high, low, close, k_period=5, d_period=3):
        """Stochastic Oscillator %K and %D"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    @staticmethod
    def adx(high, low, close, period=7):
        """Average Directional Index"""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)

        # Smooth the values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def williams_r(high, low, close, period=7):
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r

    # ——— CROSS-ASSET INDICATORS ———

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

    @staticmethod
    def calculate_returns(series, periods=[1, 3]):
        """Calculate returns for multiple periods"""
        results = {}
        for period in periods:
            results[f'ret_{period}'] = series.pct_change(periods=period)
        return results

    # ——— NEW INDICATORS: SIC CODES AND CALENDAR FEATURES ———

    def get_sic_code_for_ticker(self, ticker):
        """Get SIC code for a ticker using Polygon API with rate limiting and caching"""
        # Check cache first
        if ticker in self.sic_code_cache:
            return self.sic_code_cache[ticker]

        # Rate limiting: ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.api_call_delay:
            sleep_time = self.api_call_delay - time_since_last_call
            time.sleep(sleep_time)

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                self.last_api_call_time = time.time()
                details = self.client.get_ticker_details(ticker)
                sic_code = getattr(details, 'sic_code', None)

                # Cache the result (even if None)
                self.sic_code_cache[ticker] = sic_code
                return sic_code

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limiting error (429)
                if '429' in error_str or 'too many' in error_str or 'rate limit' in error_str:
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit for {ticker}, retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Warning: Rate limit exceeded for {ticker} after {max_retries} attempts: {e}")
                        # Cache the failure to avoid repeated attempts
                        self.sic_code_cache[ticker] = None
                        return None
                else:
                    # For non-rate-limiting errors, don't retry
                    print(f"Warning: Could not fetch SIC code for {ticker}: {e}")
                    # Cache the failure
                    self.sic_code_cache[ticker] = None
                    return None

        # If we get here, all retries failed
        self.sic_code_cache[ticker] = None
        return None

    def create_sector_indicator(self, tickers):
        """Create sector indicator based on SIC codes"""
        sic_to_sector = {
            # Technology
            range(3570, 3580): 'Technology',  # Computer and office equipment
            range(3600, 3700): 'Technology',  # Electronic equipment
            range(7370, 7380): 'Technology',  # Computer programming and data processing

            # Financial Services
            range(6000, 6100): 'Financial',   # Banking
            range(6200, 6300): 'Financial',   # Security and commodity brokers
            range(6300, 6400): 'Financial',   # Insurance carriers
            range(6700, 6800): 'Financial',   # Holding and investment offices

            # Healthcare
            range(2830, 2840): 'Healthcare',  # Drugs
            range(3840, 3850): 'Healthcare',  # Surgical and medical instruments
            range(8000, 8100): 'Healthcare',  # Health services

            # Energy
            range(1300, 1400): 'Energy',      # Oil and gas extraction
            range(2900, 3000): 'Energy',      # Petroleum refining

            # Consumer Discretionary
            range(2300, 2400): 'Consumer_Discretionary',  # Apparel
            range(3700, 3800): 'Consumer_Discretionary',  # Transportation equipment
            range(5000, 5200): 'Consumer_Discretionary',  # Wholesale trade
            range(5200, 5600): 'Consumer_Discretionary',  # Retail trade

            # Consumer Staples
            range(2000, 2100): 'Consumer_Staples',  # Food products
            range(5400, 5500): 'Consumer_Staples',  # Food stores

            # Industrials
            range(1500, 1800): 'Industrials',  # Construction
            range(3300, 3400): 'Industrials',  # Primary metal industries
            range(3400, 3500): 'Industrials',  # Fabricated metal products
            range(3500, 3600): 'Industrials',  # Industrial machinery

            # Materials
            range(1000, 1500): 'Materials',    # Mining
            range(2600, 2700): 'Materials',    # Paper and allied products
            range(2800, 2900): 'Materials',    # Chemicals

            # Utilities
            range(4900, 5000): 'Utilities',    # Electric, gas, and sanitary services

            # Real Estate
            range(6500, 6600): 'Real_Estate',  # Real estate

            # Communication Services
            range(4800, 4900): 'Communication',  # Communications
        }

        ticker_sectors = {}
        for ticker in tickers:
            sic_code = self.get_sic_code_for_ticker(ticker)
            if sic_code:
                try:
                    # Convert SIC code to integer for range comparison
                    sic_code_int = int(sic_code)
                    sector = 'Other'  # Default sector
                    for sic_range, sector_name in sic_to_sector.items():
                        if isinstance(sic_range, range) and sic_code_int in sic_range:
                            sector = sector_name
                            break
                    ticker_sectors[ticker] = sector
                except (ValueError, TypeError):
                    # If SIC code can't be converted to int, mark as Unknown
                    ticker_sectors[ticker] = 'Unknown'
            else:
                ticker_sectors[ticker] = 'Unknown'

        return ticker_sectors

    @staticmethod
    def add_calendar_indicators(df):
        """Add calendar-based indicators"""
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek

        # Day of month (1-31)
        df['day_of_month'] = df.index.day

        # Days to month end
        # Get the last day of each month for each date
        month_ends = df.index.to_period('M').end_time
        current_dates = df.index
        df['days_to_month_end'] = (month_ends - current_dates).days

        return df

    def get_earnings_dates_for_ticker(self, ticker):
        """Get earnings dates for a ticker (placeholder implementation)"""
        # Note: This would require a more comprehensive earnings calendar API
        # For now, we'll create a placeholder that can be replaced with actual earnings data
        try:
            # Placeholder: In a real implementation, you would fetch earnings dates
            # from an earnings calendar API or financial data provider
            # For now, return None to indicate no earnings data available
            return None
        except Exception as e:
            print(f"Warning: Could not fetch earnings dates for {ticker}: {e}")
            return None

    @staticmethod
    def add_earnings_dummy(df, earnings_dates_dict=None, lookforward_days=10):
        """Add earnings dummy indicator (0/1 flag for next 10 trading days)"""
        # Initialize earnings dummy column
        df['earnings_dummy_10d'] = 0

        if earnings_dates_dict:
            for ticker, earnings_dates in earnings_dates_dict.items():
                if earnings_dates:
                    ticker_mask = df['Ticker'] == ticker
                    ticker_df = df[ticker_mask].copy()

                    for earnings_date in earnings_dates:
                        earnings_date = pd.to_datetime(earnings_date)
                        # Find dates within lookforward_days before earnings
                        start_date = earnings_date - pd.Timedelta(days=lookforward_days)
                        end_date = earnings_date

                        # Set dummy to 1 for dates in the range
                        date_mask = (ticker_df.index >= start_date) & (ticker_df.index <= end_date)
                        df.loc[ticker_mask & date_mask, 'earnings_dummy_10d'] = 1

        return df

    # ——— OPTIMIZED Core Indicator Loop ———
    def add_technical_indicators(self):
        self.dataset_ex_df.index = pd.to_datetime(self.dataset_ex_df.index).tz_localize(None)
        df = self.dataset_ex_df.copy()

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

        # Process cross-asset indicators
        cross_asset_indicators = ['spy_ret_1', 'spy_ret_3', 'sector_etf_ret_1', 'vix_delta_1', 'yc_2y10y_delta']
        needed_cross_asset = [ind for ind in cross_asset_indicators if ind in self.indicator_list]

        if needed_cross_asset:
            # SPY returns
            if 'spy_ret_1' in self.indicator_list or 'spy_ret_3' in self.indicator_list:
                spy_data = self.get_spy_data()
                if spy_data is not None:
                    spy_returns = self.calculate_returns(spy_data['close'], [1, 3])
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
                sector_data = self.get_sector_etf_data(['XLK'])  # Technology sector as example
                if 'XLK' in sector_data:
                    xlk_ret_1 = sector_data['XLK']['close'].pct_change().rename('sector_etf_ret_1')
                    df = df.merge(xlk_ret_1.to_frame(), left_index=True, right_index=True, how='left')
                else:
                    # Create placeholder column if sector ETF data fetch fails
                    df['sector_etf_ret_1'] = np.nan

            # VIX delta
            if 'vix_delta_1' in self.indicator_list:
                vix_data = self.get_vix_data()
                if vix_data is not None:
                    vix_delta_1 = vix_data['close'].diff().rename('vix_delta_1')
                    df = df.merge(vix_delta_1.to_frame(), left_index=True, right_index=True, how='left')
                else:
                    # Create placeholder column if VIX data fetch fails
                    df['vix_delta_1'] = np.nan

            # Yield curve (2y-10y) - placeholder implementation
            # Note: This would require FRED API or similar for actual yield data
            if 'yc_2y10y_delta' in self.indicator_list:
                # For now, create a placeholder that can be replaced with actual yield curve data
                df['yc_2y10y_delta'] = 0.0  # Placeholder - replace with actual yield curve data

        # ——— NEW INDICATORS: SIC SECTOR AND CALENDAR FEATURES ———

        # Process SIC sector indicator
        if 'sic_sector' in self.indicator_list:
            unique_tickers = df['Ticker'].unique()
            ticker_sectors = self.create_sector_indicator(unique_tickers)
            df['sic_sector'] = df['Ticker'].map(ticker_sectors)

        # Process calendar indicators
        calendar_indicators = ['day_of_week', 'day_of_month', 'days_to_month_end']
        needed_calendar = [ind for ind in calendar_indicators if ind in self.indicator_list]
        if needed_calendar:
            df = self.add_calendar_indicators(df)

        # Process earnings dummy indicator
        if 'earnings_dummy_10d' in self.indicator_list:
            # Get unique tickers and fetch earnings dates
            unique_tickers = df['Ticker'].unique()
            earnings_dates_dict = {}
            for ticker in unique_tickers:
                earnings_dates = self.get_earnings_dates_for_ticker(ticker)
                if earnings_dates:
                    earnings_dates_dict[ticker] = earnings_dates

            df = self.add_earnings_dummy(df, earnings_dates_dict, lookforward_days=10)

        # VECTORIZED APPROACH: Process all tickers at once using groupby
        grouped = df.groupby('Ticker')

        # Process log returns
        log_ret_indicators = ['log_ret_1', 'log_ret_2', 'log_ret_3']
        needed_log_rets = [ind for ind in log_ret_indicators if ind in self.indicator_list]
        if needed_log_rets:
            for ticker, group in grouped:
                log_ret_results = self.log_returns(group['Close'])
                for indicator in needed_log_rets:
                    if indicator not in df.columns:
                        df[indicator] = np.nan
                    df.loc[group.index, indicator] = log_ret_results[indicator]

        # Process SMA indicators
        if 'sma_5_close' in self.indicator_list:
            df['sma_5_close'] = grouped['Close'].transform(lambda x: self.sma(x, 5))

        # Process EMA crossover
        if 'ema_fast5_slow10' in self.indicator_list:
            df['ema_fast5_slow10'] = grouped['Close'].transform(lambda x: self.ema_crossover_diff(x, 5, 10))

        # Process RSI indicators
        rsi_indicators = ['rsi_3', 'rsi_7']
        for indicator in rsi_indicators:
            if indicator in self.indicator_list:
                period = int(indicator.split('_')[1])
                df[indicator] = grouped['Close'].transform(lambda x: self.rsi(x, period))

        # Process custom MACD
        if 'macd_fast5_slow13' in self.indicator_list:
            df['macd_fast5_slow13'] = grouped['Close'].transform(lambda x: self.macd_custom(x, 5, 13, 3))

        # Process ATR with period 5
        if 'atr_5' in self.indicator_list:
            df['atr_5'] = np.nan
            for ticker, group in grouped:
                atr_values = self.atr(group['Close'], group['High'], group['Low'], 5)
                df.loc[group.index, 'atr_5'] = atr_values

        # Process Bollinger Band Width
        if 'bb_width_10' in self.indicator_list:
            df['bb_width_10'] = grouped['Close'].transform(lambda x: self.bollinger_band_width(x, 10, 2))

        # Process Realized Volatility
        if 'real_vol_5' in self.indicator_list:
            df['real_vol_5'] = grouped['Close'].transform(lambda x: self.realized_volatility(x, 5))

        # Process OBV
        if 'obv' in self.indicator_list:
            df['obv'] = np.nan
            for ticker, group in grouped:
                obv_values = self.obv(group['Close'], group['Volume'])
                df.loc[group.index, 'obv'] = obv_values

        # Process MFI
        if 'mfi_5' in self.indicator_list:
            df['mfi_5'] = np.nan
            for ticker, group in grouped:
                mfi_values = self.mfi(group['High'], group['Low'], group['Close'], group['Volume'], 5)
                df.loc[group.index, 'mfi_5'] = mfi_values

        # Process Dollar Volume Z-score
        if 'dollar_vol_z' in self.indicator_list:
            df['dollar_vol_z'] = np.nan
            for ticker, group in grouped:
                dollar_vol_values = self.dollar_volume_zscore(group['Close'], group['Volume'], 30)
                df.loc[group.index, 'dollar_vol_z'] = dollar_vol_values

        # Process Stochastic Oscillator
        stoch_indicators = ['stoch_k_5', 'stoch_d_5']
        if any(ind in self.indicator_list for ind in stoch_indicators):
            if 'stoch_k_5' in self.indicator_list:
                df['stoch_k_5'] = np.nan
            if 'stoch_d_5' in self.indicator_list:
                df['stoch_d_5'] = np.nan
            for ticker, group in grouped:
                k_values, d_values = self.stochastic_oscillator(group['High'], group['Low'], group['Close'], 5, 3)
                if 'stoch_k_5' in self.indicator_list:
                    df.loc[group.index, 'stoch_k_5'] = k_values
                if 'stoch_d_5' in self.indicator_list:
                    df.loc[group.index, 'stoch_d_5'] = d_values

        # Process ADX
        if 'adx_7' in self.indicator_list:
            df['adx_7'] = np.nan
            for ticker, group in grouped:
                adx_values = self.adx(group['High'], group['Low'], group['Close'], 7)
                df.loc[group.index, 'adx_7'] = adx_values

        # Process Williams %R
        if 'williams_r_7' in self.indicator_list:
            df['williams_r_7'] = np.nan
            for ticker, group in grouped:
                williams_values = self.williams_r(group['High'], group['Low'], group['Close'], 7)
                df.loc[group.index, 'williams_r_7'] = williams_values

        # Process EMA indicators using vectorized operations
        for period in (20, 50, 100, 200):
            name = f'ema_{period}'
            if name in self.indicator_list:
                df[name] = grouped['Close'].transform(lambda x: self.ema(x, period))

        # Process Stochastic RSI indicators
        for period in (14, 28):
            name = f'stoch_rsi{period}'
            if name in self.indicator_list:
                df[name] = grouped['Close'].transform(lambda x: self.stochastic_rsi(x, period))

        # Process single-series indicators
        if 'b_percent' in self.indicator_list:
            df['b_percent'] = grouped['Close'].transform(self.bollinger_percent_b)

        if 'macd' in self.indicator_list:
            df['macd'] = grouped['Close'].transform(self.macd)

        if 'ssa_trend' in self.indicator_list:
            df['ssa_trend'] = grouped['Close'].transform(self.ssa_trend)

        if 'dmd' in self.indicator_list:
            df['dmd'] = grouped['Close'].transform(self.dmd_prediction)

        if 'close_denoised_L1' in self.indicator_list:
            df['close_denoised_L1'] = grouped['Close'].transform(self.wavelet_denoise)

        # Process multi-column indicators
        if 'cmf' in self.indicator_list:
            cmf_results = []
            for ticker, group in grouped:
                cmf_values = self.compute_cmf(group[['High', 'Low', 'Close', 'Volume']])
                cmf_results.append(cmf_values)
            df['cmf'] = pd.concat(cmf_results)

        if 'cci' in self.indicator_list:
            cci_results = []
            for ticker, group in grouped:
                cci_values = self.compute_cci(group[['High', 'Low', 'Close', 'Volume']])
                cci_results.append(cci_values)
            df['cci'] = pd.concat(cci_results)

        if 'parabolic_sar' in self.indicator_list:
            sar_results = []
            for ticker, group in grouped:
                sar_values = self.compute_parabolic_sar(group[['High', 'Low', 'Close']])
                sar_results.append(sar_values)
            df['parabolic_sar'] = pd.concat(sar_results)

        # Process multi-output indicators
        if {'keltner_upper', 'keltner_lower'} & set(self.indicator_list):
            keltner_upper_results = []
            keltner_lower_results = []
            for ticker, group in grouped:
                upper, lower = self.keltner_channel(group['High'], group['Low'], group['Close'])
                keltner_upper_results.append(upper)
                keltner_lower_results.append(lower)
            if 'keltner_upper' in self.indicator_list:
                df['keltner_upper'] = pd.concat(keltner_upper_results)
            if 'keltner_lower' in self.indicator_list:
                df['keltner_lower'] = pd.concat(keltner_lower_results)

        if {'price_momentum', 'volume_momentum'} & set(self.indicator_list):
            price_momentum_results = []
            volume_momentum_results = []
            for ticker, group in grouped:
                price_mom, volume_mom = self.momentum_signals(group['Close'], group['Volume'])
                price_momentum_results.append(price_mom)
                volume_momentum_results.append(volume_mom)
            if 'price_momentum' in self.indicator_list:
                df['price_momentum'] = pd.concat(price_momentum_results)
            if 'volume_momentum' in self.indicator_list:
                df['volume_momentum'] = pd.concat(volume_momentum_results)

        if {'bullish_engulfing', 'bearish_engulfing'} & set(self.indicator_list):
            bullish_results = []
            bearish_results = []
            for ticker, group in grouped:
                bullish, bearish = self.engulfing_patterns(group['Open'], group['Close'])
                bullish_results.append(bullish)
                bearish_results.append(bearish)
            if 'bullish_engulfing' in self.indicator_list:
                df['bullish_engulfing'] = pd.concat(bullish_results)
            if 'bearish_engulfing' in self.indicator_list:
                df['bearish_engulfing'] = pd.concat(bearish_results)

        # OPTIMIZED rolling window calculations using pandas rolling with apply
        # These are the most expensive operations, so we optimize them heavily
        expensive_indicators = ['hurst_100', 'perm_entropy_50', 'sampen_50', 'fractal_50']
        needed_expensive = [ind for ind in expensive_indicators if ind in self.indicator_list]

        if needed_expensive:
            # Process expensive indicators with optimized rolling operations
            for indicator in needed_expensive:
                if indicator == 'hurst_100':
                    window_size = 100
                    func = self.hurst_window
                else:
                    window_size = 50
                    if indicator == 'perm_entropy_50':
                        func = self.perm_entropy_window
                    elif indicator == 'fractal_50':
                        func = self.fractal
                    elif indicator == 'sampen_50':
                        func = self.sampen_window

                # Use pandas rolling with apply - much faster than nested loops
                df[indicator] = grouped['Close'].transform(
                    lambda x: x.rolling(window=window_size, min_periods=window_size).apply(func, raw=True)
                )

        self.dataset_ex_df = df
        return df

    # ——— Final Merge Based on indicators & mode ———
    def merge_data(self):
        cols = ['Ticker']
        if self.prediction_mode:
            cols += list(self.indicator_list)
        else:
            cols += ['shifted_prices'] + list(self.indicator_list)
        self.final_df = self.dataset_ex_df[cols].replace([float('inf'), float('-inf')], float('nan')).dropna()
        return self.final_df

    def process_all(self):
        #self.fetch_stock_data()
        self.preprocess_data()
        self.add_technical_indicators()
        return self.merge_data()
