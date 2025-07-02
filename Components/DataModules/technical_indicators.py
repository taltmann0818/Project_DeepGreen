"""
Technical indicators module for TickerData.
Contains basic technical analysis indicators and calculations.
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Collection of technical analysis indicators"""

    @staticmethod
    def ema(series, period):
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def trend(series, period1=8, period2=21, period3=55):
        """Trend indicator using multiple EMAs"""
        ema1 = TechnicalIndicators.ema(series, period1)
        ema2 = TechnicalIndicators.ema(series, period2)
        ema3 = TechnicalIndicators.ema(series, period3)
        return (ema1 > ema2).astype(int) + (ema2 > ema3).astype(int)

    @staticmethod
    def stochastic_rsi(series, rsi_period=14, stoch_period=14):
        """Stochastic RSI"""
        rsi = TechnicalIndicators.rsi(series, rsi_period)
        min_rsi = rsi.rolling(window=stoch_period).min()
        max_rsi = rsi.rolling(window=stoch_period).max()
        stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
        return stoch_rsi

    @staticmethod
    def macd(series, fast_period=12, slow_period=26, signal_period=9):
        """MACD indicator"""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def compute_cmf(data, period=20):
        """Chaikin Money Flow"""
        high, low, close, volume = data['High'], data['Low'], data['Close'], data['Volume']
        mfv = ((close - low) - (high - close)) / (high - low) * volume
        mfv = mfv.fillna(0)  # Handle division by zero
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf

    @staticmethod
    def compute_cci(data, period=20):
        """Commodity Channel Index"""
        high, low, close = data['High'], data['Low'], data['Close']
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def momentum_signals(close, volume):
        """Price and volume momentum signals"""
        price_momentum = close.pct_change(5)
        volume_momentum = volume.pct_change(5)
        return price_momentum, volume_momentum

    @staticmethod
    def bollinger_percent_b(series, period=20, std_dev=2):
        """Bollinger %B"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        percent_b = (series - lower_band) / (upper_band - lower_band)
        return percent_b

    @staticmethod
    def keltner_channel(high, low, close, ema_period=20, atr_period=10, multiplier=2):
        """Keltner Channel"""
        ema = close.ewm(span=ema_period, adjust=False).mean()
        atr = TechnicalIndicators.atr(close, high, low, atr_period)
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        return upper, lower

    @staticmethod
    def compute_parabolic_sar(data, step=0.02, max_step=0.2):
        """Parabolic SAR"""
        high, low, close = data['High'], data['Low'], data['Close']
        length = len(close)
        sar = np.zeros(length)
        trend = np.zeros(length)
        af = np.zeros(length)
        ep = np.zeros(length)

        # Initialize
        sar[0] = low.iloc[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        af[0] = step
        ep[0] = high.iloc[0]

        index = close.index

        for i in range(1, length):
            # Calculate SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

            # Determine trend
            if trend[i-1] == 1:  # Uptrend
                if low.iloc[i] <= sar[i]:
                    trend[i] = -1  # Switch to downtrend
                    sar[i] = ep[i-1]  # SAR becomes the previous EP
                    af[i] = step  # Reset AF
                    ep[i] = low.iloc[i]  # New EP is current low
                else:
                    trend[i] = 1  # Continue uptrend
                    af[i] = af[i-1]
                    ep[i] = ep[i-1]
                    if high.iloc[i] > ep[i]:
                        ep[i] = high.iloc[i]  # Update extreme point
                        af[i] = min(af[i] + step, max_step)  # Increase AF
                    # Ensure SAR doesn't go above previous two lows
                    sar[i] = min(sar[i], low.iloc[i-1])
                    if i > 1:
                        sar[i] = min(sar[i], low.iloc[i-2])
            else:  # Downtrend
                if high.iloc[i] >= sar[i]:
                    trend[i] = 1  # Switch to uptrend
                    sar[i] = ep[i-1]  # SAR becomes the previous EP
                    af[i] = step  # Reset AF
                    ep[i] = high.iloc[i]  # New EP is current high
                else:
                    trend[i] = -1  # Continue downtrend
                    af[i] = af[i-1]
                    ep[i] = ep[i-1]
                    if low.iloc[i] < ep[i]:
                        ep[i] = low.iloc[i]  # Update extreme point
                        af[i] = min(af[i] + step, max_step)  # Increase AF
                    # Ensure SAR doesn't go below previous two highs
                    sar[i] = max(sar[i], high.iloc[i-1])
                    if i > 1:
                        sar[i] = max(sar[i], high.iloc[i-2])

        return pd.Series(sar, index=index)

    @staticmethod
    def engulfing_patterns(open_, close):
        """Bullish and bearish engulfing patterns"""
        cur_b = close > open_
        prev_b = close.shift() > open_.shift()
        cur_body = (close - open_).abs()
        prev_body = (close.shift() - open_.shift()).abs()
        bull = ( cur_b & ~prev_b & (open_ <= close.shift()) & (close >= open_.shift()) & (cur_body > prev_body) )
        bear = (~cur_b &  prev_b & (open_ >= close.shift()) & (close <= open_.shift()) & (cur_body > prev_body) )
        return bull.astype(int), bear.astype(int)

    @staticmethod
    def atr(close, high, low, period=14):
        """Average True Range"""
        true_range = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

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

    @staticmethod
    def calculate_returns(series, periods=[1, 3]):
        """Calculate returns for multiple periods"""
        results = {}
        for period in periods:
            results[f'ret_{period}'] = series.pct_change(periods=period)
        return results

    @staticmethod
    def add_technical_indicators(df, grouped, indicator_list, nasdaq_data=None):
        """
        Add technical indicators to the dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            The main dataframe to add indicators to
        grouped : pd.DataFrameGroupBy
            Grouped dataframe by ticker
        indicator_list : set
            Set of indicators to calculate
        nasdaq_data : pd.DataFrame, optional
            NASDAQ data for cross-market indicators

        Returns:
        --------
        pd.DataFrame
            DataFrame with technical indicators added
        """
        # Process NASDAQ data once if needed
        if nasdaq_data is not None and {'nasdaq_rsi', 'nasdaq_returns'} & indicator_list:
            nasdaq_data.index = pd.to_datetime(nasdaq_data.index).tz_localize(None)
            if 'nasdaq_rsi' in indicator_list:
                nasdaq_rsi = TechnicalIndicators.stochastic_rsi(nasdaq_data['close']).rename('nasdaq_rsi')
                df = df.merge(nasdaq_rsi.to_frame(), left_index=True, right_index=True, how='left')
            if 'nasdaq_returns' in indicator_list:
                nasdaq_returns = nasdaq_data['close'].pct_change().rename('nasdaq_returns')
                df = df.merge(nasdaq_returns.to_frame(), left_index=True, right_index=True, how='left')

        # Process single-series indicators that can be applied with transform
        single_series_indicators = {
            'trend': lambda x: TechnicalIndicators.trend(x),
            'b_percent': lambda x: TechnicalIndicators.bollinger_percent_b(x),
            'macd': lambda x: TechnicalIndicators.macd(x),
        }

        for indicator, func in single_series_indicators.items():
            if indicator in indicator_list:
                df[indicator] = grouped['Close'].transform(func)

        # Process parameterized indicators
        parameterized_indicators = [
            ('sma', [5, 10, 20]), ('ema', [5, 10, 20, 50, 100]), ('rsi', [5, 14, 21]),
            ('stoch_rsi', [5, 14, 21])
        ]

        for base_name, periods in parameterized_indicators:
            for period in periods:
                name = f'{base_name}_{period}'
                if name in indicator_list:
                    if base_name == 'sma':
                        df[name] = grouped['Close'].transform(lambda x: TechnicalIndicators.sma(x, period))
                    elif base_name == 'ema':
                        df[name] = grouped['Close'].transform(lambda x: TechnicalIndicators.ema(x, period))
                    elif base_name == 'rsi':
                        df[name] = grouped['Close'].transform(lambda x: TechnicalIndicators.rsi(x, period))
                    elif base_name == 'stoch_rsi':
                        df[name] = grouped['Close'].transform(lambda x: TechnicalIndicators.stochastic_rsi(x, period))

        # Process multi-column indicators
        multi_column_indicators = ['cmf', 'cci', 'parabolic_sar']
        for indicator in multi_column_indicators:
            if indicator in indicator_list:
                results = []
                for ticker, group in grouped:
                    if indicator == 'cmf':
                        values = TechnicalIndicators.compute_cmf(group[['High', 'Low', 'Close', 'Volume']])
                    elif indicator == 'cci':
                        values = TechnicalIndicators.compute_cci(group[['High', 'Low', 'Close', 'Volume']])
                    elif indicator == 'parabolic_sar':
                        values = TechnicalIndicators.compute_parabolic_sar(group[['High', 'Low', 'Close']])
                    results.append(values)
                concatenated = pd.concat(results)
                df[indicator] = concatenated.reindex(df.index)

        # Process multi-output indicators
        if {'keltner_upper', 'keltner_lower'} & indicator_list:
            keltner_upper_results = []
            keltner_lower_results = []
            for ticker, group in grouped:
                upper, lower = TechnicalIndicators.keltner_channel(group['High'], group['Low'], group['Close'])
                keltner_upper_results.append(upper)
                keltner_lower_results.append(lower)
            if 'keltner_upper' in indicator_list:
                concatenated_upper = pd.concat(keltner_upper_results)
                df['keltner_upper'] = concatenated_upper.reindex(df.index)
            if 'keltner_lower' in indicator_list:
                concatenated_lower = pd.concat(keltner_lower_results)
                df['keltner_lower'] = concatenated_lower.reindex(df.index)

        if {'price_momentum', 'volume_momentum'} & indicator_list:
            price_momentum_results = []
            volume_momentum_results = []
            for ticker, group in grouped:
                price_mom, volume_mom = TechnicalIndicators.momentum_signals(group['Close'], group['Volume'])
                price_momentum_results.append(price_mom)
                volume_momentum_results.append(volume_mom)
            if 'price_momentum' in indicator_list:
                concatenated_price = pd.concat(price_momentum_results)
                df['price_momentum'] = concatenated_price.reindex(df.index)
            if 'volume_momentum' in indicator_list:
                concatenated_volume = pd.concat(volume_momentum_results)
                df['volume_momentum'] = concatenated_volume.reindex(df.index)

        if {'bullish_engulfing', 'bearish_engulfing'} & indicator_list:
            bullish_results = []
            bearish_results = []
            for ticker, group in grouped:
                bullish, bearish = TechnicalIndicators.engulfing_patterns(group['Open'], group['Close'])
                bullish_results.append(bullish)
                bearish_results.append(bearish)
            if 'bullish_engulfing' in indicator_list:
                concatenated_bullish = pd.concat(bullish_results)
                df['bullish_engulfing'] = concatenated_bullish.reindex(df.index)
            if 'bearish_engulfing' in indicator_list:
                concatenated_bearish = pd.concat(bearish_results)
                df['bearish_engulfing'] = concatenated_bearish.reindex(df.index)

        # Process log returns indicators
        if {'log_ret_1', 'log_ret_2', 'log_ret_3'} & indicator_list:
            for period in [1, 2, 3]:
                indicator_name = f'log_ret_{period}'
                if indicator_name in indicator_list:
                    results = []
                    for ticker, group in grouped:
                        log_returns = TechnicalIndicators.log_returns(group['Close'], periods=[period])
                        results.append(log_returns[indicator_name])
                    concatenated = pd.concat(results)
                    df[indicator_name] = concatenated.reindex(df.index)

        # Process specific named indicators with custom naming conventions
        specific_indicators = {
            'sma_5_close': lambda group: TechnicalIndicators.sma(group['Close'], 5),
            'ema_fast5_slow10': lambda group: TechnicalIndicators.ema_crossover_diff(group['Close'], 5, 10),
            'rsi_3': lambda group: TechnicalIndicators.rsi(group['Close'], 3),
            'rsi_7': lambda group: TechnicalIndicators.rsi(group['Close'], 7),
            'macd_fast5_slow13': lambda group: TechnicalIndicators.macd_custom(group['Close'], 5, 13, 3),
            'atr_5': lambda group: TechnicalIndicators.atr(group['Close'], group['High'], group['Low'], 5),
            'bb_width_10': lambda group: TechnicalIndicators.bollinger_band_width(group['Close'], 10),
            'real_vol_5': lambda group: TechnicalIndicators.realized_volatility(group['Close'], 5),
            'obv': lambda group: TechnicalIndicators.obv(group['Close'], group['Volume']),
            'mfi_5': lambda group: TechnicalIndicators.mfi(group['High'], group['Low'], group['Close'], group['Volume'], 5),
            'dollar_vol_z': lambda group: TechnicalIndicators.dollar_volume_zscore(group['Close'], group['Volume'], 30),
            'adx_7': lambda group: TechnicalIndicators.adx(group['High'], group['Low'], group['Close'], 7),
            'williams_r_7': lambda group: TechnicalIndicators.williams_r(group['High'], group['Low'], group['Close'], 7)
        }

        for indicator_name, func in specific_indicators.items():
            if indicator_name in indicator_list:
                results = []
                for ticker, group in grouped:
                    values = func(group)
                    results.append(values)
                concatenated = pd.concat(results)
                df[indicator_name] = concatenated.reindex(df.index)

        # Process stochastic oscillator indicators (stoch_k_5, stoch_d_5)
        if {'stoch_k_5', 'stoch_d_5'} & indicator_list:
            stoch_k_results = []
            stoch_d_results = []
            for ticker, group in grouped:
                k_percent, d_percent = TechnicalIndicators.stochastic_oscillator(
                    group['High'], group['Low'], group['Close'], k_period=5, d_period=3
                )
                stoch_k_results.append(k_percent)
                stoch_d_results.append(d_percent)
            if 'stoch_k_5' in indicator_list:
                concatenated_k = pd.concat(stoch_k_results)
                df['stoch_k_5'] = concatenated_k.reindex(df.index)
            if 'stoch_d_5' in indicator_list:
                concatenated_d = pd.concat(stoch_d_results)
                df['stoch_d_5'] = concatenated_d.reindex(df.index)

        return df
