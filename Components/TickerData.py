import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TickerData:
    def __init__(self, ticker, years=1, prediction_window=5):
        """
        Initialize the StockAnalyzer with a ticker symbol and number of past days to fetch.
        """
        self.ticker = ticker
        self.days = years * 365
        self.stock_data = None
        self.q_income_stmt = None
        self.y_income_stmt = None
        self.earnings_data = None
        self.dataset_ex_df = None
        self.fft_df_real = None
        self.fft_df_imag = None
        self.merged_df = None
        self.prediction_window = -abs(prediction_window)

    def fetch_stock_data(self):
        """
        Fetch historical equity price and financial data using yfinance.
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            current_date = datetime.today() - timedelta(days=1)
            past_date = current_date - timedelta(days=self.days)
            self.stock_data = ticker_obj.history(start=past_date.strftime("%Y-%m-%d"),
                                                   end=current_date.strftime("%Y-%m-%d"),
                                                   interval="1d")

            y_income_stmt = ticker_obj.get_income_stmt(freq='yearly').T
            self.y_income_stmt = y_income_stmt.reset_index().rename(columns={"index": "Date"}).sort_values('Date')
            earnings_data = ticker_obj.get_earnings_dates()
            self.earnings_data = earnings_data.reset_index().rename(
                columns={"Earnings Date": "Date", "EPS Estimate": "eps_estimate", "Reported EPS": "eps",
                         "Surprise(%)": "eps_surprise"}).sort_values('Date')

            return self.stock_data, self.q_income_stmt, self.y_income_stmt, self.earnings_data
        # Handling of delisted stocks
        except AttributeError:
            self.stock_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            self.q_income_stmt = pd.DataFrame()
            self.y_income_stmt = pd.DataFrame()
            self.earnings_data = pd.DataFrame(index=[0])

            # Log this error
            print(f"AttributeError: Could not fetch data for ticker {self.ticker}, returning empty dataframes")

            return

    def preprocess_data(self):
        """
        Preprocess the fetched stock data:
          - Reset index and convert dates
          - Merge in earnings and income statement data from yfinance
          - Define the target variable
        """
        try:
            # Add ticker name
            self.stock_data['Ticker'] = self.ticker
            self.dataset_ex_df = self.stock_data.copy().reset_index()
            self.dataset_ex_df['Date'] = pd.to_datetime(self.dataset_ex_df['Date'])
            self.dataset_ex_df = self.dataset_ex_df.sort_values('Date')
            self.dataset_ex_df.set_index('Date', inplace=True)

            # Merge in earnings call data
            self.dataset_ex_df = pd.merge_asof(self.dataset_ex_df, self.earnings_data, on='Date', direction='backward')

            # Merge in income statment data for ttm eps and pe ratio
            self.y_income_stmt['Date'] = pd.to_datetime(self.y_income_stmt['Date']).dt.tz_localize('America/New_York')
            self.y_income_stmt['ttm_eps'] = self.y_income_stmt['NetIncome'] / self.y_income_stmt['BasicAverageShares']
            self.dataset_ex_df = pd.merge_asof(self.dataset_ex_df, self.y_income_stmt[['Date', 'ttm_eps']], on='Date', direction='backward')
            self.dataset_ex_df['ttm_pe'] = self.dataset_ex_df['Close'] / self.dataset_ex_df['ttm_eps']

            self.dataset_ex_df['shifted_prices'] = self.dataset_ex_df['Close'].shift(self.prediction_window)

            return self.stock_data, self.dataset_ex_df
            
        except Exception:
            print(f"Error while processing the data for {self.ticker}")
            pass

    @staticmethod
    def ema(close, period=20):
        try:
            return close.ewm(span=period, adjust=False).mean()
        except Exception:
                return np.nan

    @staticmethod
    def stochastic_rsi(close, rsi_period=14, stoch_period=14):
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(rsi_period).mean()
            avg_loss = loss.rolling(rsi_period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            stoch_rsi = (rsi - rsi.rolling(stoch_period).min()) / (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
            return stoch_rsi
        except Exception:
            return np.nan

    @staticmethod
    def macd(close, fast_period=12, slow_period=26, signal_period=9):
        try:
            fast_ema = close.ewm(span=fast_period, adjust=False).mean()
            slow_ema = close.ewm(span=slow_period, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            return macd_line
        except Exception:
            return np.nan

    def obv(self, close, volume):
        try:
            obv_values = np.where(close > close.shift(), volume,
                                  np.where(close < close.shift(), -volume, 0))
            return pd.Series(obv_values, index=close.index).cumsum()
        except Exception:
            return np.nan

    @staticmethod
    def aroon(close, period=25):
        try:
            aroon_up = close.rolling(window=period).apply(lambda x: x.argmax()).add(1).mul(100.0).div(period)
            aroon_down = close.rolling(window=period).apply(lambda x: x.argmin()).add(1).mul(100.0).div(period)
            return aroon_up, aroon_down
        except Exception:
            return np.nan, np.nan

    @staticmethod
    def ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26, senkou_period=52):
        try:
            tenkan_sen = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
            kijun_sen = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
            senkou_span_b = ((high.rolling(senkou_period).max() + low.rolling(senkou_period).min()) / 2).shift(kijun_period)
            chikou_span = close.shift(-kijun_period)
            return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
    @staticmethod
    def relative_vigor_index(high, low, close, open, period=14):
        try:
            rvi = (close - open).rolling(period).mean() / (high - low).rolling(period).mean()
            return rvi
        except Exception:
            return np.nan
        
    @staticmethod
    def klinger_oscillator(high, low, close, volume, short_period=34, long_period=55):
        try:
            trend = np.where(high + low + close > high.shift(1) + low.shift(1) + close.shift(1), 1, -1)
            dm = high - low
            cm = dm.rolling(2).apply(lambda x: x[0] + x[1] if trend[-1] == trend[-2] else dm[-1], raw=True)
            vf = volume * abs(dm / cm) * trend * 100
            kvo = vf.ewm(span=short_period).mean() - vf.ewm(span=long_period).mean()
            return kvo
        except Exception:
            return np.nan
        
    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3):
        try:
            atr = (high - low).rolling(period).mean()
            upper_band = (high + low) / 2 + multiplier * atr
            lower_band = (high + low) / 2 - multiplier * atr
            supertrend = [0.0] * len(high)

            for i in range(1, len(high)):
                if close.iloc[i] > upper_band.iloc[i-1]:
                    supertrend[i] = lower_band.iloc[i]
                elif close.iloc[i] < lower_band.iloc[i-1]:
                    supertrend[i] = upper_band.iloc[i]
                else:
                    supertrend[i] = supertrend[i-1]
            return supertrend
        except Exception:
            return np.nan

    @staticmethod     
    def bollinger_percent_b(close, period=20, std_dev=2):
        try:
            sma = close.rolling(period).mean()
            rolling_std = close.rolling(period).std()
            upper_band = sma + std_dev * rolling_std
            lower_band = sma - std_dev * rolling_std
            b_percent = (close - lower_band) / (upper_band - lower_band)
            return b_percent
        except Exception:
            return np.nan
        
    @staticmethod    
    def elder_ray(high, low, close, ema_period=13):
        try:
            ema = close.ewm(span=ema_period).mean()
            bull_power = high - ema
            bear_power = low - ema
            return bull_power, bear_power
        except Exception:
            return np.nan, np.nan
        
    @staticmethod   
    def market_facilitation_index(high, low, volume):
        try:
            mfi = (high - low) / volume.replace(0, 1e-8)  # Avoid division by zero
            return mfi
        except Exception:
            return np.nan
        
    @staticmethod
    def keltner_channel(high, low, close, ema_period=20, atr_period=10, multiplier=2):
        try:
            ema = close.ewm(span=ema_period).mean()
            atr = (high - low).rolling(atr_period).mean()
            keltner_upper = ema + multiplier * atr
            keltner_lower = ema - multiplier * atr
            return keltner_upper, keltner_lower
        except Exception:
            return np.nan, np.nan
        
    @staticmethod
    def trix(close, period=15):
        try:
            ema1 = close.ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            ema3 = ema2.ewm(span=period).mean()
            trix = ema3.pct_change() * 100
            return trix
        except Exception:
            return np.nan
        
    @staticmethod
    def vortex(high, low, period=14):
        try:
            vm_plus = abs(high - low.shift(1))
            vm_minus = abs(low - high.shift(1))
            tr = (high - low).abs()
            vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
            vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
            return vi_plus, vi_minus
        except Exception:
            return np.nan, np.nan
        
    @staticmethod
    def gri(high, low, close, open):
        try:
            intraday_range = high - low
            interday_volatility = abs(close - open)
            gri = intraday_range / (interday_volatility + 1e-8)  # Avoid division by zero
            return gri
        except Exception:
            return np.nan

    def add_technical_indicators(self):
        # Calculate and add technical indicators to the dataset.
        self.dataset_ex_df['ema_20'] = self.ema(self.dataset_ex_df["Close"], 20)
        self.dataset_ex_df['ema_50'] = self.ema(self.dataset_ex_df["Close"], 50)
        self.dataset_ex_df['ema_100'] = self.ema(self.dataset_ex_df["Close"], 100)

        self.dataset_ex_df['stoch_rsi'] = self.stochastic_rsi(self.dataset_ex_df["Close"])
        self.dataset_ex_df['macd'] = self.macd(self.dataset_ex_df["Close"])

        #self.dataset_ex_df['obv'] = self.obv(self.dataset_ex_df["Close"], self.stock_data["Volume"])
        #self.dataset_ex_df['aroon_up'], self.dataset_ex_df['aroon_down'] = self.aroon(self.dataset_ex_df["Close"], period=25)
        #self.dataset_ex_df['tenkan_sen'], self.dataset_ex_df['kijun_sen'], self.dataset_ex_df['senkou_span_a'], self.dataset_ex_df['senkou_span_b'], self.dataset_ex_df['chikou_span'] = self.ichimoku_cloud(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"])
        #self.dataset_ex_df['rvi'] = self.relative_vigor_index(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"],self.dataset_ex_df["Open"])
        #self.dataset_ex_df['kvo'] = self.klinger_oscillator(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"],self.dataset_ex_df["Volume"])
        #self.dataset_ex_df['supertrend'] = self.supertrend(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"])
        #self.dataset_ex_df['b_percent'] = self.bollinger_percent_b(self.dataset_ex_df["Close"])
        #self.dataset_ex_df['bull_power'], self.dataset_ex_df['bear_power']  = self.elder_ray(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"])
        #self.dataset_ex_df['mfi'] = self.market_facilitation_index(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Volume"])
        #self.dataset_ex_df['keltner_upper'], self.dataset_ex_df['keltner_lower']  = self.keltner_channel(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"])
        #self.dataset_ex_df['trix'] = self.trix(self.dataset_ex_df["Close"])
        #self.dataset_ex_df['vortex_plus'], self.dataset_ex_df['vortex_minus']  = self.vortex(self.dataset_ex_df["High"],self.dataset_ex_df["Low"])
        #self.dataset_ex_df['gri'] = self.gri(self.dataset_ex_df["High"],self.dataset_ex_df["Low"],self.dataset_ex_df["Close"],self.dataset_ex_df["Open"])
        
        return self.dataset_ex_df

    def merge_data(self):
        """
        Merge Fourier transform and technical indicator data into one DataFrame.
        """
        try:

            technical_indicators_df = self.dataset_ex_df[['Date','Ticker','ema_20', 'ema_50', 'ema_100', 'stoch_rsi', 'macd','Close', 'shifted_prices']]
            self.final_df = technical_indicators_df.dropna()
            self.final_df.set_index('Date', inplace=True)
            return self.final_df
        except Exception:
            self.final_df = pd.DataFrame()

    def process_all(self):
        """
        Run the full processing pipeline:
          1. Fetch stock data
          2. Preprocess data
          3. [add the fundemental data]
          4. Add technical indicators
          5. Merge and return the final DataFrame
        """
        self.fetch_stock_data()
        self.preprocess_data()
        self.add_technical_indicators()
        return self.merge_data(), self.stock_data

