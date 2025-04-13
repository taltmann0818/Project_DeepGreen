import yfinance as yf
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
    def __init__(self, ticker, years=1, prediction_window=5,**kwargs):
        """
        Initialize the StockAnalyzer with a ticker symbol and number of past days to fetch.
        """
        self.ticker = ticker
        self.stock_data = None
        self.q_income_stmt = None
        self.y_income_stmt = None
        self.earnings_data = None
        self.dataset_ex_df = None
        self.fft_df_real = None
        self.fft_df_imag = None
        self.merged_df = None
        self.prediction_window = -abs(prediction_window)
        self.days = years * 365

        # Kwargs
        self.start_date = kwargs.get('start_date', None)
        self.end_date = kwargs.get('end_date', None)
        self.prediction_mode = kwargs.get('prediction_mode', False)

    def fetch_stock_data(self):
        """
        Fetch historical equity price and financial data using yfinance.
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            if self.days != 0:
                current_date = datetime.today() - timedelta(days=1)
                past_date = current_date - timedelta(days=self.days)
                self.stock_data = ticker_obj.history(start=past_date.strftime("%Y-%m-%d"),
                                                       end=current_date.strftime("%Y-%m-%d"),
                                                       interval="1d")
                
            elif (self.start_date and self.end_date) is not None:
                self.stock_data = ticker_obj.history(start=self.start_date.strftime("%Y-%m-%d"),
                                       end=self.end_date.strftime("%Y-%m-%d"),
                                       interval="1d")
            else:
                raise ValueError("Days must be non-zero or a start_date and end_date provided.")

            #y_income_stmt = ticker_obj.get_income_stmt(freq='yearly').T
            #self.y_income_stmt = y_income_stmt.reset_index().rename(columns={"index": "Date"}).sort_values('Date')
            #earnings_data = ticker_obj.get_earnings_dates()
            #self.earnings_data = earnings_data.reset_index().rename(
            #    columns={"Earnings Date": "Date", "EPS Estimate": "eps_estimate", "Reported EPS": "eps",
            #             "Surprise(%)": "eps_surprise"}).sort_values('Date')

            return self.stock_data #, self.q_income_stmt, self.y_income_stmt, self.earnings_data
        # Handling of delisted stocks
        except AttributeError:
            self.stock_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
            #self.q_income_stmt = pd.DataFrame()
            #self.y_income_stmt = pd.DataFrame()
            #self.earnings_data = pd.DataFrame(index=[0])

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
            #self.dataset_ex_df.set_index('Date', inplace=True)

            # Merge in earnings call data
            #self.dataset_ex_df = pd.merge_asof(self.dataset_ex_df, self.earnings_data, on='Date', direction='backward')

            # Merge in income statment data for ttm eps and pe ratio
            #self.y_income_stmt['Date'] = pd.to_datetime(self.y_income_stmt['Date']).dt.tz_localize('America/New_York')
            #self.y_income_stmt['ttm_eps'] = self.y_income_stmt['NetIncome'] / self.y_income_stmt['BasicAverageShares']
            #self.dataset_ex_df = pd.merge_asof(self.dataset_ex_df, self.y_income_stmt[['Date', 'ttm_eps']], on='Date', direction='backward')
            #self.dataset_ex_df['ttm_pe'] = self.dataset_ex_df['Close'] / self.dataset_ex_df['ttm_eps']

            # Get market regimes
            self.dataset_ex_df = MarketRegimes(self.dataset_ex_df, "hmm_model.pkl").run_regime_detection()

            # Target or outcome variable
            if not self.prediction_mode:
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

    @staticmethod
    def obv(close, volume):
        try:
            obv_values = np.where(close > close.shift(), volume,
                                  np.where(close < close.shift(), -volume, 0))
            return pd.Series(obv_values, index=close.index).cumsum()
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
    def adx(high, low, close, period=14, smoothing=14):
        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=smoothing).mean()

        # Calculate Directional Movement (DM)
        pos_dm_raw = high.diff()
        neg_dm_raw = low.diff()

        # Create the conditions as numpy arrays
        pos_condition = (pos_dm_raw > 0) & (pos_dm_raw > neg_dm_raw.abs())
        neg_condition = (neg_dm_raw < 0) & (neg_dm_raw.abs() > pos_dm_raw)

        # Apply conditions and create Series with the same index
        pos_dm = pd.Series(np.where(pos_condition, pos_dm_raw, 0), index=high.index)
        neg_dm = pd.Series(np.where(neg_condition, neg_dm_raw.abs(), 0), index=low.index)

        # Smooth DM
        pos_di = 100 * (pos_dm.rolling(window=smoothing).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=smoothing).mean() / atr)

        # Calculate Directional Index (DX)
        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))

        # Calculate ADX
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def engulfing_patterns(open, close):
        # Determine if candle is bullish (close > open) or bearish (close < open)
        current_bullish = close > open
        prev_bullish = close.shift(1) > open.shift(1)

        # Real body size (absolute difference between open and close)
        current_body_size = abs(close - open)
        prev_body_size = abs(close.shift(1) - open.shift(1))

        # Bullish engulfing: current candle is bullish, previous is bearish,
        # current candle's body completely engulfs previous candle's body
        bullish_engulfing = (
                current_bullish &  # Current candle is bullish
                ~prev_bullish &  # Previous candle is bearish
                (open <= close.shift(1)) &  # Current open <= Previous close
                (close >= open.shift(1)) &  # Current close >= Previous open
                (current_body_size > prev_body_size)  # Current body larger than previous
        )

        # Bearish engulfing: current candle is bearish, previous is bullish,
        # current candle's body completely engulfs previous candle's body
        bearish_engulfing = (
                ~current_bullish &  # Current candle is bearish
                prev_bullish &  # Previous candle is bullish
                (open >= close.shift(1)) &  # Current open >= Previous close
                (close <= open.shift(1)) &  # Current close <= Previous open
                (current_body_size > prev_body_size)  # Current body larger than previous
        )

        # Convert to binary indicators (0 or 1)
        bullish_engulfing = bullish_engulfing.astype(int)
        bearish_engulfing = bearish_engulfing.astype(int)

        return bullish_engulfing, bearish_engulfing

    def add_technical_indicators(self):
        # Calculate and add technical indicators to the dataset.
        self.dataset_ex_df['ema_20'] = self.ema(self.dataset_ex_df["Close"], 20)
        self.dataset_ex_df['ema_50'] = self.ema(self.dataset_ex_df["Close"], 50)
        self.dataset_ex_df['ema_200'] = self.ema(self.dataset_ex_df["Close"], 200)
        #self.dataset_ex_df['obv'] = self.obv(self.dataset_ex_df["Close"], self.dataset_ex_df["Volume"])
        self.dataset_ex_df['stoch_rsi'] = self.stochastic_rsi(self.dataset_ex_df["Close"])
        self.dataset_ex_df['macd'] = self.macd(self.dataset_ex_df["Close"])
        self.dataset_ex_df['b_percent'] = self.bollinger_percent_b(self.dataset_ex_df["Close"])
        self.dataset_ex_df['keltner_upper'], self.dataset_ex_df['keltner_lower'] = self.keltner_channel(
            self.dataset_ex_df["High"], self.dataset_ex_df["Low"], self.dataset_ex_df["Close"])
        self.dataset_ex_df['adx'] = self.adx(self.dataset_ex_df["High"], self.dataset_ex_df["Low"],self.dataset_ex_df["Close"])

        # Calculate and add candlestick patterns to the dataset.
        #self.dataset_ex_df['bullish_engulfing'], self.dataset_ex_df['bearish_engulfing'] = self.engulfing_patterns(self.dataset_ex_df["Open"], self.dataset_ex_df["Close"])

        return self.dataset_ex_df

    def merge_data(self):
        """
        Merge all data sources into one DataFrame.
        """
        try:
            indicators = ['ema_20', 'ema_50', 'ema_200', 'stoch_rsi', 'macd', 'b_percent', 'keltner_lower', 'keltner_upper','adx', 'Close']
            if self.prediction_mode:
                self.dataset_ex_df = self.dataset_ex_df[['Date','Ticker']+indicators]
                self.final_df = self.dataset_ex_df.dropna()
                self.final_df.set_index('Date', inplace=True)
                return self.final_df
            else:
                self.dataset_ex_df = self.dataset_ex_df[['Date','Ticker','shifted_prices']+indicators]
                self.final_df = self.dataset_ex_df.dropna()
                self.final_df.set_index('Date', inplace=True)
                return self.final_df
        except Exception as e:
            self.final_df = pd.DataFrame()
            print(f"Error while merging data for {self.ticker}; error: {e}")

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