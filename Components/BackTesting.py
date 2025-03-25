from datetime import datetime, timedelta
import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.graph_objects as go
from vectorbt.portfolio.enums import SizeType, Direction, NoOrder, OrderStatus, OrderSide

class BackTesting:
    def __init__(self, data, ticker, initial_capital, slippage=0.001, transaction_fees=0.000, use_confidence=True, use_fractional_shares=True):

        #Intialize arguments
        self.ticker = ticker
        self.initial_cash = initial_capital
        self.slippage = slippage
        self.transaction_fees = transaction_fees
        self.use_fractional_shares = use_fractional_shares

        if type(ticker) == str:
            # Prepare data
            self.data = data[data['Ticker'] == ticker]
        else:
            raise ValueError("Invalid Ticker. Please provide a string.")

        if use_confidence:
            self.position_size = self.data['Confidence']
        else:
            self.position_size = 1.0
            
    @staticmethod
    def VectorBTBackTestSignals(self, bt_data, initial_cash, size):
        
        portfolio = vbt.Portfolio.from_signals(
            bt_data['Close'],
            entries=bt_data['entry_signal'],
            exits=bt_data['exit_signal'],
            init_cash=initial_cash,
            freq='1D',
            fees=self.transaction_fees,
            slippage=self.slippage,
            allow_partial=self.use_fractional_shares,
            size=size,
            size_type=SizeType.Percent,
        )

        return portfolio
        
    @staticmethod
    def BenchMarkSignals(bm_data, short_window=25, long_window=50, rsi_window=10, rsi_type='simple'):

        bm_data["short_ma"] = bm_data["Close"].rolling(short_window).mean()
        bm_data["long_ma"] = bm_data["Close"].rolling(long_window).mean()

        # Calculate RSI
        close_delta = bm_data["Close"].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if rsi_type == "simple":
            ma_up = up.rolling(window=rsi_window).mean()
            ma_down = down.rolling(window=rsi_window).mean()
        elif rsi_type == "exponential":
            ma_up = up.ewm(span=rsi_window).mean()
            ma_down = down.ewm(span=rsi_window).mean()
        else:
            raise ValueError("Invalid RSI type")

        rs = ma_up / ma_down
        bm_data["rsi"] = 100 - (100 / (1 + rs))

        # Calculate EMA200
        bm_data["ema200"] = bm_data["Close"].ewm(span=200).mean()

        # Generate signals
        bm_data["signal"] = 0
        bm_data.loc[(bm_data["short_ma"] > bm_data["long_ma"]) & (bm_data["Close"] > bm_data["ema200"]) & (bm_data["rsi"] < 30), "signal"] = 1
        bm_data.loc[(bm_data["short_ma"] < bm_data["long_ma"]) | (bm_data["rsi"] > 70), "signal"] = -1

        bm_data['entry_signal'] = bm_data['signal'] == 1  # Buy signal
        bm_data['exit_signal'] = bm_data['signal'] == -1  # Sell signal

        return bm_data
        

    def run_simulation(self):

        print(f"Running vectorbt backtest for {self.ticker}")

        # Run the backtest using vectorbt's Portfolio
        self.pf = self.VectorBTBackTestSignals(self, self.data, self.initial_cash, self.position_size)
        # Run the backtest with a benchmark of a simple moving average and RSI strategy
        self.benchmark_pf = self.VectorBTBackTestSignals(self, self.BenchMarkSignals(self.data.copy()),
                                                         self.initial_cash, 1.0)
        
        # Create simplified results df
        benchmark_return = self.pf.stats(metrics=['end_value'])[0]
        end_value, total_return, sharpe_ratio   = self.pf.stats(metrics=['end_value'])[0], self.pf.stats(metrics=['total_return'])[0], self.pf.stats(metrics=['sharpe_ratio'])[0]
        benchmark_improvement = ((end_value / benchmark_return) / benchmark_return) * 100
        
        results = pd.DataFrame({'Total Portfolio Value': end_value,
                                'Return %': total_return,
                                'Sharpe Ratio': sharpe_ratio,
                                'Benchmark Portfolio Value': benchmark_return,
                                'Improvement Over Benchmark': benchmark_improvement}
                                ,index=[0])
        full_results = self.pf.stats()


        return results, full_results

    def plot_performance(self):
        fig = self.data[["Open", "High", "Low", "Close"]].vbt.ohlcv.plot()
        self.pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)
        fig.update_xaxes(
            range=[self.data['Date'].min(), self.data['Date'].max()],  # Set range to min and max dates in your data
            tickformat='%YYYY-%mm-%dd',  # Adjust tick format as needed
        )

        gauge = vbt.plotting.Gauge(
            value=2,
            value_range=(1, 3),
            label='My Gauge'
        )

        return fig, gauge.fig