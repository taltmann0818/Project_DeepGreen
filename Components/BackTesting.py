from datetime import datetime, timedelta
import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.pyplot import xlabel
from sympy.physics.units import volume
from vectorbt.portfolio.enums import SizeType, Direction, NoOrder, OrderStatus, OrderSide
import quantstats as qs

class BackTesting:
    def __init__(self,data,ticker,initial_capital,pct_change_entry=0.05,pct_change_exit=0.05,**kwargs):

        #Intialize required and default arguments
        self.ticker = ticker
        self.initial_cash = initial_capital
        self.pct_change_entry = pct_change_entry
        self.pct_change_exit = -abs(pct_change_exit)

        #Kwargs
        self.use_fractional_shares = kwargs.get('use_fractional_shares', True)
        self.slippage = kwargs.get('slippage', 0.001)
        self.transaction_fees = kwargs.get('transaction_fees', 0.000)
        self.use_confidence = kwargs.get('use_confidence', False)

        if type(ticker) == str:
            # Prepare data
            self.data = data[data['Ticker'] == ticker]

            # Ensure the signals are correctly set
            self.data['entry_signal'] = (self.data['Predicted'] - self.data['Close']) / self.data['Close'] >= self.pct_change_entry
            self.data['exit_signal'] = (self.data['Predicted'] - self.data['Close']) / self.data['Close'] <= self.pct_change_exit
        
        else:
            raise ValueError("Invalid Ticker. Please provide a string.")

        if self.use_confidence:
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
            accumulate=True
        )

        return portfolio
        
    @staticmethod
    def BenchMarkSignals(bm_data, short_window=50, long_window=200):

        fast_ma = vbt.MA.run(bm_data['Close'], short_window, short_name='fast_ma')
        slow_ma = vbt.MA.run(bm_data['Close'], long_window, short_name='slow_ma')
        bm_data['entry_signal'] = fast_ma.ma_crossed_above(slow_ma)
        bm_data['exit_signal'] = fast_ma.ma_crossed_below(slow_ma)

        return bm_data
        

    def run_simulation(self):

        print(f"Running vectorbt backtest for {self.ticker}")

        # Run the backtest using vectorbt's Portfolio
        self.pf = self.VectorBTBackTestSignals(self, self.data, self.initial_cash, self.position_size)
        # Run the backtest with a benchmark of a simple moving average and RSI strategy
        #self.benchmark_pf = self.VectorBTBackTestSignals(self, self.BenchMarkSignals(self.data.copy()), self.initial_cash, 1.0)
        
        # Create simplified results df
        #benchmark_return = self.benchmark_pf.stats(metrics=['end_value'])[0]
        end_value, total_return, sharpe_ratio   = self.pf.stats(metrics=['end_value'])[0], self.pf.stats(metrics=['total_return'])[0], self.pf.stats(metrics=['sharpe_ratio'])[0]
        #benchmark_improvement = ((end_value / benchmark_return) / benchmark_return) * 100
        
        results = pd.DataFrame({'Total Portfolio Value': end_value,
                                'Return %': total_return,
                                'Sharpe Ratio': sharpe_ratio}
                                ,index=[0])
        full_results = self.pf.stats()


        return results, full_results
        
    def plot_performance(self):
        vbt.settings.set_theme("dark")

        # Plot OHLCV - Order PnL
        trades_fig = self.data[["Open", "High", "Low", "Close","Volume"]].vbt.ohlcv.plot(xaxis=dict(rangeslider=dict(visible=False)))
        self.pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=trades_fig)
        trades_fig.update_layout(yaxis=dict(title="Price (USD)"))

        # Plot Portfolio Value
        value_fig = self.pf.plot_value(trace_kwargs=dict(name="Strategy",line=dict(color='blue')))
        self.benchmark_pf.plot_value(trace_kwargs=dict(name="Benchmark",line=dict(color='orange')), fig=value_fig)
        value_fig.update_layout(yaxis=dict(title="Portfolio Value (USD)"))

        # Plot Exposure Value
        exposure_fig = self.pf.plot_net_exposure(trace_kwargs=dict(name="Strategy",line=dict(color='blue')))
        self.benchmark_pf.plot_net_exposure(trace_kwargs=dict(name="Benchmark", line=dict(color='orange')), fig=exposure_fig)
        exposure_fig.update_layout(yaxis=dict(title="Exposure %",range=[0.0, 1.0]))

        gauge = vbt.plotting.Gauge(
            value=2,
            value_range=(1, 3),
            label='My Gauge'
        ).fig

        return trades_fig, value_fig, exposure_fig
