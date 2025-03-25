from datetime import datetime, timedelta
import vectorbt as vbt
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.pyplot import xlabel
from vectorbt.portfolio.enums import SizeType, Direction, NoOrder, OrderStatus, OrderSide
import quantstats as qt

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
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.set_index('Date')
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
        sim_fig = make_subplots(rows=3, cols=1, subplot_titles=('Order PnL', 'Portfolio Value','Net Exposure'))
        
        # Plot OHLCV - Order PnL
        self.data[["Open", "High", "Low", "Close","Volume"]].vbt.ohlcv.plot(ohlc_add_trace_kwargs=dict(row=1, col=1), fig=sim_fig,show_volume=True)
        self.pf.positions.plot(close_trace_kwargs=dict(visible=False), add_trace_kwargs=dict(row=1, col=1), fig=sim_fig)
        
        # Plot Portfolio Value
        self.pf.plot_value(trace_kwargs=dict(name='Strategy',color='blue',row=2,col=1), fig=sim_fig, free=True)
        self.benchmark_pf.plot_value(trace_kwargs=dict(name='Benchmark',color='blue',row=2,col=1), fig=sim_fig, free=True)

        # Plot Exposure Value
        self.pf.plot_net_exposure(trace_names='Strategy',add_trace_kwargs=dict(color='blue',row=3,col=1),fig=sim_fig)
        self.benchmark_pf.plot_net_exposure(trace_names='Benchmark',add_trace_kwargs=dict(color='yellow',row=3,col=1), fig=sim_fig)
        

        sim_fig.update_layout(xaxis_title='Date',
                          yaxis_title='Price (USD)',
                          #template='plotly_white',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02)
                         )
        sim_fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        sim_fig.update_yaxes(title_text="Portfolio Value (USD)", row=2, col=1)
        sim_fig.update_yaxes(title_text="Exposure %", row=3, col=1)

        gauge = vbt.plotting.Gauge(
            value=2,
            value_range=(1, 3),
            label='My Gauge'
        ).fig

        return sim_fig
