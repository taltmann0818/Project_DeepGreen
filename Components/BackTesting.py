import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction, NoOrder, OrderStatus, OrderSide
from Agents.portfolio_manager import PortfolioManagerAgent

class BackTesting:
    def __init__(self,data,ticker,initial_capital,pct_change_entry=0.05,pct_change_exit=0.05,**kwargs):

        #Intialize required and default arguments
        self.ticker = ticker
        self.initial_cash = initial_capital
        self.pct_change_entry = pct_change_entry
        self.pct_change_exit = -abs(pct_change_exit)

        #kwargs
        self.use_fractional_shares = kwargs.get('use_fractional_shares', True)
        self.slippage = kwargs.get('slippage', 0.001)
        self.transaction_fees = kwargs.get('transaction_fees', 0.000)
        self.use_sizing = kwargs.get('use_sizing', False)

        if type(ticker) == str:
            # Prepare data
            self.data = data[data['Ticker'] == ticker]

            # Ensure the signals are correctly set
            self.data['entry_signal'] = ((self.data['Predicted'] - self.data['Close']) / self.data['Close']*100) >= self.pct_change_entry
            self.data['exit_signal'] = ((self.data['Predicted'] - self.data['Close']) / self.data['Close']*100) <= self.pct_change_exit

            self.data['signal'] = 'hold'
            self.data.loc[self.data['entry_signal'], 'signal'] = 'buy'
            self.data.loc[self.data['exit_signal'], 'signal'] = 'sell'
        
        else:
            raise ValueError("Invalid Ticker. Please provide a string.")

        if self.use_sizing:
            size_df = self.data
            size_df['sharpe_series'] = 0.68
            size_df['var_series'] = 0.05
            size_df['holdings'] = 0.0

            self.position_size = PortfolioManagerAgent(capital=200_000).compute_sizes_from_series(size_df['signal'],
                                                              size_df['Close'],
                                                              size_df['var_series'],
                                                              size_df['sharpe_series'],
                                                              size_df['holdings'])
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
            size_type=SizeType.Amount,
            accumulate=True
        )

        return portfolio
        

    def run_simulation(self):
        #print(f"Running vectorbt backtest for {self.ticker}")
        # Run the backtest using vectorbt's Portfolio
        self.pf = self.VectorBTBackTestSignals(self, self.data, self.initial_cash, self.position_size)
        
    def plot_performance(self):
        vbt.settings.set_theme("dark")

        # Plot OHLCV - Order PnL
        trades_fig = self.data[["Open", "High", "Low", "Close","Volume"]].vbt.ohlcv.plot(xaxis=dict(rangeslider=dict(visible=False)))
        self.pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=trades_fig)
        trades_fig.update_layout(yaxis=dict(title="Price (USD)"))

        # Plot Portfolio Value
        value_fig = self.pf.plot_value(trace_kwargs=dict(name="Strategy",line=dict(color='blue')))
        value_fig.update_layout(yaxis=dict(title="Portfolio Value (USD)"))

        # Plot Exposure Value
        exposure_fig = self.pf.plot_net_exposure(trace_kwargs=dict(name="Strategy",line=dict(color='blue')))
        exposure_fig.update_layout(yaxis=dict(title="Exposure %",range=[0.0, 1.0]))

        return trades_fig, value_fig, exposure_fig