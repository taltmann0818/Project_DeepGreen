from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import logging

# Conditional import for portfolio manager (requires cvxpy)
try:
    from Agents.portfolio_manager import PortfolioManagerAgent
    PORTFOLIO_MANAGER_AVAILABLE = True
except ImportError:
    PORTFOLIO_MANAGER_AVAILABLE = False
    PortfolioManagerAgent = None

class CustomBacktestingEngine:
    """
    Custom backtesting engine that processes alpha signals day by day and tracks portfolio performance.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        alpaca_api_key: str = '',
        alpaca_api_secret: str = '',
        risk_aversion: float = 5.0,
        turnover_penalty: float = 0.01,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_turnover: float = 0.50,
        transaction_cost_bps: float = 5.0,
        market_impact_coeff: float = 0.0,
        reserve_fraction: float = 0.02,
        min_position_size: float = 0.01,
    ):
        """
        Initialize the custom backtesting engine.

        Parameters
        ----------
        initial_capital : float
            Starting capital for backtesting
        alpaca_api_key : str
            Alpaca API key (empty for backtesting mode)
        alpaca_api_secret : str
            Alpaca API secret (empty for backtesting mode)
        **kwargs
            Additional parameters passed to PortfolioManagerAgent
        """
        self.initial_capital = initial_capital

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManagerAgent(
            alpaca_api_key=alpaca_api_key,
            alpaca_api_secret=alpaca_api_secret,
            paper=True,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            max_position_pct=max_position_pct,
            max_sector_pct=max_sector_pct,
            max_turnover=max_turnover,
            transaction_cost_bps=transaction_cost_bps,
            market_impact_coeff=market_impact_coeff,
            reserve_fraction=reserve_fraction,
            min_position_size=min_position_size,
        )

        # Set fixed capital for backtesting
        self.portfolio_manager.capital = initial_capital

        # Initialize tracking variables
        self.trade_records = []
        self.portfolio_history = []
        self.returns_history = []
        self.current_positions = {}  # Track simulated positions
        self.cash = initial_capital  # Track simulated cash

        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def prepare_price_data(self, stock_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare price history and price map from stock data.

        Parameters
        ----------
        stock_data : pd.DataFrame
            Stock data with columns ['date', 'Ticker', 'Close', etc.]

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Price history pivot table and price map dictionary
        """
        # Create price history pivot table
        price_history = stock_data.pivot(index='date', columns='Ticker', values='Close')

        # Create price map: {date: {ticker: price}}
        price_map = dict(zip(
            stock_data['date'].unique(),
            [dict(zip(stock_data[stock_data['date'] == d]['Ticker'],
                     stock_data[stock_data['date'] == d]['Close']))
             for d in stock_data['date'].unique()]
        ))

        return price_history, price_map

    def run_backtest(
        self,
        alpha_dict: Dict[pd.Timestamp, Dict[str, float]],
        stock_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the complete backtesting process.

        Parameters
        ----------
        alpha_dict : Dict[pd.Timestamp, Dict[str, float]]
            Alpha signals by date and ticker
        stock_data : pd.DataFrame
            Historical stock price data
        benchmark_data : pd.DataFrame, optional
            Benchmark return data for comparison

        Returns
        -------
        pd.DataFrame
            Daily returns DataFrame suitable for quantstats analysis
        """
        logging.info(f"Starting backtest with {len(alpha_dict)} dates...")

        # Prepare price data
        price_history, price_map = self.prepare_price_data(stock_data)

        # Filter stock data to only include tickers in alpha signals
        all_tickers = set()
        for signals in alpha_dict.values():
            all_tickers.update(signals.keys())

        filtered_stock_data = stock_data[stock_data['Ticker'].isin(all_tickers)]

        # Initialize portfolio tracking
        current_portfolio_value = self.initial_capital
        previous_portfolio_value = self.initial_capital

        # Iterate through each date in the alpha dictionary
        for date, alpha_signals in tqdm(alpha_dict.items(), desc="Processing dates"):
            try:
                # Get the price map for this specific date
                if date in price_map:
                    current_price_map = price_map[date]
                else:
                    # Try to find the closest date if exact match not found
                    available_dates = list(price_map.keys())
                    if not available_dates:
                        continue
                    closest_date = min(available_dates, key=lambda x: abs((x - date).days))
                    current_price_map = price_map[closest_date]
                    logging.warning(f"Using closest date {closest_date} for alpha date {date}")

                # Filter alpha signals to only include tickers with available prices
                filtered_alpha = {
                    ticker: alpha for ticker, alpha in alpha_signals.items()
                    if ticker in current_price_map and current_price_map[ticker] > 0
                }

                if len(filtered_alpha) < 2:
                    logging.debug(f"Skipping date {date}: insufficient valid tickers ({len(filtered_alpha)})")
                    # Record no change in portfolio value
                    self.returns_history.append({
                        'date': date,
                        'portfolio_value': current_portfolio_value,
                        'daily_return': 0.0,
                        'num_positions': 0
                    })
                    continue

                # Get optimal weights from portfolio manager
                tickers = list(filtered_alpha.keys())
                alpha_vector = pd.Series(filtered_alpha)

                # Filter price history to available tickers
                available_tickers = [t for t in tickers if t in price_history.columns]
                if len(available_tickers) < len(tickers):
                    tickers = available_tickers
                    alpha_vector = alpha_vector.reindex(tickers)

                if len(tickers) < 2:
                    continue

                # Estimate risk model
                price_hist_subset = price_history[tickers].dropna()
                if len(price_hist_subset) < 10:
                    continue

                cov_matrix = self.portfolio_manager._estimate_risk_model(price_hist_subset)

                # Calculate current weights
                current_portfolio_value = sum(
                    pos * current_price_map.get(ticker, 0) 
                    for ticker, pos in self.current_positions.items()
                ) + self.cash

                current_weights = pd.Series({
                    t: self.current_positions.get(t, 0) * current_price_map.get(t, 0) / current_portfolio_value 
                    for t in tickers
                })

                # Get optimal weights
                optimal_weights = self.portfolio_manager.optimize_portfolio_qp(
                    alpha_vector=alpha_vector,
                    cov_matrix=cov_matrix,
                    current_weights=current_weights,
                    sector_constraints=False
                )

                # Calculate target positions in shares
                investable_capital = current_portfolio_value * 0.98  # Keep 2% cash reserve
                target_dollar_positions = optimal_weights * investable_capital
                target_share_positions = target_dollar_positions / pd.Series(current_price_map).reindex(tickers)

                # Calculate trades needed
                current_share_positions = pd.Series({t: self.current_positions.get(t, 0) for t in tickers})
                trade_shares = target_share_positions - current_share_positions

                # Execute trades and record them
                trades_executed = 0
                executed_trades = {}

                for ticker in tickers:
                    trade_qty = trade_shares.get(ticker, 0)
                    if abs(trade_qty) > 0.01:  # Only execute significant trades
                        price = current_price_map.get(ticker, 0)

                        # Record the trade before execution
                        side = 'buy' if trade_qty > 0 else 'sell'
                        executed_trades[ticker] = {
                            'side': side,
                            'qty': abs(trade_qty),
                            'price': price,
                            'status': 'executed'
                        }

                        # Update positions
                        self.current_positions[ticker] = target_share_positions.get(ticker, 0)

                        # Update cash (negative for buys, positive for sells)
                        self.cash -= trade_qty * price
                        trades_executed += 1

                        # Remove zero positions
                        if abs(self.current_positions[ticker]) < 0.01:
                            self.current_positions.pop(ticker, None)

                # Optional: Log execution info for monitoring
                if len(self.returns_history) < 3:
                    logging.debug(f"Date {date}: Executed {trades_executed} trades, Cash: ${self.cash:.2f}, Positions: {len(self.current_positions)}")

                # Calculate portfolio value after trades
                portfolio_value_from_positions = sum(
                    pos * current_price_map.get(ticker, 0) 
                    for ticker, pos in self.current_positions.items()
                )
                current_portfolio_value = portfolio_value_from_positions + self.cash

                # Calculate daily return with safeguards
                if previous_portfolio_value > 0:
                    daily_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
                    # Cap extreme returns to prevent numerical issues
                    daily_return = max(min(daily_return, 1.0), -0.99)  # Cap at +100% and -99%
                else:
                    daily_return = 0.0

                # Record portfolio state
                self.returns_history.append({
                    'date': date,
                    'portfolio_value': current_portfolio_value,
                    'daily_return': daily_return,
                    'num_positions': len(self.current_positions)
                })

                # Record executed trades
                for ticker, trade_info in executed_trades.items():
                    trade_record = {
                        'date': date,
                        'ticker': ticker,
                        'side': trade_info['side'],
                        'qty': trade_info['qty'],
                        'price': trade_info['price'],
                        'dollar_amount': trade_info['qty'] * trade_info['price'],
                        'alpha_signal': filtered_alpha.get(ticker, 0),
                        'status': trade_info['status']
                    }
                    self.trade_records.append(trade_record)

                # Record alpha signals for tickers with no trades
                for ticker, alpha_val in filtered_alpha.items():
                    if ticker not in executed_trades:
                        trade_record = {
                            'date': date,
                            'ticker': ticker,
                            'side': 'no_trade',
                            'qty': 0,
                            'price': current_price_map.get(ticker, 0),
                            'dollar_amount': 0,
                            'alpha_signal': alpha_val,
                            'status': 'no_trade'
                        }
                        self.trade_records.append(trade_record)

                previous_portfolio_value = current_portfolio_value

            except Exception as e:
                logging.error(f"Error processing date {date}: {str(e)}")
                # Record no change for this date
                self.returns_history.append({
                    'date': date,
                    'portfolio_value': current_portfolio_value,
                    'daily_return': 0.0,
                    'num_positions': 0
                })
                continue

        # Convert returns history to DataFrame
        returns_df = pd.DataFrame(self.returns_history)

        if len(returns_df) > 0:
            returns_df = returns_df.sort_values('date').reset_index(drop=True)
            returns_df['cumulative_return'] = (1 + returns_df['daily_return']).cumprod() - 1

            logging.info(f"Backtest completed:")
            logging.info(f"Total records: {len(returns_df)}")
            #logging.info(f"Date range: {returns_df['date'].min()} to {returns_df['date'].max()}")
            logging.info(f"Final portfolio value: ${returns_df['portfolio_value'].iloc[-1]:,.2f}")
            logging.info(f"Total return: {returns_df['cumulative_return'].iloc[-1]:.2%}")

        return returns_df

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades executed during backtest.

        Returns
        -------
        pd.DataFrame
            DataFrame with all trade records
        """
        if not self.trade_records:
            return pd.DataFrame()

        trades_df = pd.DataFrame(self.trade_records)
        trades_df = trades_df.sort_values(['date', 'ticker']).reset_index(drop=True)

        return trades_df

    def get_performance_metrics(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic performance metrics from returns.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns DataFrame from run_backtest

        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if len(returns_df) == 0:
            return {}

        return {
            'total_return': returns_df['cumulative_return'].iloc[-1],
            'num_trades': len(self.trade_records),
            'avg_positions': returns_df['num_positions'].mean()
        }


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the backtesting engine
    pass
