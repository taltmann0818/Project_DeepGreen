from __future__ import annotations

import bisect
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

def _closest_prev_date(dates: List[pd.Timestamp], target: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the closest **prior** date in *dates* ≤ target, or None."""
    idx = bisect.bisect_right(dates, target)
    if idx == 0:
        return None
    return dates[idx - 1]

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
        rebalance_every: int = 3,
        lookback_window: int = 60,
        max_long_positions: int = 30,
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
        self.rebalance_every = rebalance_every
        self.lookback_window = lookback_window
        self.max_long_positions = max_long_positions

        # Initialize portfolio manager
        self.pm = PortfolioManagerAgent(
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
        self.pm.capital = initial_capital

        # Initialize tracking variables
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.records: List[Dict] = []
        self.trades: List[Dict] = []

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

        # 1) Build price history pivot & map -----------------------------
        price_history = stock_data.pivot(index="date", columns="Ticker", values="Close").sort_index()
        price_map_by_date = price_history.to_dict("index")
        date_index = price_history.index.tolist()

        pv_prev = self.initial_capital

        for step, (date, alpha_raw) in enumerate(sorted(alpha_dict.items())):
            # Pick closest trading date <= alpha date --------------------
            px_date = date if date in price_map_by_date else _closest_prev_date(date_index, date)
            if px_date is None:
                logging.warning(f"No price data before {date} – skipping." )
                continue
            prices_today = price_map_by_date[px_date]

            # ---------------- Mark‑to‑market every day ------------------
            pv_positions = sum(qty * prices_today.get(t, 0.0) for t, qty in self.positions.items())
            pv_today = pv_positions + self.cash
            daily_ret = (pv_today - pv_prev) / pv_prev if pv_prev > 0 else 0.0
            self.records.append({
                "date": date,
                "portfolio_value": pv_today,
                "daily_return": np.clip(daily_ret, -0.99, 1.0),
                "num_positions": len(self.positions),
            })
            pv_prev = pv_today

            # ---------------- Rebalance on schedule ----------------------
            if step % self.rebalance_every != 0:
                continue

            # Filter to tickers with valid prices -----------------------
            alpha = {t: a for t, a in alpha_raw.items() if t in prices_today and prices_today[t] > 0}
            if not alpha:
                continue

            # Top‑K positive alpha --------------------------------------
            pos_alpha = {t: a for t, a in alpha.items() if a > 0}
            # Sort by alpha value (descending) and then by ticker name (ascending) for deterministic tie-breaking
            top = sorted(pos_alpha, key=lambda t: (-pos_alpha[t], t))[: self.max_long_positions]
            if len(top) < 2:
                logging.debug(f"{date}: <2 positive alphas – skip rebalance")
                continue
            alpha_vec = pd.Series({t: alpha[t] for t in top})

            # Covariance matrix -----------------------------------------
            hist_window = price_history[top].iloc[-self.lookback_window :].dropna()
            if len(hist_window) < 30:
                continue
            cov = hist_window.pct_change().dropna().cov()

            # Current weights in aligned order --------------------------
            pv_curr = pv_today if pv_today > 0 else 1.0
            w_prev = pd.Series({t: self.positions.get(t, 0) * prices_today[t] / pv_curr for t in top})

            # Solve QP ---------------------------------------------------
            weights = self.pm._solve_qp(alpha_vec, cov, w_prev)
            if weights is None:
                continue

            # Translate weights → target shares -------------------------
            investable = pv_curr * (1.0 - self.pm.reserve_fraction)
            tgt_dollars = weights * investable
            tgt_shares = tgt_dollars / pd.Series({t: prices_today[t] for t in top})

            # Compute trades -------------------------------------------
            trades = tgt_shares - pd.Series({t: self.positions.get(t, 0.0) for t in top})
            for t, dq in trades.items():
                if abs(dq) < 1e-2:
                    continue
                trade_val = dq * prices_today[t]
                side = "buy" if dq > 0 else "sell"
                self.trades.append({
                    "date": date,
                    "ticker": t,
                    "side": side,
                    "qty": abs(dq),
                    "price": prices_today[t],
                    "dollar": abs(trade_val),
                    "alpha": alpha_vec[t],
                })
                # Update cash & positions ------------------------------
                self.cash -= trade_val
                new_pos = self.positions.get(t, 0.0) + dq
                if abs(new_pos) < 1e-2:
                    self.positions.pop(t, None)
                else:
                    self.positions[t] = new_pos

        # Convert returns history to DataFrame
        df = pd.DataFrame(self.records).sort_values("date").reset_index(drop=True)
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
        logging.info(
            f"Back‑test done: PV = ${pv_prev:,.0f}, Total Return = {df['cumulative_return'].iloc[-1]:.2%}"
        )
        return df

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
            'num_trades': len(pd.DataFrame(self.trades)),
            'avg_positions': returns_df['num_positions'].mean()
        }

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the backtesting engine
    pass
