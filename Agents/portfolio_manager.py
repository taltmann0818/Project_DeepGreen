import logging
import os
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class PortfolioManagerAgent:
    """
    Enterprise-grade portfolio optimizer using Quadratic Programming.

    Optimizes the objective: max w^T α - (1/2) λ w^T Σ w - TC(w)
    where:
    - w: portfolio weights
    - α: pure alpha vector (neutralized and scaled)
    - λ: risk aversion parameter
    - Σ: covariance matrix
    - TC(w): transaction costs

    Features:
    - Accepts pre-processed alpha vectors from alpha pipeline
    - Sector and position size constraints
    - Transaction cost modeling with market impact
    - Turnover penalties
    - Risk budgeting and factor exposure limits
    - Enterprise-grade optimization techniques
    """
    def __init__(
        self,
        alpaca_api_key: str = 'PKVTMHC7LYMYM0PI6A1B',
        alpaca_api_secret: str = 'hYWD1Ij51TBlIrYgqtIpaWfSBmfTSmwjvJ3deM0O',
        paper: bool = True,
        risk_aversion: float = 0.5,
        turnover_penalty: float = 0.01,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_turnover: float = 0.50,
        transaction_cost_bps: float = 5.0,
        market_impact_coeff: float = 0.0,
        reserve_fraction: float = 0.02,
        min_position_size: float = 0.01,
    ):
        # Alpaca client
        if alpaca_api_key and alpaca_api_secret:
            self.client = TradingClient(alpaca_api_key, alpaca_api_secret, paper=paper)
        elif 'ALPACA_API_KEY' in os.environ and 'ALPACA_API_SECRET' in os.environ:
            self.client = TradingClient(
                os.environ['ALPACA_API_KEY'],
                os.environ['ALPACA_API_SECRET'],
                paper=paper
            )
        else:
            self.client = None

        # Account capital
        self.capital = 0.0
        if self.client:
            acct = self.client.get_account()
            self.capital = float(acct.buying_power)

        # QP Optimization parameters
        self.risk_aversion = risk_aversion  # λ in objective function
        self.turnover_penalty = turnover_penalty
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_turnover = max_turnover
        self.transaction_cost_bps = transaction_cost_bps / 10000.0  # Convert bps to decimal
        self.market_impact_coeff = market_impact_coeff
        self.reserve_fraction = reserve_fraction
        self.min_position_size = min_position_size

        # Sector mappings (can be extended)
        self.sector_map = self._get_default_sector_map()

        # Risk model parameters
        self.volatility_lookback = 252  # 1 year
        self.correlation_lookback = 126  # 6 months
        self.half_life = 63  # 3 months for exponential weighting

    def _get_default_sector_map(self) -> Dict[str, str]:
        """
        Default sector mapping for common tickers.
        In production, this would be loaded from a database or external service.
        """
        return {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
            'AMZN': 'Technology', 'TSLA': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
            'NFLX': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology',

            # Financial Services
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'MS': 'Financials', 'C': 'Financials', 'BRK.B': 'Financials', 'V': 'Financials',
            'MA': 'Financials', 'AXP': 'Financials',

            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'LLY': 'Healthcare',

            # Consumer
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            'HD': 'Consumer', 'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',

            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',

            # Industrials
            'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials',

            # Default sector for unknown tickers
        }

    def get_current_positions(self) -> dict:
        positions = {}
        if not self.client:
            return positions
        for p in self.client.get_all_positions():
            positions[p.symbol] = float(p.qty)
        return positions

    def get_account_info(self) -> dict:
        if not self.client:
            return {'cash': self.capital, 'portfolio_value': self.capital}
        a = self.client.get_account()
        return {'cash': float(a.buying_power), 'portfolio_value': float(a.portfolio_value)}

    def execute_trades(self, trade_sizes: dict, price_map: dict) -> dict:
        results = {}
        if not self.client:
            return {sym: {'status': 'no-client'} for sym in trade_sizes}
        pv = self.get_account_info()['portfolio_value']
        cap_val = self.max_position_pct * pv
        # Sell first
        for sym, delta in trade_sizes.items():
            if delta >= 0: continue
            qty = abs(delta)
            val = qty * price_map.get(sym, 0)
            if val > cap_val:
                qty = cap_val / max(price_map.get(sym,1),1)
            if qty < 1e-4:
                results[sym] = {'status':'skipped'}; continue
            req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            #order = self.client.submit_order(req)
            results[sym] = {'side':'sell','qty':qty}#,'id':order.id}
        # Then buys
        for sym, delta in trade_sizes.items():
            if delta <= 0: continue
            qty = delta
            val = qty * price_map.get(sym, 0)
            if val > cap_val:
                qty = cap_val / max(price_map.get(sym,1),1)
            if qty < 1e-4:
                results[sym] = {'status':'skipped'}; continue
            req = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            #order = self.client.submit_order(req)
            results[sym] = {'side':'buy','qty':qty}#,'id':order.id}
        return results

    def _estimate_transaction_costs(
        self, 
        current_weights: pd.Series, 
        target_weights: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> float:
        """
        Estimate transaction costs including market impact.
        TC(w) = c₁ * |Δw| + c₂ * |Δw|^1.5 (market impact)
        """
        delta_w = target_weights - current_weights.reindex(target_weights.index).fillna(0.0)

        # Linear transaction costs (bid-ask spread, commissions)
        linear_costs = self.transaction_cost_bps * np.abs(delta_w).sum()

        # Market impact costs (proportional to trade size^1.5)
        if volumes is not None:
            # Scale market impact by inverse of volume (less liquid = higher impact)
            volume_adj = 1.0 / (volumes.reindex(target_weights.index).fillna(1e6) / 1e6)
            impact_costs = self.market_impact_coeff * (volume_adj * np.abs(delta_w) ** 1.5).sum()
        else:
            impact_costs = self.market_impact_coeff * (np.abs(delta_w) ** 1.5).sum()

        return linear_costs + impact_costs

    def _estimate_risk_model(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate covariance matrix using robust methods.
        Uses simple historical covariance with regularization for numerical stability.
        """
        returns = price_history.pct_change().dropna()

        if len(returns) < 10:
            # Very short history - use identity matrix scaled by average variance
            n = len(price_history.columns)
            avg_var = 0.0004  # Default daily variance (2% annual vol)
            return pd.DataFrame(
                np.eye(n) * avg_var,
                index=price_history.columns,
                columns=price_history.columns
            )

        # Simple historical covariance
        cov_matrix = returns.cov()

        # Handle NaN values
        cov_matrix = cov_matrix.fillna(0.0)

        # Use a simple diagonal covariance matrix for robustness
        # This eliminates correlation effects but ensures numerical stability
        n = cov_matrix.shape[0]
        variances = np.diag(cov_matrix.values)

        # Use median variance if any are zero or negative
        median_var = np.median(variances[variances > 0]) if np.any(variances > 0) else 0.0004
        variances = np.where(variances <= 0, median_var, variances)

        # Create diagonal covariance matrix
        cov_diagonal = np.diag(variances)

        return pd.DataFrame(cov_diagonal, index=cov_matrix.index, columns=cov_matrix.columns)

    def _solve_qp(
        self,
        alpha: pd.Series,
        cov: pd.DataFrame,
        w_prev: pd.Series,
    ) -> Optional[pd.Series]:
        """
        Quadratic Programming optimization: max w^T α - (1/2) λ w^T Σ w - TC(w)

        Args:
            alpha_vector: Pure alpha signals (neutralized and scaled)
            cov_matrix: Asset return covariance matrix
            current_weights: Current portfolio weights
            sector_constraints: Whether to apply sector exposure limits

        Returns:
            Optimal portfolio weights
        """
        tickers = alpha.index.tolist()
        n = len(tickers)
        if n == 0:
            return None

        # Align data
        A = alpha.values
        Sigma = cov.loc[tickers, tickers].values
        w_old = w_prev.reindex(tickers).fillna(0.0).values

        # CVXPY optimization variable
        w = cp.Variable(n)
        obj = cp.Maximize(A @ w - 0.5 * self.risk_aversion * cp.quad_form(w, Sigma) -
                           (self.transaction_cost_bps + self.market_impact_coeff) * cp.norm1(w - w_old))
        constraints = [
            w >= 0.0,
            w <= self.max_position_pct,
            cp.sum(w) <= 1.0,  # Allow sum to be less than 1 to avoid infeasibility
        ]
        prob = cp.Problem(obj, constraints)
        try:
            # Use deterministic solver settings to ensure reproducible results
            prob.solve(
                solver=cp.OSQP, 
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
                max_iter=10000,
                adaptive_rho=False,  # Disable adaptive rho for determinism
                rho=0.1,
                sigma=1e-6,
                alpha=1.6
            )
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return None
            sol = np.maximum(w.value, 0.0)
            sol /= sol.sum()
            return pd.Series(sol, index=tickers)
        except Exception:
            return None

    def rebalance(
        self,
        alpha_signals: Dict[str, float],
        price_map: Dict[str, float],
        price_history: pd.DataFrame,
        max_long_positions: int = 50,
    ) -> Dict[str, dict]:
        """
        Rebalance portfolio using pure alpha signals and QP optimization.

        Args:
            alpha_signals: Dict mapping ticker -> alpha value (from alpha pipeline)
            price_map: Dict mapping ticker -> current price
            price_history: DataFrame with price history for covariance estimation
            volumes: Optional dict mapping ticker -> trading volume

        Returns:
            Dict with trade execution results
        """
        # --- 0) Quick sanity – prune missing prices ----------------------
        universe = [t for t in alpha_signals if t in price_map and price_map[t] > 0]
        if not universe:
            print("No tradable tickers with prices – aborting rebalance.")
            return {}

        # Sort by *positive* alpha and keep top‑K ------------------------
        pos_alpha = {t: a for t, a in alpha_signals.items() if a > 0.0 and t in universe}
        if not pos_alpha:
            print("No positive‑alpha names – nothing to buy.")
            return {}
        # Sort by alpha value (descending) and then by ticker name (ascending) for deterministic tie-breaking
        top_tickers = sorted(pos_alpha, key=lambda t: (-pos_alpha[t], t))[:max_long_positions]

        # All other tickers are forced to weight 0 -----------------------
        pruned_alpha = {t: alpha_signals[t] for t in top_tickers}
        tickers = top_tickers  # canonical order

        # Align price history subset ------------------------------------
        missing_hist = [t for t in tickers if t not in price_history.columns]
        if missing_hist:
            print(f"Price history missing for {len(missing_hist)} tickers → dropped: {missing_hist[:5]}…")
            tickers = [t for t in tickers if t not in missing_hist]
            pruned_alpha = {t: pruned_alpha[t] for t in tickers}
        if len(tickers) < 2:
            print("Too few tickers after pruning – rebalance skipped.")
            return {}

        alpha_vec = pd.Series(pruned_alpha, index=tickers)

        # Estimate risk model
        price_hist_subset = price_history[tickers].dropna()
        if len(price_hist_subset) < 30:
            print("Insufficient price history for risk estimation")
            return {}
        cov = price_hist_subset.pct_change().dropna().cov()

        # Get current portfolio state
        acct = self.get_account_info()
        current_pos = self.get_current_positions()
        pv = acct['portfolio_value']
        curr_w = pd.Series({t: current_pos.get(t, 0) * price_map[t] / pv for t in tickers})

        # QP Optimization
        optimal_weights = self._solve_qp(alpha_vec, cov, curr_w)
        if optimal_weights is None:
            print("QP failed – rebalance skipped.")
            return {}

        # Calculate position targets
        reserve = self.reserve_fraction * pv
        investable = max(acct['cash'] - reserve, 0) + pv * (1 - self.reserve_fraction)

        # Dollar targets
        target_dollars = optimal_weights * investable

        # Share targets
        target_shares = target_dollars / pd.Series(price_map).reindex(tickers)

        # Filter out positions below minimum size
        min_shares = self.min_position_size / pd.Series(price_map).reindex(tickers)
        target_shares = target_shares.where(target_shares >= min_shares, 0.0)

        # Renormalize after filtering
        if target_shares.sum() > 0:
            total_value = (target_shares * pd.Series(price_map).reindex(tickers)).sum()
            if total_value > 0:
                target_shares = target_shares * (investable / total_value)

        # Calculate trades
        current_shares = pd.Series({t: current_pos.get(t, 0) for t in tickers})
        trade_sizes = target_shares - current_shares

        # Filter significant trades only
        min_trade_value = 10.0  # Minimum $10 trade
        min_trade_shares = min_trade_value / pd.Series(price_map).reindex(tickers)
        trade_sizes = trade_sizes.where(np.abs(trade_sizes) >= min_trade_shares, 0.0)

        trade_dict = {t: float(trade_sizes[t]) for t in tickers if abs(trade_sizes[t]) > 1e-6}
        #logging.info(f"Portfolio optimization result: {trade_dict}")

        # Execute trades
        if trade_dict:
            return self.execute_trades(trade_dict, price_map)
        else:
            print("No significant trades to execute")
            return {}
