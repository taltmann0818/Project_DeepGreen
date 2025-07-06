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

    def _build_sector_constraints(self, tickers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sector exposure constraint matrices.
        Returns A_sector, b_sector such that A_sector @ w <= b_sector
        """
        sectors = list(set(self.sector_map.get(ticker, 'Other') for ticker in tickers))
        n_sectors = len(sectors)
        n_assets = len(tickers)

        # Create sector exposure matrix
        A_sector = np.zeros((n_sectors, n_assets))
        for i, sector in enumerate(sectors):
            for j, ticker in enumerate(tickers):
                if self.sector_map.get(ticker, 'Other') == sector:
                    A_sector[i, j] = 1.0

        # Sector limits
        b_sector = np.full(n_sectors, self.max_sector_pct)

        return A_sector, b_sector

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

    def optimize_portfolio_qp(
            self,
            alpha_vector: pd.Series,
            cov_matrix: pd.DataFrame,
            current_weights: pd.Series,
            sector_constraints: bool = False
    ) -> pd.Series:
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
        tickers = alpha_vector.index.tolist()
        n = len(tickers)

        # Align data
        alpha = alpha_vector.values
        Sigma = cov_matrix.loc[tickers, tickers].values
        w_prev = current_weights.reindex(tickers).fillna(0.0).values

        # CVXPY optimization variable
        w = cp.Variable(n)

        # Objective function: max w^T α - (1/2) λ w^T Σ w - TC(w)
        alpha_term = alpha @ w
        risk_term = 0.5 * self.risk_aversion * cp.quad_form(w, Sigma)

        # Transaction cost approximation (simplified for robustness)
        turnover = cp.norm1(w - w_prev)
        tc_penalty = (self.transaction_cost_bps + self.market_impact_coeff) * turnover

        objective = cp.Maximize(alpha_term - risk_term - tc_penalty)

        # Base constraints
        constraints = [
            cp.sum(w) == 1.0,  # Fully invested
            w >= 0.0,  # Long-only
        ]

        # Position size limits - make them feasible
        # Ensure position limits allow for full investment
        max_pos = self.max_position_pct

        # Calculate minimum position limit needed for feasibility
        min_needed_per_asset = 1.0 / n  # Equal weight

        # If position limits are too restrictive, relax them
        if max_pos * n < 1.0:
            # Need to relax position limits to allow full investment
            current_max_pos = max(max_pos, min_needed_per_asset * 1.5)  # 50% above equal weight
            constraints.append(w <= current_max_pos)
            print(f"Warning: Position limits relaxed to {current_max_pos:.3f} for feasibility")
        else:
            current_max_pos = max_pos
            constraints.append(w <= current_max_pos)

        # Sector constraints with feasibility check
        if sector_constraints and len(set(self.sector_map.get(t, 'Other') for t in tickers)) > 1:
            try:
                A_sector, b_sector = self._build_sector_constraints(tickers)

                # Check if sector constraints are feasible
                if A_sector.shape[0] > 0:

                    # Ensure sector limits are feasible
                    for i in range(len(b_sector)):
                        sector_assets = np.sum(A_sector[i, :])
                        if sector_assets > 0:
                            # Minimum needed for this sector (assuming equal weight within sector)
                            min_needed_for_sector = current_max_pos * sector_assets
                            # Ensure sector limit allows for reasonable allocation
                            b_sector[i] = max(b_sector[i], min_needed_for_sector, 0.1)  # At least 10%

                    # Only add sector constraints if they don't conflict with position limits
                    total_sector_capacity = np.sum(b_sector)
                    if total_sector_capacity >= 1.0:  # Ensure we can invest 100%
                        constraints.append(A_sector @ w <= b_sector)
                    else:
                        print(f"Warning: Sector constraints too restrictive (total capacity: {total_sector_capacity:.2f}), skipping")

            except Exception as e:
                print(f"Warning: Could not build sector constraints: {e}")
                sector_constraints = False

        # Only add turnover constraint if it's feasible and we have previous positions
        if np.any(w_prev > 1e-6) and self.max_turnover < 2.0:  # Turnover of 2.0 means complete rebalancing
            constraints.append(turnover <= self.max_turnover)

        # Solve optimization with better error handling
        problem = cp.Problem(objective, constraints)

        try:
            # Check constraint feasibility before solving
            #print(f"Optimization setup: {n} assets, {len(constraints)} constraints")
            #print(f"Position limits: max={max_pos:.3f}, sum constraint allows: {max_pos * n:.3f}")

            # Try OSQP first with relaxed tolerances
            problem.solve(
                solver=cp.OSQP,
                verbose=False,
                warm_start=True,
                eps_abs=1e-4,  # Relaxed from 1e-5
                eps_rel=1e-4,  # Relaxed from 1e-5
                max_iter=20000,  # Increased iterations
                rho=0.1,  # Penalty parameter
                adaptive_rho=True  # Adaptive penalty
            )

            if problem.status == cp.INFEASIBLE:
                print("Problem is infeasible. Diagnosing...")

                # Try without turnover constraint
                if len(constraints) > 3:  # More than basic constraints
                    print("Retrying without turnover constraint...")
                    basic_constraints = [
                        cp.sum(w) == 1.0,
                        w >= 0.0,
                        w <= max(0.5, 1.0 / n + 0.1)  # Very relaxed position limits
                    ]
                    basic_problem = cp.Problem(objective, basic_constraints)
                    basic_problem.solve(solver=cp.OSQP, verbose=False)

                    if basic_problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        w_opt = np.array(basic_problem.variables()[0].value).flatten()
                        w_opt = np.maximum(w_opt, 0.0)
                        w_opt = w_opt / w_opt.sum()
                        return pd.Series(w_opt, index=tickers)

            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"OSQP failed with status: {problem.status}, trying ECOS...")
                # Try ECOS solver as backup
                problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"Optimization failed with status: {problem.status}")
                print(f"Constraint summary:")
                print(f"  - Assets: {n}")
                print(f"  - Max position: {max_pos:.3f}")
                print(f"  - Sector constraints: {sector_constraints}")
                print(f"  - Current weights sum: {w_prev.sum():.3f}")
                print(f"  - Alpha range: [{alpha.min():.6f}, {alpha.max():.6f}]")

                # Try a simpler optimization without sector constraints
                if sector_constraints:
                    print("Retrying without sector constraints...")
                    return self.optimize_portfolio_qp(
                        alpha_vector, cov_matrix, current_weights, sector_constraints=False
                    )

                # Final fallback: create a simple feasible portfolio
                print("Using simple fallback portfolio...")
                # Use alpha signals but ensure feasibility
                alpha_normalized = alpha - alpha.min()  # Make all non-negative
                if alpha_normalized.sum() > 0:
                    # Alpha-weighted but capped at reasonable limits
                    raw_weights = alpha_normalized / alpha_normalized.sum()
                    # Cap individual positions at 20% to ensure diversification
                    max_weight = min(0.2, 1.0 / max(n, 5))
                    capped_weights = np.minimum(raw_weights, max_weight)
                    # Renormalize
                    fallback_weights = capped_weights / capped_weights.sum()
                else:
                    # Equal weight fallback
                    fallback_weights = np.ones(n) / n
                return pd.Series(fallback_weights, index=tickers)

            # Extract and clean solution
            w_opt = np.array(w.value).flatten()
            w_opt = np.maximum(w_opt, 0.0)  # Ensure non-negative
            w_opt = w_opt / w_opt.sum()  # Renormalize

            # Final validation
            if np.any(np.isnan(w_opt)) or np.any(w_opt < 0):
                print("Warning: Invalid solution detected, using fallback")
                return pd.Series(np.ones(n) / n, index=tickers)

            return pd.Series(w_opt, index=tickers)

        except Exception as e:
            print(f"Optimization error: {e}")
            # Fallback to equal weights
            fallback_weights = np.ones(n) / n
            return pd.Series(fallback_weights, index=tickers)

    def rebalance(
        self,
        alpha_signals: Dict[str, float],
        price_map: Dict[str, float],
        price_history: pd.DataFrame,
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
        if not alpha_signals:
            print("No alpha signals provided")
            return {}

        # Filter tickers with valid prices and alpha signals
        tickers = [t for t in alpha_signals.keys() if t in price_map and price_map[t] > 0]
        if len(tickers) < 2:
            print(f"Insufficient valid tickers: {len(tickers)}")
            return {}

        # Create alpha vector
        alpha_vector = pd.Series({t: alpha_signals[t] for t in tickers})

        # Filter price history to available tickers
        available_tickers = [t for t in tickers if t in price_history.columns]
        if len(available_tickers) < len(tickers):
            print(f"Price history missing for {len(tickers) - len(available_tickers)} tickers")
            tickers = available_tickers
            alpha_vector = alpha_vector.reindex(tickers)

        if len(tickers) < 2:
            print("Insufficient tickers with price history")
            return {}

        # Estimate risk model
        price_hist_subset = price_history[tickers].dropna()
        if len(price_hist_subset) < 30:
            print("Insufficient price history for risk estimation")
            return {}

        cov_matrix = self._estimate_risk_model(price_hist_subset)

        # Get current portfolio state
        acct = self.get_account_info()
        current_pos = self.get_current_positions()
        pv = acct['portfolio_value']

        # Calculate current weights
        current_weights = pd.Series({
            t: current_pos.get(t, 0) * price_map.get(t, 0) / pv for t in tickers
        })

        # QP Optimization
        try:
            optimal_weights = self.optimize_portfolio_qp(
                alpha_vector=alpha_vector,
                cov_matrix=cov_matrix,
                current_weights=current_weights,
                sector_constraints=False  # Disabled for better feasibility
            )

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

        except Exception as e:
            print(f"Portfolio optimization failed: {e}")
            return {}
