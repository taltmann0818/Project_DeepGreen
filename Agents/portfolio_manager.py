import os
import numpy as np
import pandas as pd
import cvxpy as cp
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class PortfolioManagerAgent:
    """
    Daily rebalancing using Black-Litterman with turnover penalty.

    Attributes:
        delta (float): Risk aversion coefficient.
        tau (float): Scaling for prior covariance.
        view_confidence (float): Confidence in views (0-1).
        turnover_penalty (float): Penalty factor for turnover (L1 norm).
        max_position_pct (float): Max weight per asset.
        reserve_fraction (float): Cash reserve fraction.
    """
    def __init__(
        self,
        alpaca_api_key: str = 'PKVTMHC7LYMYM0PI6A1B',
        alpaca_api_secret: str = 'hYWD1Ij51TBlIrYgqtIpaWfSBmfTSmwjvJ3deM0O',
        paper: bool = True,
        delta: float = 2.5,
        tau: float = 0.05,
        view_confidence: float = 0.5,
        turnover_penalty: float = 10.0,
        max_position_pct: float = 0.20,
        reserve_fraction: float = 0.05,
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

        # BL params
        self.delta = delta
        self.tau = tau
        self.view_confidence = view_confidence
        self.turnover_penalty = turnover_penalty
        self.max_position_pct = max_position_pct
        self.reserve_fraction = reserve_fraction

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

    def compute_equilibrium_returns(self, market_weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
        return self.delta * cov_matrix.dot(market_weights)

    def compute_posterior_returns(
        self, market_weights: pd.Series, cov_matrix: pd.DataFrame, views: pd.Series
    ) -> pd.Series:
        """
        Implements Black-Litterman posterior:
        mu = μ^⊤(w) − (δ/2)w^⊤ Σw−ϕ∥w−w_t−1∥_1
        Here, P=I, Q=views, Omega=diag(view_confidence * diag(Sigma)).
        """
        tickers = cov_matrix.index
        pi = self.compute_equilibrium_returns(market_weights, cov_matrix).values
        P = np.eye(len(tickers))
        Q = views.reindex(tickers).fillna(0.0).values
        Omega = np.diag(np.clip(self.view_confidence * np.diag(cov_matrix.values), 1e-8, None))
        tau_sigma_inv = np.linalg.inv(self.tau * cov_matrix.values)
        omega_inv = np.linalg.inv(Omega)
        middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        posterior = middle @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        return pd.Series(posterior, index=tickers)

    def optimize_weights(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        current_weights: pd.Series
    ) -> pd.Series:
        """
        Mean-variance weights: w ∝ Sigma^-1 * mu, normalized to sum to 1.
        """
        tickers = expected_returns.index
        n = len(tickers)
        Sigma = cov_matrix.values
        mu = expected_returns.values
        w_prev = current_weights.reindex(tickers).fillna(0.0).values
        # CVXPY variable
        w = cp.Variable(n)
        ret_term = mu @ w
        risk_term = (self.delta/2) * cp.quad_form(w, Sigma)
        turn_term = self.turnover_penalty * cp.norm1(w - w_prev)
        objective = cp.Maximize(ret_term - risk_term - turn_term)
        constraints = [cp.sum(w) == 1,
                       w >= 0,
                       w <= self.max_position_pct]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        w_opt = np.maximum(np.array(w.value).flatten(), 0)
        w_opt = w_opt / w_opt.sum()
        return pd.Series(w_opt, index=tickers)

    def rebalance(
        self,
        predictions: pd.DataFrame,
        price_map: dict,
        cov_matrix: pd.DataFrame,
        alpha_map: dict = None,
    ) -> dict:
        """
        Rebalance daily using Black-Litterman.
        - predictions: DataFrame with columns ['Ticker','Predicted']
        - price_map: dict ticker->price
        - cov_matrix: DataFrame covariance of returns, indexed and col'd by tickers
        - market_caps: dict ticker->market cap
        """
        tickers = list(set(predictions['Ticker']))
        pred_rets = predictions.groupby('Ticker').apply(
            lambda df: (df['Predicted'].iloc[-1] - price_map.get(df.name,0))
                       / max(price_map.get(df.name,1),1)
        )
        views = pred_rets.reindex(tickers).fillna(0.0)

        # Alpha-weighted prior: normalize positive alphas
        if alpha_map:
            alphas = pd.Series(alpha_map).reindex(tickers).fillna(0.0).clip(lower=0.0)
            if alphas.sum() > 0:
                market_weights = alphas / alphas.sum()
        else:
            # Fallback to equal-weight prior
            market_weights = pd.Series(1.0 / len(tickers), index=tickers)

        # Posterior returns
        sub_cov = cov_matrix.loc[tickers, tickers]
        posterior = self.compute_posterior_returns(market_weights, sub_cov, views)
        # Current weights
        acct = self.get_account_info()
        current_pos = self.get_current_positions()
        pv = acct['portfolio_value']
        current_weights = pd.Series({
            t: current_pos.get(t,0)*price_map.get(t,0)/pv for t in tickers
        })
        # Optimize with turnover penalty
        weights = self.optimize_weights(posterior, sub_cov, current_weights)

        print(weights)

        # Compute dollar and share targets
        reserve = self.reserve_fraction * pv
        investable = max(acct['cash'] - reserve,0) + pv*(1-self.reserve_fraction)
        desired_dollar = weights * investable
        desired_shares = desired_dollar / pd.Series(price_map)

        # Build trade sizes
        trade_sizes = {t: float(desired_shares.get(t,0) - current_pos.get(t,0)) for t in tickers}

        print(trade_sizes)

        return self.execute_trades(trade_sizes, price_map)

    def compute_covariance_matrix(self, price_history: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the return covariance matrix for a set of assets.

        Parameters:
            price_history (pd.DataFrame): DataFrame with datetime index and columns as tickers' price series.

        Returns:
            pd.DataFrame: Covariance matrix of daily returns.
        """
        # Calculate daily returns
        returns = price_history.pct_change().dropna(how='all')
        # Compute and return covariance matrix
        return returns.cov()

    def format_price_history(
            self,
            long_df: pd.DataFrame,
            date_col: str = 'date',
            ticker_col: str = 'Ticker',
            price_col: str = 'price'
    ) -> pd.DataFrame:
        """
        Convert long-format DataFrame with repeated dates and tickers into a wide-format price history.

        Parameters:
            long_df (pd.DataFrame): Long-format DataFrame with columns for date, ticker, and price.
            date_col (str): Column name for dates.
            ticker_col (str): Column name for ticker symbols.
            price_col (str): Column name for price values.

        Returns:
            pd.DataFrame: Pivoted DataFrame with datetime index and tickers as columns.
        """
        # Ensure date_col is datetime
        df = long_df.copy().reset_index()
        #df[date_col] = pd.to_datetime(df[date_col])
        # Pivot to wide format
        price_history = df.pivot(index=date_col, columns=ticker_col, values=price_col)
        # Sort by date
        return price_history.sort_index()