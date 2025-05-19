import pandas as pd

class PortfolioManagerAgent:
    """
    PortfolioManagerAgent computes position sizes based on daily VaR and
    adjusts them by backtested Sharpe ratios, while ensuring sufficient
    capital for executions, handling buy/sell/hold signals, supporting
    fractional share sizes, and normalizing allocations to a target capital usage.

    Attributes:
        capital (float): Total capital available for risk calculations.
        risk_fraction (float): Fraction of capital to risk per trade (e.g., 0.01 for 1%).
        sharpe_target (float): Reference Sharpe ratio for sizing (e.g., 1.0).
        reserve_fraction (float): Fraction of available cash to hold in reserve (e.g., 0.1 for 10%).
        min_size (float or None): Minimum number of shares per trade.
        max_size (float or None): Maximum number of shares per trade.
    """

    def __init__(
        self,
        capital: float,
        risk_fraction: float = 0.01,
        sharpe_target: float = 1.0,
        reserve_fraction: float = 0.0,
        min_size: float = None,
        max_size: float = None,
    ):
        self.capital = capital
        self.risk_fraction = risk_fraction
        self.sharpe_target = sharpe_target
        self.reserve_fraction = reserve_fraction
        self.min_size = min_size
        self.max_size = max_size

    def compute_baseline_size(self, var_daily: float, price: float) -> float:
        """
        Compute baseline share count based on daily VaR and price.

        N_baseline = (risk_fraction * capital) / (VaR_daily * price)
        """
        return (self.risk_fraction * self.capital) / (var_daily * price)

    def compute_adjusted_size(self, var_daily: float, price: float, sharpe: float) -> float:
        """
        Compute un-normalized position size (can be fractional), adjusting baseline by Sharpe ratio
        and applying optional caps/floors.

        N_raw = N_baseline * (sharpe / sharpe_target)
        """
        baseline = self.compute_baseline_size(var_daily, price)
        factor = sharpe / self.sharpe_target if self.sharpe_target != 0 else 0
        size = baseline * factor

        if self.min_size is not None:
            size = max(size, self.min_size)
        if self.max_size is not None:
            size = min(size, self.max_size)

        return size

    def compute_buy_sizes(
        self,
        var_map: dict,
        price_map: dict,
        sharpe_map: dict,
        available_cash: float,
    ) -> dict:
        """
        Compute buy share counts (fractional) for multiple tickers.
        Normalize across tickers so total spend = available_cash * (1 - reserve_fraction).

        Returns: {ticker: shares_to_buy (float)}
        """
        # Calculate raw sizes and dollar demands
        raw_sizes = {}
        raw_dollars = {}
        for ticker, var in var_map.items():
            price = price_map[ticker]
            sharpe = sharpe_map[ticker]
            n_raw = self.compute_adjusted_size(var, price, sharpe)
            raw_sizes[ticker] = n_raw
            raw_dollars[ticker] = n_raw * price

        total_raw_dollar = sum(raw_dollars.values())
        target_capital = available_cash * (1 - self.reserve_fraction)

        # Determine scaling factor (<=1 to not exceed target_capital)
        scale = min(1.0, target_capital / total_raw_dollar) if total_raw_dollar > 0 else 0

        # Apply scaling and ensure affordability
        sizes = {}
        cash_used = 0.0
        for ticker, n_raw in raw_sizes.items():
            price = price_map[ticker]
            n_scaled = n_raw * scale
            # Ensure not exceeding cash by construction
            sizes[ticker] = n_scaled
            cash_used += n_scaled * price

        return sizes

    def compute_trade_sizes(
        self,
        signals: dict,
        var_map: dict,
        price_map: dict,
        sharpe_map: dict,
        holdings_map: dict,
        available_cash: float,
    ) -> dict:
        """
        Compute trade sizes for buy/sell/hold signals with fractional shares.
        Prioritize sell signals to free up cash before sizing buys.

        Returns: {ticker: delta_shares (float)}, positive = buy, negative = sell.
        """
        # === 1. Process sells first to free up cash ===
        sell_sizes = {}
        total_proceeds = 0.0
        for ticker, signal in signals.items():
            if signal.lower() == 'sell':
                var = var_map.get(ticker, 0)
                price = price_map.get(ticker, 0)
                sharpe = sharpe_map.get(ticker, 0)
                held = holdings_map.get(ticker, 0)
                desired = self.compute_adjusted_size(var, price, sharpe)
                size = min(desired, held)
                sell_sizes[ticker] = size
                total_proceeds += size * price

        # Update cash with proceeds from sells
        cash_after_sells = available_cash + total_proceeds

        # === 2. Process buys with updated cash ===
        buy_signals = {t: sig for t, sig in signals.items() if sig.lower() == 'buy'}
        buy_sizes = self.compute_buy_sizes(
            var_map={t: var_map[t] for t in buy_signals},
            price_map={t: price_map[t] for t in buy_signals},
            sharpe_map={t: sharpe_map[t] for t in buy_signals},
            available_cash=cash_after_sells
        )

        # === 3. Combine buy and sell deltas ===
        trade_sizes = {}
        for ticker in signals:
            sig = signals[ticker].lower()
            if sig == 'sell':
                trade_sizes[ticker] = -sell_sizes.get(ticker, 0.0)
            elif sig == 'buy':
                trade_sizes[ticker] = buy_sizes.get(ticker, 0.0)
            else:
                trade_sizes[ticker] = 0.0

        return trade_sizes

    def compute_sizes_from_series(
            self,
            signals: pd.Series,
            close_prices: pd.Series,
            var_series: pd.Series,
            sharpe_series: pd.Series,
            holdings: pd.Series,
    ) -> pd.Series:
        """
        Convenience wrapper that accepts pandas Series inputs (indexed by ticker) for
        signals ('buy','sell','hold'), close prices, daily VaR, Sharpe, and holdings,
        and returns a pandas Series of trade size deltas without cash constraints.
        """
        # Convert Series to dictionaries
        signals_dict = signals.to_dict()
        price_map = close_prices.to_dict()
        var_map = var_series.to_dict()
        sharpe_map = sharpe_series.to_dict()
        holdings_map = holdings.to_dict()

        trade_sizes = {}
        for ticker, signal in signals_dict.items():
            var = var_map.get(ticker, 0.0)
            price = price_map.get(ticker, 0.0)
            sharpe = sharpe_map.get(ticker, 0.0)
            held = holdings_map.get(ticker, 0.0)
            desired = self.compute_adjusted_size(var, price, sharpe)

            if signal.lower() == 'buy':
                trade_sizes[ticker] = desired
            elif signal.lower() == 'sell':
                trade_sizes[ticker] = desired
            else:
                trade_sizes[ticker] = 0.0

        # Return results aligned to original signals index
        return pd.Series(trade_sizes).reindex(signals.index).fillna(0.0)