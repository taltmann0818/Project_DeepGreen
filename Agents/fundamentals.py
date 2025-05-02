from Components.Fundamentals import search_line_items
from pandas import DataFrame

class FundamentalsAgent:
    """Analyzes fundamental data and generates trading signals for a ticker."""
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = 'Fundamentals'
        self.analysis_data = {}
        self.metrics = metrics
        self.ticker = ticker

        self.period = kwargs.get('analysis_period','FY')
        self.limit = kwargs.get('analysis_limit',10)

    def analyze(self):
        # Get the financial metrics
        financial_line_items = search_line_items(
            self.ticker,
            [
                "return_on_equity",
                "net_margin",
                "operating_margin",
                "revenue",
                "earnings_per_share",
                "book_value",
                "current_ratio",
                "debt_to_equity",
                "free_cash_flow_per_share",
                "price_to_book_ratio",
                "market_cap",
                "net_income",
            ],
            period=self.period,
            limit=self.limit,
            df=self.metrics
        )
        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        # 1. Profitability Analysis
        return_on_equity = financial_line_items.return_on_equity.values[0]
        net_margin = financial_line_items.return_on_equity.values[0]
        operating_margin = financial_line_items.return_on_equity.values[0]

        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
        ]
        profitability_score = sum(metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity != 0 else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin != 0 else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }
        
        # 2. Growth Analysis
        revenues = financial_line_items.revenue.values
        revenue_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        earnings = financial_line_items.earnings_per_share.values
        earnings_growth = (earnings[0] - earnings[-1]) / abs(earnings[-1])
        book_values = financial_line_items.book_value.values
        book_value_growth = (book_values[0] - book_values[-1]) / abs(book_values[-1])

        thresholds = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
            (book_value_growth, 0.10),  # 10% book value growth
        ]
        growth_score = sum(metric > threshold for metric, threshold in thresholds)
        
        signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth != 0 else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth != 0 else "Earnings Growth: N/A"),
        }

        # 3. Financial Health
        current_ratio = financial_line_items.current_ratio.values[0]
        debt_to_equity = financial_line_items.debt_to_equity.values[0]
        free_cash_flow_per_share = financial_line_items.free_cash_flow_per_share.values[0]

        health_score = 0
        if current_ratio != 0 and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity != 0 and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if free_cash_flow_per_share != 0 and earnings[0] != 0 and free_cash_flow_per_share > earnings[0] * 0.8:  # Strong FCF conversion
            health_score += 1

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio != 0 else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity != 0 else "D/E: N/A"),
        }
        
        # 4. Price to X ratios
        pe_ratio = financial_line_items.market_cap.values[0] / financial_line_items.net_income.values[0]
        pb_ratio = financial_line_items.price_to_book_ratio.values[0]
        ps_ratio = financial_line_items.market_cap.values[0] / financial_line_items.revenue.values[0]

        thresholds = [
            (pe_ratio, 25),  # Reasonable P/E ratio
            (pb_ratio, 3),  # Reasonable P/B ratio
            (ps_ratio, 5),  # Reasonable P/S ratio
        ]
        price_ratio_score = sum(metric > threshold for metric, threshold in thresholds)

        signals.append("bearish" if price_ratio_score >= 2 else "bullish" if price_ratio_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio != 0 else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio != 0 else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio != 0 else "P/S: N/A"),
        }

        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level
        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        self.analysis_data = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        return self.analysis_data