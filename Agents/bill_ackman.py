from pandas import DataFrame
from Components.Fundamentals import search_line_items, get_metric_value
import pandas as pd

class BillAckmanAgent:
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = "Bill Ackman"
        self.metrics = metrics
        self.ticker = ticker
        self.period = kwargs.get('analysis_period')
        self.limit = kwargs.get('analysis_limit')
        self.SIC_code = self.metrics['4digit_SIC_code'][0] if self.metrics['2digit_SIC_code'][0] == '73' else self.metrics['2digit_SIC_code'][0]
        if len(self.SIC_code) > 2:
            self.threshold_matrix = pd.read_csv('Agents/Matrices/Fundamentals Matrix - 4digit SIC 73 - Business Services.csv')
        else:
            self.threshold_matrix = pd.read_csv('Agents/Matrices/Fundamentals Matrix - 2digit SIC.csv')

        self.analysis_data = {} # Storing returned results in dict

    def analyze(self):
        #metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = search_line_items(
            self.ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "return_on_equity",
                # Optional: intangible_assets if available
                "intangible_assets",
                "market_cap"
            ],
            period=self.period, 
            limit=self.limit,
            df=self.metrics
        )
        quality_analysis = self.analyze_business_quality(financial_line_items)
        balance_sheet_analysis = self.analyze_financial_discipline(financial_line_items)
        activism_analysis = self.analyze_activism_potential(financial_line_items)
        valuation_analysis = self.analyze_valuation(financial_line_items)

        # Combine partial scores or signals
        total_score = (
            quality_analysis["score"]
            + balance_sheet_analysis["score"]
            + activism_analysis["score"]
            + valuation_analysis["score"]
        )
        max_possible_score = 20  # Adjust weighting as desired (5 from each sub-analysis, for instance)

        # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        self.analysis_data = {
            "name": self.agent_name,
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis
        }

        return self.analysis_data


    def analyze_business_quality(self, financial_line_items: DataFrame):
        """
        Analyze whether the company has a high-quality business with stable or growing cash flows,
        durable competitive advantages (moats), and potential for long-term growth.
        Also tries to infer brand strength if intangible_assets data is present (optional).
        """
        score = 0
        details = []

        if financial_line_items.empty:
            return {
                "score": 0,
                "details": "Insufficient data to analyze business quality"
            }

        # 1. Multi-period revenue growth analysis
        revenue_growth_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'revenue_growth_qoq')
        revenue_growth_ackman_threshold = (revenue_growth_threshold*0.4) + revenue_growth_threshold
        revenues = financial_line_items.revenue.values
        if len(revenues) >= 2:
            initial, final = revenues[1], revenues[0]
            if initial and final and final > initial:
                growth_rate = (final - initial) / abs(initial)
                if growth_rate > revenue_growth_ackman_threshold:  # e.g., 40% baseline cumulative growth
                    score += 2
                    details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the prior period (strong growth).")
                else:
                    score += 1
                    details.append(f"Revenue growth is positive but under {(revenue_growth_threshold*100):.1f}% cumulatively ({(growth_rate*100):.1f}%).")
            else:
                details.append("Revenue did not grow significantly or data insufficient.")
        else:
            details.append("Not enough revenue data for multi-period trend.")

        # 2. Operating margin and free cash flow consistency

        op_margin_vals = financial_line_items.operating_margin.values
        op_margin_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'operating_margin')
        op_margin_ackman_threshold = (op_margin_threshold*.5)+op_margin_threshold
        if op_margin_vals is not None:
            above_15 = sum(1 for m in op_margin_vals if m > op_margin_ackman_threshold)
            if above_15 >= (len(op_margin_vals) // 2 + 1):
                score += 2
                details.append(f"Operating margins have often exceeded {op_margin_ackman_threshold:.1%} (indicates good profitability).")
            else:
                details.append(f"Operating margin not consistently above {op_margin_ackman_threshold:.1%}.")
        else:
            details.append("No operating margin data across periods.")

        fcf_vals = financial_line_items.free_cash_flow.values
        if fcf_vals is not None:
            positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
            if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
                score += 1
                details.append("Majority of periods show positive free cash flow.")
            else:
                details.append("Free cash flow not consistently positive.")
        else:
            details.append("No free cash flow data across periods.")

        # 3. Return on Equity (ROE) check from the latest metrics
        return_on_equity = financial_line_items.return_on_equity.values[0]
        roe_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'return_on_equity')
        roe_ackman_threshold = (roe_threshold*.5)+roe_threshold
        if return_on_equity and return_on_equity > roe_ackman_threshold:
            score += 2
            details.append(f"High ROE of {return_on_equity:.1%}, indicating a competitive advantage.")
        elif return_on_equity:
            details.append(f"ROE of {return_on_equity:.1%} is moderate.")
        else:
            details.append("ROE data not available.")

        # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
        intangible_vals = financial_line_items.intangible_assets.values if financial_line_items.intangible_assets.values.any() else None
        if intangible_vals is not None and sum(intangible_vals) > 0:
            details.append("Significant intangible assets may indicate brand value or proprietary tech.")
            score += 1

        return {
            "score": score,
            "details": "; ".join(details)
        }


    def analyze_financial_discipline(self, financial_line_items: DataFrame):
        """
        Evaluate the company's balance sheet over multiple periods:
        - Debt ratio trends
        - Capital returns to shareholders over time (dividends, buybacks)
        """
        score = 0
        details = []

        if financial_line_items.empty:
            return {
                "score": 0,
                "details": "Insufficient data to analyze financial discipline"
            }

        # 1. Multi-period debt ratio or debt_to_equity
        debt_to_equity_vals = financial_line_items.debt_to_equity.values if financial_line_items.debt_to_equity.values.any() else None
        dte_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'debt_to_equity')
        dte_ackman_threshold = (dte_threshold*1.0)+dte_threshold
        if debt_to_equity_vals is not None:
            below_one_count = sum(1 for d in debt_to_equity_vals if d < dte_ackman_threshold)
            if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
                score += 2
                details.append(f"Debt-to-equity < {dte_ackman_threshold} for the majority of periods (reasonable leverage).")
            else:
                details.append(f"Debt-to-equity >= {dte_ackman_threshold} in many periods (could be high leverage).")
        else:
            details.append("No consistent leverage ratio data available.")

        # 2. Capital allocation approach (dividends + share counts)
        dividends_list = financial_line_items.dividends_and_other_cash_distributions.values if financial_line_items.dividends_and_other_cash_distributions.values.any() else None
        if dividends_list is not None:
            paying_dividends_count = sum(1 for d in dividends_list if d < 0)
            if paying_dividends_count >= (len(dividends_list) // 2 + 1):
                score += 1
                details.append("Company has a history of returning capital to shareholders (dividends).")
            else:
                details.append("Dividends not consistently paid or no data on distributions.")
        else:
            details.append("No dividend data found across periods.")

        # Check for decreasing share count (simple approach)
        shares = financial_line_items.outstanding_shares.values if financial_line_items.outstanding_shares.values.any() else None
        if len(shares) >= 2:
            if shares[1] < shares[0]:
                score += 1
                details.append("Outstanding shares have decreased compared to previous period (possible buybacks).")
            else:
                details.append("Outstanding shares have not decreased compared to previous period.")
        else:
            details.append("No multi-period share count data to assess buybacks.")

        return {
            "score": score,
            "details": "; ".join(details)
        }


    def analyze_activism_potential(self, financial_line_items: DataFrame):
        """
        Bill Ackman often engages in activism if a company has a decent brand or moat
        but is underperforming operationally.

        We'll do a simplified approach:
        - Look for positive revenue trends but subpar margins
        - That may indicate 'activism upside' if operational improvements could unlock value.
        """
        if financial_line_items.empty:
            return {
                "score": 0,
                "details": "Insufficient data for activism potential"
            }

        # Check revenue growth vs. operating margin
        revenues = financial_line_items.revenue.values
        op_margins = financial_line_items.operating_margin.values

        if len(revenues) < 2 or not op_margins.any():
            return {
                "score": 0,
                "details": "Not enough data to assess activism potential (need multi-year revenue + margins)."
            }

        initial, final = revenues[1], revenues[0]
        revenue_growth = (final - initial) / abs(initial) if initial else 0
        avg_margin = sum(op_margins) / len(op_margins)

        score = 0
        details = []

        # Suppose if there's decent revenue growth but margins are below 10%, Ackman might see activism potential.
        op_margin_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'operating_margin')
        if revenue_growth > 0.15 and avg_margin < op_margin_threshold:
            score += 2
            details.append(
                f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low (avg {avg_margin*100:.1f}%). "
                "Activism could unlock margin improvements."
            )
        else:
            details.append("No clear sign of activism opportunity (either margins are already decent or growth is weak).")

        return {"score": score, "details": "; ".join(details)}


    def analyze_valuation(self, financial_line_items: DataFrame):
        """
        Ackman invests in companies trading at a discount to intrinsic value.
        Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
        """
        if financial_line_items.empty or not financial_line_items.market_cap.values.any():
            return {
                "score": 0,
                "details": "Insufficient data to perform valuation"
            }

        fcf = financial_line_items.free_cash_flow.values[0] if financial_line_items.free_cash_flow.values.any() else 0
        market_cap = financial_line_items.market_cap.values[0]

        if fcf == 0:
            return {
                "score": 0,
                "details": f"No positive FCF for valuation; FCF = {fcf}",
                "intrinsic_value": None
            }

        # Basic DCF assumptions
        growth_rate = 0.06
        discount_rate = 0.10
        terminal_multiple = 15
        projection_years = 5

        present_value = 0
        for year in range(1, projection_years + 1):
            future_fcf = fcf * (1 + growth_rate) ** year
            pv = future_fcf / ((1 + discount_rate) ** year)
            present_value += pv

        # Terminal Value
        terminal_value = (
            fcf * (1 + growth_rate) ** projection_years * terminal_multiple
        ) / ((1 + discount_rate) ** projection_years)

        intrinsic_value = present_value + terminal_value
        margin_of_safety = (intrinsic_value - market_cap) / market_cap

        score = 0
        # Simple scoring
        if margin_of_safety > 0.3:
            score += 3
        elif margin_of_safety > 0.1:
            score += 1

        details = [
            f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
            f"Market cap: ~{market_cap:,.2f}",
            f"Margin of safety: {margin_of_safety:.2%}"
        ]

        return {
            "score": score,
            "details": "; ".join(details),
            "intrinsic_value": intrinsic_value,
            "margin_of_safety": margin_of_safety
        }