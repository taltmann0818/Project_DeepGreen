from pandas import DataFrame
from Components.Fundamentals import search_line_items, get_metric_value
import pandas as pd

class CharlieMungerAgent:
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = "Charlie Munger"
        self.metrics = metrics
        self.ticker = ticker
        self.period = kwargs.get('analysis_period')
        self.limit = kwargs.get('analysis_limit')
        self.SIC_code = self.metrics['4digit_SIC_code'][0] if self.metrics['2digit_SIC_code'][0] == '73' else self.metrics['2digit_SIC_code'][0]
        if len(self.SIC_code) > 2:
            self.threshold_matrix = pd.read_csv(kwargs.get('threshold_matrix_path',None).get('business_services_sic'))
        else:
            self.threshold_matrix = pd.read_csv(kwargs.get('threshold_matrix_path',None).get('two_digit_sic'))

        self.analysis_data = {} # Storing returned results in dict

    def analyze(self):
        financial_line_items = search_line_items(
            self.ticker,
            [
                "revenue",
                "net_income",
                "operating_income",
                "return_on_invested_capital",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "debt_to_equity",
                "outstanding_shares",
                "research_and_development",
                "goodwill_and_intangible_assets",
                "market_cap"
            ],
            period=self.period,
            limit=self.limit,  # Munger examines long-term trends
            df=self.metrics
        )
        moat_analysis = self.analyze_moat_strength(financial_line_items)
        management_analysis = self.analyze_management_quality(financial_line_items, None)
        predictability_analysis = self.analyze_predictability(financial_line_items)
        valuation_analysis = self.calculate_munger_valuation(financial_line_items)

        # Combine partial scores with Munger's weighting preferences
        # Munger weights quality and predictability higher than current valuation
        total_score = (
            moat_analysis["score"] * 0.35 +
            management_analysis["score"] * 0.25 +
            predictability_analysis["score"] * 0.25 +
            valuation_analysis["score"] * 0.15
        )

        max_possible_score = 10  # Scale to 0-10

        # Generate a simple buy/hold/sell signal
        if total_score >= 7.5:  # Munger has very high standards
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        self.analysis_data = {
            "name": self.agent_name,
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
            "valuation_analysis": valuation_analysis,
            # Include some qualitative assessment from news
            "news_sentiment": "No news data available"
        }

        return self.analysis_data


    def analyze_moat_strength(self, financial_line_items: DataFrame):
        """
        Analyze the business's competitive advantage using Munger's approach:
        - Consistent high returns on capital (ROIC)
        - Pricing power (stable/improving gross margins)
        - Low capital requirements
        - Network effects and intangible assets (R&D investments, goodwill)
        """
        score = 0
        details = []

        if financial_line_items.empty:
            return {
                "score": 0,
                "details": "Insufficient data to analyze moat strength"
            }

        # 1. Return on Invested Capital (ROIC) analysis - Munger's favorite metric
        roic_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'return_on_invested_capital')
        roic_values = financial_line_items.return_on_invested_capital.values
        if roic_values is not None:
            # Check if ROIC consistently above base 15% (Munger's threshold)
            high_roic_count = sum(1 for r in roic_values if r > roic_threshold)
            if high_roic_count >= len(roic_values) * 0.8:  # 80% of periods show high ROIC
                score += 3
                details.append(f"Excellent ROIC: >{roic_threshold*100}% in {high_roic_count}/{len(roic_values)} periods")
            elif high_roic_count >= len(roic_values) * 0.5:  # 50% of periods
                score += 2
                details.append(f"Good ROIC: >{roic_threshold*100}% in {high_roic_count}/{len(roic_values)} periods")
            elif high_roic_count > 0:
                score += 1
                details.append(f"Mixed ROIC: >{roic_threshold*100}% in only {high_roic_count}/{len(roic_values)} periods")
            else:
                details.append(f"Poor ROIC: Never exceeds {roic_threshold*100}% threshold")
        else:
            details.append("No ROIC data available")

        # 2. Pricing power - check gross margin stability and trends
        gross_margins = financial_line_items.gross_margin.values
        gross_margins_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'gross_margin')
        if gross_margins is not None and len(gross_margins) >= 3:
            # Munger likes stable or improving gross margins
            margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
            if margin_trend >= len(gross_margins) * 0.7:  # Improving in 70% of periods
                score += 2
                details.append("Strong pricing power: Gross margins consistently improving")
            elif sum(gross_margins) / len(gross_margins) > gross_margins_threshold:  # Average margin > 30%
                score += 1
                details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
            else:
                details.append("Limited pricing power: Low or declining gross margins")
        else:
            details.append("Insufficient gross margin data")


        # 3. Capital intensity - Munger prefers low capex businesses
        if len(financial_line_items) >= 3:
            capex = financial_line_items.capital_expenditure.values
            revenue = financial_line_items.revenue.values
            if capex is not None and revenue is not None:
                capex_to_revenue = abs(capex) / revenue
            else:
                capex_to_revenue = None

            if capex_to_revenue is not None:
                avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
                if avg_capex_ratio < 0.05:  # Less than 5% of revenue
                    score += 2
                    details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
                elif avg_capex_ratio < 0.10:  # Less than 10% of revenue
                    score += 1
                    details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
                else:
                    details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append("No capital expenditure data available")
        else:
            details.append("Insufficient data for capital intensity analysis")

        # 4. Intangible assets - Munger values R&D and intellectual property
        r_and_d = financial_line_items.research_and_development.values
        goodwill_and_intangible_assets = financial_line_items.goodwill_and_intangible_assets.values
        if r_and_d is not None and len(r_and_d) > 0:
            if sum(r_and_d) > 0:  # If company is investing in R&D
                score += 1
                details.append("Invests in R&D, building intellectual property")
        if goodwill_and_intangible_assets is not None and len(goodwill_and_intangible_assets) > 0:
            score += 1
            details.append("Significant goodwill/intangible assets, suggesting brand value or IP")

        # Scale score to 0-10 range
        final_score = min(10, score * 10 / 9)  # Max possible raw score is 9

        return {
            "score": final_score,
            "details": "; ".join(details)
        }

    def analyze_management_quality(self, financial_line_items: DataFrame, insider_trades):
        """
        Evaluate management quality using Munger's criteria:
        - Capital allocation wisdom
        - Insider ownership and transactions
        - Cash management efficiency
        - Candor and transparency
        - Long-term focus
        """
        score = 0
        details = []

        if financial_line_items.empty:
            return {
                "score": 0,
                "details": "Insufficient data to analyze management quality"
            }

        # 1. Capital allocation - Check FCF to net income ratio
        # Munger values companies that convert earnings to cash
        fcf_values = financial_line_items.free_cash_flow.values
        net_income_values = financial_line_items.net_income.values
        if fcf_values is not None and net_income_values is not None and len(fcf_values) == len(net_income_values):
            # Calculate FCF to Net Income ratio for each period
            fcf_to_ni_ratios = []
            for i in range(len(fcf_values)):
                if net_income_values[i] and net_income_values[i] != 0:
                    fcf_to_ni_ratios.append(fcf_values[i] / net_income_values[i])

            if fcf_to_ni_ratios:
                avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
                if avg_ratio > 1.1:  # FCF > net income suggests good accounting
                    score += 3
                    details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                elif avg_ratio > 0.9:  # FCF roughly equals net income
                    score += 2
                    details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                elif avg_ratio > 0.7:  # FCF somewhat lower than net income
                    score += 1
                    details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                else:
                    details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
            else:
                details.append("Could not calculate FCF to Net Income ratios")
        else:
            details.append("Missing FCF or Net Income data")

        # 2. Debt management - Munger is cautious about debt
        de_ratio_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'debt_to_equity')
        de_ratio_threshold_low = de_ratio_threshold - (de_ratio_threshold * 0.4)
        de_ratio_threshold_mod = (de_ratio_threshold * 0.4) + de_ratio_threshold
        de_ratio_threshold_high = (de_ratio_threshold * 2) + de_ratio_threshold

        recent_de_ratio = financial_line_items.debt_to_equity.values[0]
        if recent_de_ratio:
            if recent_de_ratio < de_ratio_threshold_low:  # Very low debt
                score += 3
                details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
            elif recent_de_ratio < de_ratio_threshold_mod:  # Moderate debt
                score += 2
                details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
            elif recent_de_ratio < de_ratio_threshold_high:  # Higher but still reasonable debt
                score += 1
                details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
            else:
                details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append("Missing debt or equity data")

        # 3. Cash management efficiency - Munger values appropriate cash levels
        cash_values = financial_line_items.cash_and_equivalents.values
        revenue_values = financial_line_items.revenue.values
        if cash_values is not None and revenue_values is not None and len(cash_values) > 0 and len(revenue_values) > 0:
            # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
            cash_to_revenue = cash_values[0] / revenue_values[0] if revenue_values[0] != 0 else 0

            if 0.1 <= cash_to_revenue <= 0.25:
                # Goldilocks zone - not too much, not too little
                score += 2
                details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
                # Reasonable but not ideal
                score += 1
                details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            elif cash_to_revenue > 0.4:
                # Too much cash - potentially inefficient capital allocation
                details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            else:
                # Too little cash - potentially risky
                details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            details.append("Insufficient cash or revenue data")

        # 4. Insider activity - Munger values skin in the game
        if insider_trades and len(insider_trades) > 0:
            # Count buys vs. sells
            buys = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and
                       trade.transaction_type and trade.transaction_type.lower() in ['buy', 'purchase'])
            sells = sum(1 for trade in insider_trades if hasattr(trade, 'transaction_type') and
                        trade.transaction_type and trade.transaction_type.lower() in ['sell', 'sale'])

            # Calculate the buy ratio
            total_trades = buys + sells
            if total_trades > 0:
                buy_ratio = buys / total_trades
                if buy_ratio > 0.7:  # Strong insider buying
                    score += 2
                    details.append(f"Strong insider buying: {buys}/{total_trades} transactions are purchases")
                elif buy_ratio > 0.4:  # Balanced insider activity
                    score += 1
                    details.append(f"Balanced insider trading: {buys}/{total_trades} transactions are purchases")
                elif buy_ratio < 0.1 and sells > 5:  # Heavy selling
                    score -= 1  # Penalty for excessive selling
                    details.append(f"Concerning insider selling: {sells}/{total_trades} transactions are sales")
                else:
                    details.append(f"Mixed insider activity: {buys}/{total_trades} transactions are purchases")
            else:
                details.append("No recorded insider transactions")
        else:
            details.append("No insider trading data available")
            score += 2

        # 5. Consistency in share count - Munger prefers stable/decreasing shares
        share_counts = financial_line_items.outstanding_shares.values
        if share_counts is not None and len(share_counts) >= 3:
            if share_counts[0] < share_counts[-1] * 0.95:  # 5%+ reduction in shares
                score += 2
                details.append("Shareholder-friendly: Reducing share count over time")
            elif share_counts[0] < share_counts[-1] * 1.05:  # Stable share count
                score += 1
                details.append("Stable share count: Limited dilution")
            elif share_counts[0] > share_counts[-1] * 1.2:  # >20% dilution
                score -= 1  # Penalty for excessive dilution
                details.append("Concerning dilution: Share count increased significantly")
            else:
                details.append("Moderate share count increase over time")
        else:
            details.append("Insufficient share count data")

        # Scale score to 0-10 range
        # Maximum possible raw score would be 12 (3+3+2+2+2)
        final_score = max(0, min(10, score * 10 / 12))

        return {
            "score": final_score,
            "details": "; ".join(details)
        }


    def analyze_predictability(self, financial_line_items: DataFrame):
        """
        Assess the predictability of the business - Munger strongly prefers businesses
        whose future operations and cashflows are relatively easy to predict.
        """
        score = 0
        details = []

        if financial_line_items.empty or len(financial_line_items) < 4:
            return {
                "score": 0,
                "details": "Insufficient data to analyze business predictability (need 4+ periods)"
            }

        # 1. Revenue stability and growth
        revenues = financial_line_items.revenue.values
        if revenues is not None and len(revenues) >= 4:
            # Calculate period-over-period (either YoY or QoQ) growth rates
            growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
            avg_growth = sum(growth_rates) / len(growth_rates)
            growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)

            revenue_growth_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'revenue_growth_qoq')
            revenue_growth_threshold_steady = (abs(revenue_growth_threshold)*0.5) + revenue_growth_threshold

            if avg_growth > revenue_growth_threshold_steady and growth_volatility < 0.1:
                # Steady, consistent growth (Munger loves this)
                score += 3
                details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
            elif avg_growth > revenue_growth_threshold and growth_volatility < 0.2:
                # Positive but somewhat volatile growth
                score += 2
                details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
            elif avg_growth > revenue_growth_threshold:
                # Growing but unpredictable
                score += 1
                details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
            else:
                details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
        else:
            details.append("Insufficient revenue history for predictability analysis")

        # 2. Operating income stability
        op_income = financial_line_items.operating_income.values
        if op_income is not None and len(op_income) >= 5:
            # Count positive operating income periods
            positive_periods = sum(1 for income in op_income if income > 0)
            if positive_periods == len(op_income):
                # Consistently profitable operations
                score += 3
                details.append("Highly predictable operations: Operating income positive in all periods")
            elif positive_periods >= len(op_income) * 0.8:
                # Mostly profitable operations
                score += 2
                details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
            elif positive_periods >= len(op_income) * 0.6:
                # Somewhat profitable operations
                score += 1
                details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(op_income)} periods")
            else:
                details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(op_income)} periods")
        else:
            details.append("Insufficient operating income history")

        # 3. Margin consistency - Munger values stable margins
        op_margins = financial_line_items.operating_margin.values
        if op_margins is not None and len(op_margins) >= 4:
            # Calculate margin volatility
            avg_margin = sum(op_margins) / len(op_margins)
            margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)

            if margin_volatility < 0.03:  # Very stable margins
                score += 2
                details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
            elif margin_volatility < 0.07:  # Moderately stable margins
                score += 1
                details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
            else:
                details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
        else:
            details.append("Insufficient margin history")

        # 4. Cash generation reliability
        fcf_values = financial_line_items.free_cash_flow.values
        if fcf_values is not None and len(fcf_values) >= 4:
            # Count positive FCF periods
            positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
            if positive_fcf_periods == len(fcf_values):
                # Consistently positive FCF
                score += 2
                details.append("Highly predictable cash generation: Positive FCF in all periods")
            elif positive_fcf_periods >= len(fcf_values) * 0.8:
                # Mostly positive FCF
                score += 1
                details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
            else:
                details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
        else:
            details.append("Insufficient free cash flow history")

        # Scale score to 0-10 range
        # Maximum possible raw score would be 10 (3+3+2+2)
        final_score = min(10, score * 10 / 10)

        return {
            "score": final_score,
            "details": "; ".join(details)
        }


    def calculate_munger_valuation(self, financial_line_items: DataFrame):
        """
        Calculate intrinsic value using Munger's approach:
        - Focus on owner earnings (approximated by FCF)
        - Simple multiple on normalized earnings
        - Prefer paying a fair price for a wonderful business
        """
        score = 0
        details = []

        if financial_line_items.empty or not financial_line_items.market_cap.values.any():
            return {
                "score": score,
                "details": "Insufficient data to perform valuation"
            }

        # Get FCF values (Munger's preferred "owner earnings" metric)
        fcf_values = financial_line_items.free_cash_flow.values
        market_cap = financial_line_items.market_cap.values[0]
        if fcf_values is None or len(fcf_values) < 3:
            return {
                "score": 0,
                "details": "Insufficient free cash flow data for valuation"
            }

        # 1. Normalize earnings by taking average of last 3-5 years
        # (Munger prefers to normalize earnings to avoid over/under-valuation based on cyclical factors)
        normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
        if normalized_fcf == 0:
            return {
                "score": 0,
                "details": f"Negative or zero normalized FCF ({normalized_fcf}), cannot value",
                "intrinsic_value": None
            }

        # 2. Calculate FCF yield (inverse of P/FCF multiple)
        fcf_yield = normalized_fcf / market_cap

        # 3. Apply Munger's FCF multiple based on business quality
        # Munger would pay higher multiples for wonderful businesses
        # Let's use a sliding scale where higher FCF yields are more attractive
        if fcf_yield > 0.08:  # >8% FCF yield (P/FCF < 12.5x)
            score += 4
            details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.05:  # >5% FCF yield (P/FCF < 20x)
            score += 3
            details.append(f"Good value: {fcf_yield:.1%} FCF yield")
        elif fcf_yield > 0.03:  # >3% FCF yield (P/FCF < 33x)
            score += 1
            details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
        else:
            details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

        # 4. Calculate simple intrinsic value range
        # Munger tends to use straightforward valuations, avoiding complex DCF models
        conservative_value = normalized_fcf * 10  # 10x FCF = 10% yield
        reasonable_value = normalized_fcf * 15    # 15x FCF ≈ 6.7% yield
        optimistic_value = normalized_fcf * 20    # 20x FCF = 5% yield

        # 5. Calculate margins of safety
        current_to_reasonable = (reasonable_value - market_cap) / market_cap

        if current_to_reasonable > 0.3:  # >30% upside
            score += 3
            details.append(f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        elif current_to_reasonable > 0.1:  # >10% upside
            score += 2
            details.append(f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
        elif current_to_reasonable > -0.1:  # Within 10% of reasonable value
            score += 1
            details.append(f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})")
        else:
            details.append(f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value")

        # 6. Check earnings trajectory for additional context
        # Munger likes growing owner earnings
        if len(fcf_values) >= 3:
            recent_avg = sum(fcf_values[:3]) / 3
            older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]

            if recent_avg > older_avg * 1.2:  # >20% growth in FCF
                score += 3
                details.append("Growing FCF trend adds to intrinsic value")
            elif recent_avg > older_avg:
                score += 2
                details.append("Stable to growing FCF supports valuation")
            else:
                details.append("Declining FCF trend is concerning")

        # Scale score to 0-10 range
        # Maximum possible raw score would be 10 (4+3+3)
        final_score = min(10, score * 10 / 10)

        return {
            "score": final_score,
            "details": "; ".join(details),
            "intrinsic_value_range": {
                "conservative": conservative_value,
                "reasonable": reasonable_value,
                "optimistic": optimistic_value
            },
            "fcf_yield": fcf_yield,
            "normalized_fcf": normalized_fcf
        }