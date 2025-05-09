import statistics
from pandas import DataFrame
from Components.Fundamentals import search_line_items, get_metric_value
import pandas as pd

class PhilFisherAgent():
    """
    Analyzes stocks using Phil Fisher's investing principles:
      - Seek companies with long-term above-average growth potential
      - Emphasize quality of management and R&D
      - Look for strong margins, consistent growth, and manageable leverage
      - Combine fundamental 'scuttlebutt' style checks with basic sentiment and insider data
      - Willing to pay up for quality, but still mindful of valuation
      - Generally focuses on long-term compounding

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = 'Phil Fisher'
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
        # Include relevant line items for Phil Fisher's approach:
        #   - Growth & Quality: revenue, net_income, earnings_per_share, R&D expense
        #   - Margins & Stability: operating_income, operating_margin, gross_margin
        #   - Management Efficiency & Leverage: total_debt, shareholders_equity, free_cash_flow
        #   - Valuation: net_income, free_cash_flow (for P/E, P/FCF), ebit, ebitda
        financial_line_items = search_line_items(
            self.ticker,
            [
                "revenue",
                "net_income",
                "earnings_per_share",
                "free_cash_flow",
                "research_and_development",
                "operating_income",
                "operating_margin",
                "gross_margin",
                "total_debt",
                "shareholders_equity",
                "cash_and_equivalents",
                "ebit",
                "ebitda",
                "market_cap",
                "return_on_equity",
                "debt_to_equity",
            ],
            period=self.period,
            limit=self.limit,
            df=self.metrics
        )
        
        # Perform sub-analyses
        growth_quality = self.analyze_fisher_growth_quality(financial_line_items)
        margins_stability = self.analyze_margins_stability(financial_line_items)
        mgmt_efficiency = self.analyze_management_efficiency_leverage(financial_line_items)
        fisher_valuation = self.analyze_fisher_valuation(financial_line_items)
        insider_activity = self.analyze_insider_activity(None)
        sentiment_analysis = self.analyze_sentiment(None)

        # Combine partial scores with weights typical for Fisher:
        #   30% Growth & Quality
        #   25% Margins & Stability
        #   20% Management Efficiency
        #   15% Valuation
        #   5% Insider Activity
        #   5% Sentiment
        insider_activity["score"] = 5 # Setting to default due to missing data
        sentiment_analysis["score"] = 5
        total_score = (
            growth_quality["score"] * 0.30
            + margins_stability["score"] * 0.25
            + mgmt_efficiency["score"] * 0.20
            + fisher_valuation["score"] * 0.15
            + insider_activity["score"] * 0.05
            + sentiment_analysis["score"] * 0.05
        )

        max_possible_score = 10

        # Simple bullish/neutral/bearish signal
        if total_score >= 7.5:
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
            "growth_quality": growth_quality,
            "margins_stability": margins_stability,
            "management_efficiency": mgmt_efficiency,
            "valuation_analysis": fisher_valuation,
            "insider_activity": insider_activity,
            "sentiment_analysis": sentiment_analysis,
        }

        return self.analysis_data


    def analyze_fisher_growth_quality(self, financial_line_items: DataFrame):
        """
        Evaluate growth & quality:
          - Consistent Revenue Growth
          - Consistent EPS Growth
          - R&D as a % of Revenue (if relevant, indicative of future-oriented spending)
        """
        if financial_line_items.empty or len(financial_line_items) < 2:
            return {
                "score": 0,
                "details": "Insufficient financial data for growth/quality analysis",
            }

        details = []
        raw_score = 0  # up to 9 raw points => scale to 0–10

        # 1. Revenue Growth (Period over Period: YoY or QoQ)
        revenues = financial_line_items.revenue.values
        rev_growth_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'revenue_growth_qoq')
        rev_growth_threshold_mod = (rev_growth_threshold*3) + rev_growth_threshold
        rev_growth_threshold_strong = (rev_growth_threshold*8) + rev_growth_threshold
        if len(revenues) >= 2:
            # We'll look at the earliest vs. latest to gauge multi-year growth if possible
            latest_rev, oldest_rev = revenues[0], revenues[1]
            if oldest_rev > 0:
                rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
                if rev_growth > rev_growth_threshold_strong:
                    raw_score += 3
                    details.append(f"Very strong period-over-period revenue growth: {rev_growth:.1%}")
                elif rev_growth > rev_growth_threshold_mod:
                    raw_score += 2
                    details.append(f"Moderate period-over-period revenue growth: {rev_growth:.1%}")
                elif rev_growth > rev_growth_threshold:
                    raw_score += 1
                    details.append(f"Slight period-over-period revenue growth: {rev_growth:.1%}")
                else:
                    details.append(f"Minimal or negative period-over-period revenue growth: {rev_growth:.1%}")
            else:
                details.append("Oldest revenue is zero/negative; cannot compute growth.")
        else:
            details.append("Not enough revenue data points for growth calculation.")

        # 2. EPS Growth (Period over Period: YoY or QoQ)
        eps_values = financial_line_items.earnings_per_share.values
        eps_growth_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'eps_growth_qoq')
        eps_growth_threshold_mod = (eps_growth_threshold * 3) + eps_growth_threshold
        eps_growth_threshold_strong = (eps_growth_threshold * 8) + eps_growth_threshold
        if len(eps_values) >= 2:
            latest_eps, oldest_eps = eps_values[0], eps_values[1]
            if abs(oldest_eps) > 1e-9:
                eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
                if eps_growth > eps_growth_threshold_strong:
                    raw_score += 3
                    details.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
                elif eps_growth > eps_growth_threshold_mod:
                    raw_score += 2
                    details.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
                elif eps_growth > eps_growth_threshold:
                    raw_score += 1
                    details.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
                else:
                    details.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
            else:
                details.append("Oldest EPS near zero; skipping EPS growth calculation.")
        else:
            details.append("Not enough EPS data points for growth calculation.")

        # 3. R&D as % of Revenue (if we have R&D data)
        rnd_values = financial_line_items.research_and_development.values
        if rnd_values.any() and revenues.any() and len(rnd_values) == len(revenues):
            # We'll just look at the most recent for a simple measure
            recent_rnd = rnd_values[0]
            recent_rev = revenues[0] if revenues[0] else 1e-9
            rnd_ratio = recent_rnd / recent_rev
            # Generally, Fisher admired companies that invest aggressively in R&D,
            # but it must be appropriate. We'll assume "3%-12%" is healthy, just as an example.
            if 0.03 <= rnd_ratio <= 0.12:
                raw_score += 3
                details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
            elif rnd_ratio > 0.12:
                raw_score += 2
                details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
            elif rnd_ratio > 0.0:
                raw_score += 1
                details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
            else:
                details.append("No meaningful R&D expense ratio")
        else:
            details.append("Insufficient R&D data to evaluate")

        # scale raw_score (max 9) to 0–10
        final_score = min(10, (raw_score / 9) * 10)
        return {"score": final_score, "details": "; ".join(details)}


    def analyze_margins_stability(self, financial_line_items: DataFrame):
        """
        Looks at margin consistency (gross/operating margin) and general stability over time.
        """
        if financial_line_items.empty or len(financial_line_items) < 2:
            return {
                "score": 0,
                "details": "Insufficient data for margin stability analysis",
            }

        details = []
        raw_score = 0  # up to 6 => scale to 0-10

        # 1. Operating Margin Consistency
        op_margins = financial_line_items.operating_margin.values
        if len(op_margins) >= 2:
            # Check if margins are stable or improving (comparing oldest to newest)
            oldest_op_margin = op_margins[-1]
            newest_op_margin = op_margins[0]
            if newest_op_margin >= oldest_op_margin > 0:
                raw_score += 2
                details.append(f"Operating margin stable or improving ({oldest_op_margin:.1%} -> {newest_op_margin:.1%})")
            elif newest_op_margin > 0:
                raw_score += 1
                details.append(f"Operating margin positive but slightly declined")
            else:
                details.append(f"Operating margin may be negative or uncertain")
        else:
            details.append("Not enough operating margin data points")

        # 2. Gross Margin Level
        gm_values = financial_line_items.gross_margin.values
        gm_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'gross_margin')
        gm_threshold_mod = gm_threshold - (gm_threshold*0.4)
        if gm_values.any():
            # We'll just take the most recent
            recent_gm = gm_values[0]
            if recent_gm > gm_threshold:
                raw_score += 2
                details.append(f"Strong gross margin: {recent_gm:.1%}")
            elif recent_gm > gm_threshold_mod:
                raw_score += 1
                details.append(f"Moderate gross margin: {recent_gm:.1%}")
            else:
                details.append(f"Low gross margin: {recent_gm:.1%}")
        else:
            details.append("No gross margin data available")

        # 3. Multi-year Margin Stability
        #   e.g. if we have at least 3 data points, see if standard deviation is low.
        if len(op_margins) >= 3:
            stdev = statistics.pstdev(op_margins)
            if stdev < 0.02:
                raw_score += 2
                details.append("Operating margin extremely stable over multiple years")
            elif stdev < 0.05:
                raw_score += 1
                details.append("Operating margin reasonably stable")
            else:
                details.append("Operating margin volatility is high")
        else:
            details.append("Not enough margin data points for volatility check")

        # scale raw_score (max 6) to 0-10
        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}


    def analyze_management_efficiency_leverage(self, financial_line_items: DataFrame):
        """
        Evaluate management efficiency & leverage:
          - Return on Equity (ROE)
          - Debt-to-Equity ratio
          - Possibly check if free cash flow is consistently positive
        """
        if financial_line_items.empty:
            return {
                "score": 0,
                "details": "No financial data for management efficiency analysis",
            }

        details = []
        raw_score = 0  # up to 6 => scale to 0–10

        # 1. Return on Equity (ROE)
        roe = financial_line_items.return_on_equity.values[0]
        roe_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'return_on_equity')
        roe_threshold_high = roe_threshold + roe_threshold
        roe_threshold_low = roe_threshold - roe_threshold
        if roe > roe_threshold_high:
            raw_score += 3
            details.append(f"High ROE: {roe:.1%}")
        elif roe > roe_threshold:
            raw_score += 2
            details.append(f"Moderate ROE: {roe:.1%}")
        elif roe > roe_threshold_low:
            raw_score += 1
            details.append(f"Low ROE: {roe:.1%}")
        else:
            details.append(f"ROE is low for industry: {roe:.1%}")

        # 2. Debt-to-Equity
        dte = financial_line_items.debt_to_equity.values[0]
        dte_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'debt_to_equity')
        dte_threshold_low = dte_threshold - (dte_threshold*0.4)
        if dte < dte_threshold_low:
            raw_score += 2
            details.append(f"Low debt-to-equity: {dte:.2f}")
        elif dte < dte_threshold:
            raw_score += 1
            details.append(f"Manageable debt-to-equity: {dte:.2f}")
        else:
            details.append(f"High debt-to-equity: {dte:.2f}")

        # 3. FCF Consistency
        fcf_values = financial_line_items.free_cash_flow.values
        if fcf_values.any() and len(fcf_values) >= 2:
            # Check if FCF is positive in recent years
            positive_fcf_count = sum(1 for x in fcf_values if x and x > 0)
            # We'll be simplistic: if most are positive, reward
            ratio = positive_fcf_count / len(fcf_values)
            if ratio > 0.8:
                raw_score += 1
                details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(fcf_values)})")
            else:
                details.append(f"Free cash flow is inconsistent or often negative")
        else:
            details.append("Insufficient or no FCF data to check consistency")

        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}


    def analyze_fisher_valuation(self, financial_line_items: DataFrame):
        """
        Phil Fisher is willing to pay for quality and growth, but still checks:
          - P/E
          - P/FCF
          - (Optionally) Enterprise Value metrics, but simpler approach is typical
        We will grant up to 2 points for each of two metrics => max 4 raw => scale to 0–10.
        """
        if financial_line_items.empty:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        details = []
        raw_score = 0

        # Gather needed data
        net_incomes = financial_line_items.net_income.values
        fcf_values = financial_line_items.free_cash_flow.values
        market_cap = financial_line_items.market_cap.values

        # 1) P/E
        recent_net_income = net_incomes[0] if net_incomes.any() else None
        pe_ratio_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'price_to_earning_ratio')
        pe_ratio_threshold_low = pe_ratio_threshold - (pe_ratio_threshold*0.2)
        pe_ratio_threshold_high = pe_ratio_threshold + (pe_ratio_threshold*0.2)
        if recent_net_income and recent_net_income != 0:
            pe = market_cap[0] / recent_net_income
            pe_points = 0
            if pe < pe_ratio_threshold_low:
                pe_points = 2
                details.append(f"Reasonably attractive P/E: {pe:.2f}")
            elif pe < pe_ratio_threshold_high:
                pe_points = 1
                details.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
            else:
                details.append(f"Very high P/E: {pe:.2f}")
            raw_score += pe_points
        else:
            details.append("No positive net income for P/E calculation")

        # 2) P/FCF
        recent_fcf = fcf_values[0] if fcf_values.any() else None
        price_to_fcf_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'price_to_fcf')
        price_to_fcf_threshold_low = price_to_fcf_threshold + (price_to_fcf_threshold*0.33)
        price_to_fcf_threshold_high = price_to_fcf_threshold + (price_to_fcf_threshold*2)
        if recent_fcf and recent_fcf != 0:
            pfcf = market_cap[0] / recent_fcf
            pfcf_points = 0
            if pfcf < price_to_fcf_threshold_low:
                pfcf_points = 2
                details.append(f"Reasonable P/FCF: {pfcf:.2f}")
            elif pfcf < price_to_fcf_threshold_high:
                pfcf_points = 1
                details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
            else:
                details.append(f"Excessively high P/FCF: {pfcf:.2f}")
            raw_score += pfcf_points
        else:
            details.append("No positive free cash flow for P/FCF calculation")

        # scale raw_score (max 4) to 0–10
        final_score = min(10, (raw_score / 4) * 10)
        return {"score": final_score, "details": "; ".join(details)}


    def analyze_insider_activity(self, insider_trades: list) -> dict:
        """
        Simple insider-trade analysis:
          - If there's heavy insider buying, we nudge the score up.
          - If there's mostly selling, we reduce it.
          - Otherwise, neutral.
        """
        # Default is neutral (5/10).
        score = 5
        details = []

        if not insider_trades:
            details.append("No insider trades data; defaulting to neutral")
            return {"score": score, "details": "; ".join(details)}

        buys, sells = 0, 0
        for trade in insider_trades:
            if trade.transaction_shares is not None:
                if trade.transaction_shares > 0:
                    buys += 1
                elif trade.transaction_shares < 0:
                    sells += 1

        total = buys + sells
        if total == 0:
            details.append("No buy/sell transactions found; neutral")
            return {"score": score, "details": "; ".join(details)}

        buy_ratio = buys / total
        if buy_ratio > 0.7:
            score = 8
            details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.4:
            score = 6
            details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
        else:
            score = 4
            details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

        return {"score": score, "details": "; ".join(details)}


    def analyze_sentiment(self, news_items: list) -> dict:
        """
        Basic news sentiment: negative keyword check vs. overall volume.
        """
        if not news_items:
            return {"score": 5, "details": "No news data; defaulting to neutral sentiment"}

        negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
        negative_count = 0
        for news in news_items:
            title_lower = (news.title or "").lower()
            if any(word in title_lower for word in negative_keywords):
                negative_count += 1

        details = []
        if negative_count > len(news_items) * 0.3:
            score = 3
            details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
        elif negative_count > 0:
            score = 6
            details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
        else:
            score = 8
            details.append("Mostly positive/neutral headlines")

        return {"score": score, "details": "; ".join(details)}