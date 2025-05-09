import statistics
from pandas import DataFrame
from Components.Fundamentals import search_line_items, get_metric_value
import pandas as pd

class StanleyDruckenmillerAgent():
    """
    Analyzes stocks using Stanley Druckenmiller's investing principles:
      - Seeking asymmetric risk-reward opportunities
      - Emphasizing growth, momentum, and sentiment
      - Willing to be aggressive if conditions are favorable
      - Focus on preserving capital by avoiding high-risk, low-reward bets

    Returns a bullish/bearish/neutral signal with confidence and reasoning.
    """
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = "Stanley Druckenmiller"
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
        # Include relevant line items for Stan Druckenmiller's approach:
        #   - Growth & momentum: revenue, EPS, operating_income, ...
        #   - Valuation: net_income, free_cash_flow, ebit, ebitda
        #   - Leverage: total_debt, shareholders_equity
        #   - Liquidity: cash_and_equivalents
        financial_line_items = search_line_items(
            self.ticker,
            [
                "revenue",
                "earnings_per_share",
                "net_income",
                "operating_income",
                "gross_margin",
                "operating_margin",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "outstanding_shares",
                "enterprise_value",
                "ebit",
                "ebitda",
                "market_cap",
                "debt_to_equity",
            ],
            period=self.period,
            limit=self.limit,
            df=self.metrics
        )
        growth_momentum_analysis = self.analyze_growth_and_momentum(financial_line_items)
        sentiment_analysis = self.analyze_sentiment(DataFrame())
        insider_activity = self.analyze_insider_activity(DataFrame())
        risk_reward_analysis = self.analyze_risk_reward(financial_line_items)
        valuation_analysis = self.analyze_druckenmiller_valuation(financial_line_items)

        # Combine partial scores with weights typical for Druckenmiller:
        #   35% Growth/Momentum, 20% Risk/Reward, 20% Valuation,
        #   15% Sentiment, 10% Insider Activity = 100%
        total_score = (
            growth_momentum_analysis["score"] * 0.35
            + risk_reward_analysis["score"] * 0.20
            + valuation_analysis["score"] * 0.20
            + sentiment_analysis["score"] * 0.15
            + insider_activity["score"] * 0.10
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
            "growth_momentum_analysis": growth_momentum_analysis,
            "sentiment_analysis": sentiment_analysis,
            "insider_activity": insider_activity,
            "risk_reward_analysis": risk_reward_analysis,
            "valuation_analysis": valuation_analysis,
        }

        return self.analysis_data

    def analyze_growth_and_momentum(self, financial_line_items: DataFrame):
        """
        Evaluate:
          - Revenue Growth (YoY)
          - EPS Growth (YoY)
          - Price Momentum
        """
        if financial_line_items.empty or len(financial_line_items) < 2:
            return {"score": 0, "details": "Insufficient financial data for growth analysis"}

        details = []
        raw_score = 0  # We'll sum up a maximum of 9 raw points, then scale to 0–10

        #
        # 1. Revenue Growth
        #
        revenues = financial_line_items.revenue.values
        rev_growth_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'revenue_growth_qoq')
        rev_growth_threshold_strong = (rev_growth_threshold*2) + rev_growth_threshold
        rev_growth_threshold_mod = (rev_growth_threshold*0.5) + rev_growth_threshold
        rev_growth_threshold_slight = rev_growth_threshold - (rev_growth_threshold*0.5)
        if len(revenues) >= 2:
            latest_rev, older_rev = revenues[0], revenues[1]
            if older_rev > 0:
                rev_growth = (latest_rev - older_rev) / abs(older_rev)
                if rev_growth > rev_growth_threshold_strong:
                    raw_score += 3
                    details.append(f"Strong revenue growth: {rev_growth:.1%}")
                elif rev_growth > rev_growth_threshold_mod:
                    raw_score += 2
                    details.append(f"Moderate revenue growth: {rev_growth:.1%}")
                elif rev_growth > rev_growth_threshold_slight:
                    raw_score += 1
                    details.append(f"Slight revenue growth: {rev_growth:.1%}")
                else:
                    details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
            else:
                details.append("Older revenue is zero/negative; can't compute revenue growth.")
        else:
            details.append("Not enough revenue data points for growth calculation.")

        #
        # 2. EPS Growth
        #
        eps_values = financial_line_items.earnings_per_share.values
        eps_growth_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'eps_growth_qoq')
        eps_growth_threshold_strong = (eps_growth_threshold * 2) + eps_growth_threshold
        eps_growth_threshold_mod = (eps_growth_threshold * 0.5) + eps_growth_threshold
        eps_growth_threshold_slight = eps_growth_threshold - (eps_growth_threshold * 0.5)
        if len(eps_values) >= 2:
            latest_eps, older_eps = eps_values[0], eps_values[1]
            # Avoid division by zero
            if abs(older_eps) > 1e-9:
                eps_growth = (latest_eps - older_eps) / abs(older_eps)
                if eps_growth > eps_growth_threshold_strong:
                    raw_score += 3
                    details.append(f"Strong EPS growth: {eps_growth:.1%}")
                elif eps_growth > eps_growth_threshold_mod:
                    raw_score += 2
                    details.append(f"Moderate EPS growth: {eps_growth:.1%}")
                elif eps_growth > eps_growth_threshold_slight:
                    raw_score += 1
                    details.append(f"Slight EPS growth: {eps_growth:.1%}")
                else:
                    details.append(f"Minimal/negative EPS growth: {eps_growth:.1%}")
            else:
                details.append("Older EPS is near zero; skipping EPS growth calculation.")
        else:
            details.append("Not enough EPS data points for growth calculation.")

        #
        # 3. Price Momentum; NOT USING DUE TO DATA CONSTRAINTS
        details.append("Not enough recent price data for momentum analysis.")
        raw_score += 3

        # We assigned up to 3 points each for:
        #   revenue growth, eps growth, momentum
        # => max raw_score = 9
        # Scale to 0–10
        final_score = min(10, (raw_score / 9) * 10)

        return {"score": final_score, "details": "; ".join(details)}


    def analyze_insider_activity(self, insider_trades: DataFrame):
        """
        Simple insider-trade analysis:
          - If there's heavy insider buying, we nudge the score up.
          - If there's mostly selling, we reduce it.
          - Otherwise, neutral.
        """
        # Default is neutral (5/10).
        score = 5
        details = []

        if insider_trades.empty:
            details.append("No insider trades data; defaulting to neutral")
            return {"score": score, "details": "; ".join(details)}

        buys, sells = 0, 0
        for trade in insider_trades:
            # Use transaction_shares to determine if it's a buy or sell
            # Negative shares = sell, positive shares = buy
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
            # Heavy buying => +3 points from the neutral 5 => 8
            score = 8
            details.append(f"Heavy insider buying: {buys} buys vs. {sells} sells")
        elif buy_ratio > 0.4:
            # Moderate buying => +1 => 6
            score = 6
            details.append(f"Moderate insider buying: {buys} buys vs. {sells} sells")
        else:
            # Low insider buying => -1 => 4
            score = 4
            details.append(f"Mostly insider selling: {buys} buys vs. {sells} sells")

        return {"score": score, "details": "; ".join(details)}


    def analyze_sentiment(self, news_items: DataFrame):
        """
        Basic news sentiment: negative keyword check vs. overall volume.
        """
        if news_items.empty:
            return {"score": 5, "details": "No news data; defaulting to neutral sentiment"}

        negative_keywords = ["lawsuit", "fraud", "negative", "downturn", "decline", "investigation", "recall"]
        negative_count = 0
        for news in news_items:
            title_lower = (news.title or "").lower()
            if any(word in title_lower for word in negative_keywords):
                negative_count += 1

        details = []
        if negative_count > len(news_items) * 0.3:
            # More than 30% negative => somewhat bearish => 3/10
            score = 3
            details.append(f"High proportion of negative headlines: {negative_count}/{len(news_items)}")
        elif negative_count > 0:
            # Some negativity => 6/10
            score = 6
            details.append(f"Some negative headlines: {negative_count}/{len(news_items)}")
        else:
            # Mostly positive => 8/10
            score = 8
            details.append("Mostly positive/neutral headlines")

        return {"score": score, "details": "; ".join(details)}


    def analyze_risk_reward(self, financial_line_items: DataFrame):
        """
        Assesses risk via:
          - Debt-to-Equity
          - Price Volatility
        Aims for strong upside with contained downside.
        """
        if financial_line_items.empty:
            return {"score": 0, "details": "Insufficient data for risk-reward analysis"}

        details = []
        raw_score = 0  # We'll accumulate up to 6 raw points, then scale to 0-10

        #
        # 1. Debt-to-Equity
        #
        de_ratio = financial_line_items.debt_to_equity.values[0]
        dte_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'debt_to_equity')
        dte_threshold_low = dte_threshold - (dte_threshold*0.4)
        dte_threshold_mod = dte_threshold + (dte_threshold*0.4)
        dte_threshold_high = dte_threshold + (dte_threshold*2.0)
        if de_ratio < dte_threshold_low:
            raw_score += 3
            details.append(f"Low debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < dte_threshold_mod:
            raw_score += 2
            details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
        elif de_ratio < dte_threshold_high:
            raw_score += 1
            details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
        else:
            details.append(f"High debt-to-equity: {de_ratio:.2f}")

        #
        # 2. Price Volatility
        # Not using due to data constraints
        raw_score += 3

        # raw_score out of 6 => scale to 0–10
        final_score = min(10, (raw_score / 6) * 10)
        return {"score": final_score, "details": "; ".join(details)}


    def analyze_druckenmiller_valuation(self, financial_line_items: DataFrame):
        """
        Druckenmiller is willing to pay up for growth, but still checks:
          - P/E
          - P/FCF
          - EV/EBIT
          - EV/EBITDA
        Each can yield up to 2 points => max 8 raw points => scale to 0–10.
        """
        if financial_line_items.empty:
            return {"score": 0, "details": "Insufficient data to perform valuation"}

        details = []
        raw_score = 0

        # Gather needed data
        recent_net_income = financial_line_items.net_income.values[0]
        recent_fcf = financial_line_items.free_cash_flow.values[0]
        recent_ebit = financial_line_items.ebit.values[0]
        recent_ebitda = financial_line_items.ebitda.values[0]
        enterprise_value = financial_line_items.enterprise_value.values[0]
        market_cap = financial_line_items.market_cap.values[0]

        # 1) P/E
        pe_ratio_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'price_to_earning_ratio')
        pe_ratio_threshold_attractive = pe_ratio_threshold - (pe_ratio_threshold*0.4)
        if recent_net_income is not None and recent_net_income != 0:
            pe = market_cap / recent_net_income
            pe_points = 0
            if abs(pe) < pe_ratio_threshold_attractive:
                pe_points = 2
                details.append(f"Attractive P/E: {pe:.2f}")
            elif abs(pe) < pe_ratio_threshold:
                pe_points = 1
                details.append(f"Fair P/E: {pe:.2f}")
            else:
                details.append(f"High or Very high P/E: {pe:.2f}")
            raw_score += pe_points
        else:
            details.append("No positive net income for P/E calculation")

        # 2) P/FCF
        price_to_fcf_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'price_to_fcf')
        price_to_fcf_threshold_high = price_to_fcf_threshold + (price_to_fcf_threshold*0.66)
        if recent_fcf is not None and recent_fcf != 0:
            pfcf = market_cap / recent_fcf
            pfcf_points = 0
            if abs(pfcf) < price_to_fcf_threshold:
                pfcf_points = 2
                details.append(f"Attractive P/FCF: {pfcf:.2f}")
            elif abs(pfcf) < price_to_fcf_threshold_high:
                pfcf_points = 1
                details.append(f"Fair P/FCF: {pfcf:.2f}")
            else:
                details.append(f"High/Very high P/FCF: {pfcf:.2f}")
            raw_score += pfcf_points
        else:
            details.append("No positive free cash flow for P/FCF calculation")

        # 3) EV/EBIT
        evEBIT_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'ev_to_ebitda')
        evEBIT_threshold_attrative = evEBIT_threshold + (evEBIT_threshold*0.5)
        evEBIT_threshold_fair = evEBIT_threshold + (evEBIT_threshold*1.5)
        if enterprise_value > 0 and recent_ebit is not None and recent_ebit != 0:
            ev_ebit = enterprise_value / recent_ebit
            ev_ebit_points = 0
            if abs(ev_ebit) < evEBIT_threshold_attrative:
                ev_ebit_points = 2
                details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
            elif abs(ev_ebit) < evEBIT_threshold_fair:
                ev_ebit_points = 1
                details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
            else:
                details.append(f"High EV/EBIT: {ev_ebit:.2f}")
            raw_score += ev_ebit_points
        else:
            details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")

        # 4) EV/EBITDA
        evEBITDA_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'ev_to_ebitda')
        evEBITDA_threshold_fair = evEBITDA_threshold + (evEBITDA_threshold*0.8)
        if enterprise_value > 0 and recent_ebitda is not None and recent_ebitda != 0:
            ev_ebitda = enterprise_value / recent_ebitda
            ev_ebitda_points = 0
            if abs(ev_ebitda) < evEBITDA_threshold:
                ev_ebitda_points = 2
                details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
            elif abs(ev_ebitda) < evEBITDA_threshold_fair:
                ev_ebitda_points = 1
                details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
            else:
                details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
            raw_score += ev_ebitda_points
        else:
            details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")

        # We have up to 2 points for each of the 4 metrics => 8 raw points max
        # Scale raw_score to 0–10
        final_score = min(10, (raw_score / 8) * 10)

        return {"score": final_score, "details": "; ".join(details)}