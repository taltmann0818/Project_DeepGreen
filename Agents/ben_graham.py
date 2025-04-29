import math
from pandas import DataFrame
from Components.Fundamentals import search_line_items

class BenGrahamAgent:
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    def __init__(self, ticker, metrics):
        self.agent_name = "Benjamin Graham"
        self.analysis_data = {}
        self.metrics = metrics
        self.ticker = ticker

    def analyze(self):
        financial_line_items = search_line_items(
            self.ticker, 
            ["earnings_per_share", 
             "revenue", 
             "net_income", 
             "book_value_per_share", 
             "total_assets", 
             "total_liabilities", 
             "current_assets", 
             "current_liabilities", 
             "dividends_and_other_cash_distributions", 
             "outstanding_shares",
             "market_cap"
            ], 
            period="FY", 
            limit=10,
            df=self.metrics
        )

        # Perform sub-analyses
        earnings_analysis = analyze_earnings_stability(financial_line_items)
        strength_analysis = analyze_financial_strength(financial_line_items)
        valuation_analysis = analyze_valuation_graham(financial_line_items)

        # Aggregate scoring
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # total possible from the three analysis functions

        # Map total_score to signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        self.analysis_data = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "earnings_analysis": earnings_analysis,
            "strength_analysis": strength_analysis,
            "valuation_analysis": valuation_analysis
        }

        return self.analysis_data

def analyze_earnings_stability(financial_line_items: DataFrame):
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    if financial_line_items.empty:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    score = 0
    details = []

    eps_vals = financial_line_items.earnings_per_share.values
    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    # 2. EPS growth from earliest to latest
    if eps_vals[-1] > eps_vals[0]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    return {"score": score, "details": "; ".join(details)}

def analyze_financial_strength(financial_line_items: DataFrame):
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    if financial_line_items.empty:
        return {"score": score, "details": "No data for financial strength analysis"}

    score = 0
    details = []

    total_assets = financial_line_items.total_assets.values[0] if financial_line_items.total_assets.values.any() else 0
    total_liabilities = financial_line_items.total_liabilities.values[0] if financial_line_items.total_liabilities.values.any() else 0
    current_assets = financial_line_items.current_assets.values[0] if financial_line_items.current_assets.values.any() else 0
    current_liabilities = financial_line_items.current_liabilities.values[0] if financial_line_items.current_liabilities.values.any() else 0

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    div_periods = financial_line_items.dividends_and_other_cash_distributions.values[0] if financial_line_items.dividends_and_other_cash_distributions.values.any() else None
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("Company paid dividends in the majority of the reported years.")
            else:
                details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "details": "; ".join(details)}

def analyze_valuation_graham(financial_line_items: DataFrame):
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if financial_line_items.empty or not financial_line_items.market_cap.values.any() or financial_line_items.market_cap.values[0] <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    total_liabilities = financial_line_items.total_liabilities.values[0] if financial_line_items.total_liabilities.values.any() else 0
    current_assets = financial_line_items.current_assets.values[0] if financial_line_items.current_assets.values.any() else 0
    book_value_ps = financial_line_items.book_value_per_share.values[0] if financial_line_items.book_value_per_share.values.any() else 0
    eps = financial_line_items.earnings_per_share.values[0] if financial_line_items.earnings_per_share.values.any() else 0
    shares_outstanding = financial_line_items.outstanding_shares.values[0] if financial_line_items.outstanding_shares.values.any() else 0
    market_cap = financial_line_items.market_cap.values[0] if financial_line_items.market_cap.values.any() else 0

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        else:
            # For partial net-net discount
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
    else:
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}