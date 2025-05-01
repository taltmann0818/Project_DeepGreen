from statistics import median
from Components.Fundamentals import search_line_items

class ValuationAgent():
    """Valuation Agent
    Implements four complementary valuation methodologies and aggregates them with
    configurable weights.
    """
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = 'Valuation'
        self.analysis_data = {}
        self.metrics = metrics
        self.ticker = ticker

        self.period = kwargs.get('analysis_period','FY')
        self.limit = kwargs.get('analysis_period', 2)

    def analyze(self):
        # --- Fine‑grained line‑items (need two periods to calc WC change) ---
        line_items = search_line_items(
            ticker=self.ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "working_capital",
                "earnings_per_share",
                "enterprise_value",
                "ebitda",
                "market_cap",
                "book_value",
                "price_to_book_ratio"
            ],
            period=self.period,
            limit=2,
            df=self.metrics
        )

        # ------------------------------------------------------------------
        # Valuation models
        # ------------------------------------------------------------------
        wc_change = line_items.working_capital.values[0] - line_items.working_capital.values[1]
        earnings = line_items.earnings_per_share.values
        earnings_growth = (earnings[0] - earnings[-1]) / abs(earnings[-1]) if not earnings[-1] > 0 else 0.05
        book_values = line_items.book_value.values
        book_value_growth = (book_values[0] - book_values[-1]) / abs(book_values[-1]) if not book_values[-1] > 0 else 0.03

        # Owner Earnings
        owner_val = calculate_owner_earnings_value(
            net_income=line_items.net_income.values[0],
            depreciation=line_items.depreciation_and_amortization.values[0],
            capex=line_items.capital_expenditure.values[0],
            working_capital_change=wc_change,
            growth_rate=earnings_growth
        )

        # Discounted Cash Flow
        dcf_val = calculate_intrinsic_value(
            free_cash_flow=line_items.free_cash_flow.values[0],
            growth_rate=earnings_growth,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )

        # Implied Equity Value
        ev_ebitda_val = calculate_ev_ebitda_value(line_items)

        # Residual Income Model
        rim_val = calculate_residual_income_value(
            market_cap=line_items.market_cap.values[0],
            net_income=line_items.net_income.values[0],
            price_to_book_ratio=line_items.price_to_book_ratio.values[0],
            book_value_growth=book_value_growth,
        )

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        market_cap = get_market_cap(self.ticker)
        method_values = {
            "dcf": {"value": dcf_val, "weight": 0.35},
            "owner_earnings": {"value": owner_val, "weight": 0.35},
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
            "residual_income": {"value": rim_val, "weight": 0.10},
        }

        total_weight = sum(v["weight"] for v in method_values.values() if v["value"] > 0)

        for v in method_values.values():
            v["gap"] = (v["value"] - market_cap) / market_cap if v["value"] > 0 else None

        weighted_gap = sum(
            v["weight"] * v["gap"] for v in method_values.values() if v["gap"] is not None
        ) / total_weight

        signal = "bullish" if weighted_gap > 0.15 else "bearish" if weighted_gap < -0.15 else "neutral"
        confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100))

        reasoning = {
            f"{m}_analysis": {
                "signal": (
                    "bullish" if vals["gap"] and vals["gap"] > 0.15 else
                    "bearish" if vals["gap"] and vals["gap"] < -0.15 else "neutral"
                ),
                "details": (
                    f"Value: ${vals['value']:,.2f}, Market Cap: ${market_cap:,.2f}, "
                    f"Gap: {vals['gap']:.1%}, Weight: {vals['weight']*100:.0f}%"
                ),
            }
            for m, vals in method_values.items() if vals["value"] > 0
        }

        self.analysis_data = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        return self.analysis_data

#############################
# Helper Valuation Functions
#############################

def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """Buffett owner‑earnings valuation with margin‑of‑safety."""
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
        return 0

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        future = owner_earnings * (1 + growth_rate) ** yr
        pv += future / (1 + required_return) ** yr

    terminal_growth = min(growth_rate, 0.03)
    term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / (
        required_return - terminal_growth
    )
    pv_term = term_val / (1 + required_return) ** num_years

    intrinsic = pv + pv_term
    return intrinsic * (1 - margin_of_safety)


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """Classic DCF on FCF with constant growth and terminal value."""
    if free_cash_flow is None or free_cash_flow <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        fcft = free_cash_flow * (1 + growth_rate) ** yr
        pv += fcft / (1 + discount_rate) ** yr

    term_val = (
        free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)
    pv_term = term_val / (1 + discount_rate) ** num_years

    return pv + pv_term


def calculate_ev_ebitda_value(financial_metrics: Dataframe):
    """Implied equity value via median EV/EBITDA multiple."""
    if financial_metrics.empty:
        return 0

    enterprise_values = financial_metrics.enterprise_value.values
    ebitdas = financial_metrics.ebitda.values
    enterprise_value_to_ebitda_ratio = enterprise_values / ebitdas

    if not (enterprise_values[0] and enterprise_value_to_ebitda_ratio[0]) or enterprise_value_to_ebitda_ratio[0] == 0:
        return 0

    ebitda_now = enterprise_value[0] / enterprise_value_to_ebitda_ratio[0]
    med_mult = median(enterprise_value_to_ebitda_ratio)
    ev_implied = med_mult * ebitdas[0]
    net_debt = (enterprise_values[0] or 0) - (financial_metrics.market_cap.values[0] or 0)
    return max(ev_implied - net_debt, 0)


def calculate_residual_income_value(
    market_cap: float,
    net_income: float,
    price_to_book_ratio: float,
    book_value_growth: float = 0.03,
    cost_of_equity: float = 0.10,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
):
    """Residual Income Model (Edwards‑Bell‑Ohlson)."""
    if not (market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0):
        return 0

    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / (
        cost_of_equity - terminal_growth_rate
    )
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    intrinsic = book_val + pv_ri + pv_term
    return intrinsic * 0.8  # 20% margin of safety
