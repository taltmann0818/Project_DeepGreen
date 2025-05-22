from statistics import median
from pandas import DataFrame
from Components.Fundamentals import search_line_items, get_metric_value
import pandas as pd
import numpy as np
    
class ValuationAgent():
    """Valuation Agent
    Implements four complementary valuation methodologies and aggregates them with
    configurable weights.
    """

    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = 'Valuation'
        self.metrics = metrics
        self.ticker = ticker
        self.period = kwargs.get('analysis_period')
        self.limit = kwargs.get('analysis_limit')
        self.analysis_data = {}  # Storing returned results in dict
        self.threshold_matrix_path = kwargs.get('threshold_matrix_path',None)

    def analyze(self):
        # --- Fine‑grained line‑items (need two periods to calc WC change) ---
        line_items, self.SIC_code = search_line_items(
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
            ],
            period=self.period,
            limit=2,  # Override kwargs to ensure this is always period-over-period
            df=self.metrics
        )
        
        self.threshold_matrix = pd.read_csv(self.threshold_matrix_path.get('business_services_sic')) if len(self.SIC_code) > 2 else pd.read_csv(self.threshold_matrix_path.get('two_digit_sic'))

        # ------------------------------------------------------------------
        # Valuation models
        # ------------------------------------------------------------------
        # --- Working capital change ---
        wc = line_items.working_capital.values
        wc_change = (wc[0] - wc[1]) if len(wc) >= 2 else 0

        # --- EPS growth ---
        eps = line_items.earnings_per_share.values
        if len(eps) >= 2 and abs(eps[1]) > 0:
            earnings_growth = (eps[0] - eps[1]) / abs(eps[1])
        else:
            earnings_growth = 0

        # --- Book value growth ---
        bv = line_items.book_value.values
        if len(bv) >= 2 and abs(bv[1]) > 0:
            book_value_growth = (bv[0] - bv[1]) / abs(bv[1])
        else:
            book_value_growth = 0

        # Owner Earnings
        owner_val = calculate_owner_earnings_value(
            net_income=line_items.net_income.values[0] or 0,
            depreciation=line_items.depreciation_and_amortization.values[0] or 0,
            capex=line_items.capital_expenditure.values[0] or 0,
            working_capital_change=wc_change,
            growth_rate=earnings_growth
        )

        # Discounted Cash Flow
        dcf_val = calculate_intrinsic_value(
            free_cash_flow=line_items.free_cash_flow.values[0] or 0,
            growth_rate=earnings_growth,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )

        # Implied Equity Value
        ev_ebitda_val = calculate_ev_ebitda_value(line_items)

        # Residual Income Model
        market_cap = line_items.market_cap.values[0] or 0
        rim_val = calculate_residual_income_value(
            market_cap=market_cap,
            net_income=line_items.net_income.values[0] or 0,
            book_val=line_items.book_value.values[0] or 0,
            book_value_growth=book_value_growth,
        )

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        method_values = {
            "dcf":             {"value": dcf_val,        "weight": 0.35},
            "owner_earnings":  {"value": owner_val,      "weight": 0.35},
            "ev_ebitda":       {"value": ev_ebitda_val,  "weight": 0.20},
            "residual_income": {"value": rim_val,        "weight": 0.10},
        }

        total_weight = sum(v["weight"] for v in method_values.values() if v["value"] > 0)

        # compute a “gap” only if market_cap > 0
        for m in method_values.values():
            val = m["value"]
            if val > 0 and market_cap > 0:
                m["gap"] = (val - market_cap) / market_cap
            else:
                m["gap"] = None

        # if all weights were filtered out, set weighted_gap = 0
        if total_weight > 0:
            weighted_gap = sum(m["weight"] * m["gap"] for m in method_values.values() if m["gap"] is not None) / total_weight
        else:
            weighted_gap = 0

        signal = (
            "bullish" if weighted_gap > 0.15 else
            "bearish" if weighted_gap < -0.15 else
            "neutral"
        )
        confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100))

        # build reasoning, only for methods that have a gap
        reasoning = {}
        for name, m in method_values.items():
            gap = m["gap"]
            if gap is not None:
                reasoning[f"{name}_analysis"] = {
                    "signal": (
                        "bullish" if gap > 0.15 else
                        "bearish" if gap < -0.15 else
                        "neutral"
                    ),
                    "details": (
                        f"Value: ${m['value']:,.2f}, "
                        f"Market Cap: ${market_cap:,.2f}, "
                        f"Gap: {gap:.1%}, "
                        f"Weight: {m['weight'] * 100:.0f}%"
                    ),
                }

        self.analysis_data = {
            "name":       self.agent_name,
            "signal":     signal,
            "confidence": confidence,
            "reasoning":  reasoning,
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

    # check denominator in terminal value
    terminal_growth = min(growth_rate, 0.03)
    denom = required_return - terminal_growth
    if denom <= 0:
        return 0
    term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / denom
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
    if free_cash_flow <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        fcft = free_cash_flow * (1 + growth_rate) ** yr
        pv += fcft / (1 + discount_rate) ** yr

    term_val = (free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    pv_term = term_val / (1 + discount_rate) ** num_years

    return pv + pv_term


def calculate_ev_ebitda_value(financial_metrics: DataFrame) -> float:
    """Implied equity value via median EV/EBITDA multiple."""
    if financial_metrics.empty:
        return 0

    evs     = financial_metrics.enterprise_value.values.astype(float)
    ebitdas = financial_metrics.ebitda.values.astype(float)

    # elementwise safe division
    ev_to_ebt = [
        (ev / e) if e != 0 else 0
        for ev, e in zip(evs, ebitdas)
    ]

    # nothing to do if the latest period has no ebitda
    if ebitdas[0] == 0:
        return 0

    # median multiple
    try:
        med = median(ev_to_ebt)
    except ValueError:
        return 0

    ev_implied = med * ebitdas[0]
    net_debt   = evs[0] - (financial_metrics.market_cap.values[0] or 0)
    return max(ev_implied - net_debt, 0)


def calculate_residual_income_value(
        market_cap: float,
        net_income: float,
        book_val: float,
        book_value_growth: float = 0.03,
        cost_of_equity: float = 0.10,
        terminal_growth_rate: float = 0.03,
        num_years: int = 5,
):
    """Residual Income Model (Edwards‑Bell‑Ohlson)."""
    if book_val <= 0 or net_income <= 0 or market_cap <= 0:
        return 0

    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    # guard terminal denom
    denom = cost_of_equity - terminal_growth_rate
    if denom <= 0:
        return 0

    term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / denom
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    intrinsic = book_val + pv_ri + pv_term
    return intrinsic * 0.8  # 20% margin of safety