import math
from pandas import DataFrame
from Components.Fundamentals import search_line_items, get_metric_value
import pandas as pd
import numpy as np


class AswathDamodaranAgent():
    """
    Analyze US equities through Aswath Damodaran’s intrinsic‑value lens:
      • Cost of Equity via CAPM (risk‑free + β·ERP)
      • 5‑yr revenue / FCFF growth trends & reinvestment efficiency
      • FCFF‑to‑Firm DCF → equity value → per‑share intrinsic value
      • Cross‑check with relative valuation (PE vs. Fwd PE sector median proxy)
    Produces a trading signal and explanation in Damodaran’s analytical voice.
    """
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = "Aswath Damodaran"
        self.metrics = metrics
        self.ticker = ticker
        self.period = kwargs.get('analysis_period')
        self.limit = kwargs.get('analysis_limit')
        self.threshold_matrix_path = kwargs.get('threshold_matrix_path',None)
        self.analysis_data = {} # Storing returned results in dict

    def analyze(self):
        financial_line_items, self.SIC_code = search_line_items(
            self.ticker, 
            [
                "free_cash_flow",
                "ebit",
                "interest_expense",
                "capital_expenditure",
                "depreciation_and_amortization",
                "outstanding_shares",
                "net_income",
                "revenue",
                "total_debt",
                "market_cap",
                "return_on_invested_capital",
                "debt_to_equity",
                "share_price",
                "earnings_per_share",
            ], 
            period=self.period, 
            limit=self.limit,
            df=self.metrics
        )

        self.threshold_matrix = pd.read_csv(self.threshold_matrix_path.get('business_services_sic')) if len(self.SIC_code) > 2 else pd.read_csv(self.threshold_matrix_path.get('two_digit_sic'))
    
        # ─── Analyses ───────────────────────────────────────────────────────────
        growth_analysis = self.analyze_growth_and_reinvestment(financial_line_items)
        risk_analysis = self.analyze_risk_profile(financial_line_items)
        intrinsic_val_analysis = self.calculate_intrinsic_value_dcf(financial_line_items)
        relative_val_analysis = self.analyze_relative_valuation(financial_line_items)

        # ─── Score & margin of safety ──────────────────────────────────────────
        total_score = (
            growth_analysis["score"]
            + risk_analysis["score"]
            + relative_val_analysis["score"]
        )
        max_score = growth_analysis["max_score"] + risk_analysis["max_score"] + relative_val_analysis["max_score"]

        intrinsic_value = intrinsic_val_analysis["intrinsic_value"]
        market_cap = financial_line_items.market_cap.values[0]
        margin_of_safety = (
            (intrinsic_value - market_cap) / market_cap if intrinsic_value and market_cap else None
        )

        # Decision rules (Damodaran tends to act with ~20‑25 % MOS)
        max_possible_score = 8
        if margin_of_safety is not None and margin_of_safety >= 0.25 and (total_score >= 0.5 * max_possible_score):
            signal = "bullish"
        elif margin_of_safety is not None and margin_of_safety <= -0.25:
            signal = "bearish"
        else:
            signal = "neutral"
            
        # ─── Push data back to manager ──────────────────────────────────────
        self.analysis_data = {
            "name": self.agent_name,
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "margin_of_safety": margin_of_safety,
            "growth_analysis": growth_analysis,
            "risk_analysis": risk_analysis,
            "relative_val_analysis": relative_val_analysis,
            "intrinsic_val_analysis": intrinsic_val_analysis,
        }

        return self.analysis_data

    # ────────────────────────────────────────────────────────────────────────────────
    # Helper analyses
    # ────────────────────────────────────────────────────────────────────────────────
    def analyze_growth_and_reinvestment(self, financial_line_items: DataFrame):
        """
        Growth score (0‑4):
          +2  5‑yr CAGR of revenue > 8 %
          +1  5‑yr CAGR of revenue > 3 %
          +1  Positive FCFF growth over 5 yr
        Reinvestment efficiency (ROIC > WACC) adds +1
        """
        max_score = 4
        if len(financial_line_items) < 2:
            return {"score": 0, "max_score": max_score, "details": "Insufficient history"}
    
        # Revenue CAGR (oldest to latest)
        revs = financial_line_items.revenue.values
        if len(revs) >= 2 and revs[-1] != 0:
            cagr = (revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1
        else:
            cagr = None
    
        score, details = 0, []
    
        if cagr is not None:
            if cagr > 0.08:
                score += 2
                details.append(f"Revenue CAGR {cagr:.1%} (>0.08%)")
            elif cagr > 0.03:
                score += 1
                details.append(f"Revenue CAGR {cagr:.1%} (>0.03%)")
            else:
                details.append(f"Sluggish revenue CAGR {cagr:.1%}")
        else:
            details.append("Revenue data incomplete")
    
        # FCFF growth (proxy: free_cash_flow trend)
        fcfs = financial_line_items.free_cash_flow.values
        if len(fcfs) >= 2 and fcfs[0] > fcfs[-1]:
            score += 1
            details.append("Positive FCFF growth")
        else:
            details.append("Flat or declining FCFF")
    
        # Reinvestment efficiency (ROIC vs. 10 % hurdle)
        latest_roic = financial_line_items.return_on_invested_capital.values[0]
        roic_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'return_on_invested_capital')
        if latest_roic and latest_roic > roic_threshold:
            score += 1
            details.append(f"ROIC {latest_roic:.1%} ( >{roic_threshold}% )")
    
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}
    
    def analyze_risk_profile(self, financial_line_items: DataFrame):
        """
        Risk score (0‑3):
          +1  Beta < 1.3
          +1  Debt/Equity < 1
          +1  Interest Coverage > 3×
        """
        max_score = 3
        if financial_line_items.empty:
            return {"score": 0, "max_score": max_score, "details": "No metrics"}
    
        score, details = 0, []
    
        # Beta
        beta = 1 # This is static for right now due to data constraints
        if beta is not None:
            if beta < 1.3:
                score += 1
                details.append(f"Beta {beta:.2f}")
            else:
                details.append(f"High beta {beta:.2f}")
        else:
            details.append("Beta NA")
    
        # Debt / Equity
        dte = financial_line_items.debt_to_equity.values[0]
        if dte is not None:
            if dte < 1:
                score += 1
                details.append(f"D/E {dte:.1f}")
            else:
                details.append(f"High D/E {dte:.1f}")
        else:
            details.append("D/E NA")
    
        # Interest coverage
        ebit = financial_line_items.ebit.values[0]
        interest = financial_line_items.interest_expense.values[0]
        if ebit and interest and interest != 0:
            coverage = ebit / abs(interest)
            if coverage > 3:
                score += 1
                details.append(f"Interest coverage {coverage:.1f}")
            else:
                details.append(f"Weak coverage {coverage:.1f}")
        else:
            details.append("Interest coverage NA")
    
        return {
            "score": score,
            "max_score": max_score,
            "details": "; ".join(details),
            "beta": beta,
        }
    
    def analyze_relative_valuation(self, financial_line_items: DataFrame):
        """
        Simple PE check vs. historical median (proxy since sector comps unavailable):
          +1 if TTM P/E < 70 % of 5‑yr median
          +0 if between 70 %‑130 %
          ‑1 if >130 %
        """
        max_score = 1
        if financial_line_items.empty or len(financial_line_items) < 3:
            return {"score": 0, "max_score": max_score, "details": "Insufficient P/E history"}
    
        pes = financial_line_items.share_price.values / financial_line_items.earnings_per_share.values
        if len(pes) < 3:
            return {"score": 0, "max_score": max_score, "details": "P/E data sparse"}
    
        ttm_pe = pes[0]
        median_pe = sorted(pes)[len(pes) // 2]
    
        if ttm_pe < 0.7 * median_pe:
            score, desc = 1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (cheap)"
        elif ttm_pe > 1.3 * median_pe and ttm_pe >= 30:
            score, desc = -1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (expensive)"
        else:
            score, desc = 0, f"P/E is neither cheap nor too expensive"
    
        return {"score": score, "max_score": max_score, "details": desc}
    
    # ────────────────────────────────────────────────────────────────────────────────
    # Intrinsic value via FCFF DCF (Damodaran style)
    # ────────────────────────────────────────────────────────────────────────────────
    def calculate_intrinsic_value_dcf(self, financial_line_items: DataFrame):
        """
        FCFF DCF with:
          • Base FCFF = latest free cash flow
          • Growth = 5‑yr revenue CAGR (capped 12 %)
          • Fade linearly to terminal growth 2.5 % by year 10
          • Discount @ cost of equity (no debt split given data limitations)
        """
        if financial_line_items.empty or len(financial_line_items) < 2:
            return {"intrinsic_value": None, "details": ["Insufficient data"]}

        fcff0 = financial_line_items.free_cash_flow.values[0]
        shares = financial_line_items.outstanding_shares.values[0]
        if not fcff0 or not shares:
            return {"intrinsic_value": None, "details": ["Missing FCFF or share count"]}
    
        # Growth assumptions
        revs = financial_line_items.revenue.values
        if len(revs) >= 2 and revs[-1] != 0:
            base_growth = min((revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1, 0.12)
        else:
            base_growth = 0.04  # fallback
    
        terminal_growth = 0.025
        years = 10
    
        # Discount rate
        discount = self.estimate_cost_of_equity(1)
    
        # Project FCFF and discount
        pv_sum = 0.0
        g = base_growth
        g_step = (terminal_growth - base_growth) / (years - 1)
        for yr in range(1, years + 1):
            fcff_t = fcff0 * (1 + g)
            pv = fcff_t / (1 + discount) ** yr
            pv_sum += pv
            g += g_step
    
        # Terminal value (perpetuity with terminal growth)
        tv = (
            fcff0
            * (1 + terminal_growth)
            / (discount - terminal_growth)
            / (1 + discount) ** years
        )
    
        equity_value = pv_sum + tv
        intrinsic_per_share = equity_value / shares
    
        return {
            "intrinsic_value": equity_value,
            "intrinsic_per_share": intrinsic_per_share,
            "assumptions": {
                "base_fcff": fcff0,
                "base_growth": base_growth,
                "terminal_growth": terminal_growth,
                "discount_rate": discount,
                "projection_years": years,
            },
            "details": ["FCFF DCF completed"],
        }

    def estimate_cost_of_equity(self, beta: float) -> float:
        """CAPM: r_e = r_f + β × ERP (use Damodaran’s long‑term averages)."""
        risk_free = 0.04          # 10‑yr US Treasury proxy
        erp = 0.05                # long‑run US equity risk premium
        beta = beta if beta is not None else 1.0
        return risk_free + beta * erp