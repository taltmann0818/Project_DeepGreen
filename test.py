import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

# --- Load JSON ---
data = {'UPST': {'BenGrahamAgent': {'name': 'Benjamin Graham',
   'signal': 'bearish',
   'score': 3,
   'max_score': 15,
   'earnings_analysis': {'score': 1,
    'details': 'EPS was negative in multiple periods.; EPS grew from latest to prior period.'},
   'strength_analysis': {'score': 2,
    'details': 'Current ratio = 1.49 (moderately strong).; Debt ratio = 0.67, somewhat high but could be acceptable.; No dividend data available to assess payout consistency.'},
   'valuation_analysis': {'score': 0,
    'details': 'Net Current Asset Value = 595,536,000.00; NCAV Per Share = 6.61; Price Per Share = 39.74; Unable to compute Graham Number (EPS or Book Value missing/<=0).'}},
  'BillAckmanAgent': {'name': 'Bill Ackman',
   'signal': 'neutral',
   'score': 7,
   'max_score': 20,
   'quality_analysis': {'score': 4,
    'details': 'Revenue grew by 27.0% over the prior period (strong growth).; Operating margin not consistently above 18.3%.; Majority of periods show positive free cash flow.; ROE of -1.1% is moderate.; Significant intangible assets may indicate brand value or proprietary tech.'},
   'balance_sheet_analysis': {'score': 1,
    'details': 'Debt-to-equity >= 1.58 in many periods (could be high leverage).; No dividend data found across periods.; Outstanding shares have decreased compared to previous period (possible buybacks).'},
   'activism_analysis': {'score': 2,
    'details': 'Revenue growth is healthy (~27.0%), but margins are low (avg -39.5%). Activism could unlock margin improvements.'},
   'valuation_analysis': {'score': 0,
    'details': 'Calculated intrinsic value: ~3,821,102,969.00; Market cap: ~3,581,438,735.90; Margin of safety: 6.69%',
    'intrinsic_value': 3821102968.995769,
    'margin_of_safety': 0.06691842322846348}},
  'CathieWoodAgent': {'name': 'Cathie Wood',
   'signal': 'neutral',
   'score': 7.916666666666667,
   'max_score': 15,
   'disruptive_analysis': {'score': 2.916666666666667,
    'details': 'Expanding gross margins: +15.6%; Positive operating leverage: Revenue growing faster than expenses; High R&D investment: 40.0% of revenue',
    'raw_score': 7,
    'max_score': 12},
   'innovation_analysis': {'score': 2.0,
    'details': 'Consistent positive FCF, good innovation funding capacity; Improving operating efficiency; Moderate investment in growth infrastructure; Strong focus on reinvestment over dividends',
    'raw_score': 6,
    'max_score': 15},
   'valuation_analysis': {'score': 3,
    'details': 'Calculated intrinsic value: ~8,258,087,323.10; Market cap: ~3,581,438,735.90; Margin of safety: 130.58%',
    'intrinsic_value': 8258087323.102577,
    'margin_of_safety': 1.305801643435723}},
  'CharlieMungerAgent': {'name': 'Charlie Munger',
   'signal': 'bearish',
   'score': 1.6444444444444444,
   'max_score': 10,
   'moat_analysis': {'score': 2.2222222222222223,
    'details': 'Poor ROIC: Never exceeds 26.8% threshold; Limited pricing power: Low or declining gross margins; High capital requirements: Avg capex 34.7% of revenue; Invests in R&D, building intellectual property; Significant goodwill/intangible assets, suggesting brand value or IP'},
   'management_analysis': {'score': 1.6666666666666667,
    'details': 'Poor cash conversion: FCF/NI ratio of only -8.67; Moderate debt level: D/E ratio of 2.04; Low cash reserves: Cash/Revenue ratio of 0.00; No insider trading data available; Concerning dilution: Share count increased significantly'},
   'predictability_analysis': {'score': 0.0,
    'details': 'Declining or highly unpredictable revenue: 6.0% avg growth; Insufficient operating income history; Unpredictable margins: -39.5% avg with high volatility (8.6%); Unpredictable cash generation: Positive FCF in only 3/4 periods'},
   'valuation_analysis': {'score': 3.0,
    'details': 'Expensive: Only 2.4% FCF yield; Expensive: 64.1% premium to reasonable value; Growing FCF trend adds to intrinsic value',
    'intrinsic_value_range': {'conservative': 858137500.0,
     'reasonable': 1287206250.0,
     'optimistic': 1716275000.0},
    'fcf_yield': 0.023960691869390694,
    'normalized_fcf': 85813750.0},
   'news_sentiment': 'No news data available'},
  'PeterLynchAgent': {'name': 'Peter Lynch',
   'signal': 'neutral',
   'score': 5.916666666666667,
   'max_score': 10.0,
   'growth_analysis': {'score': 10,
    'details': 'Strong revenue growth: 27.0%; Strong EPS growth: 88.7%'},
   'valuation_analysis': {'score': 4.0,
    'details': 'Estimated P/E: -529.96; Approx EPS growth rate: -88.7%'},
   'fundamentals_analysis': {'score': 3.333333333333333,
    'details': 'High debt-to-equity: 2.04; Low operating margin: -27.8%; Positive free cash flow: 225510000'},
   'sentiment_analysis': {'score': 5,
    'details': 'No news data; default to neutral sentiment'},
   'insider_activity': {'score': 5,
    'details': 'No insider trades data; defaulting to neutral'}},
  'PhilFisherAgent': {'name': 'Phil Fisher',
   'signal': 'bearish',
   'score': 3.6666666666666665,
   'max_score': 10,
   'growth_quality': {'score': 5.555555555555555,
    'details': 'Slight period-over-period revenue growth: 27.0%; Moderate multi-period EPS growth: 88.7%; R&D ratio 40.0% is very high (could be good if well-managed)'},
   'margins_stability': {'score': 0.0,
    'details': 'Operating margin may be negative or uncertain; Low gross margin: -27.8%; Operating margin volatility is high'},
   'management_efficiency': {'score': 0.0,
    'details': 'ROE is low for industry: -1.1%; High debt-to-equity: 2.04; Free cash flow is inconsistent or often negative'},
   'valuation_analysis': {'score': 10,
    'details': 'Reasonably attractive P/E: -529.96; Reasonable P/FCF: 15.88'},
   'insider_activity': {'score': 5,
    'details': 'No insider trades data; defaulting to neutral'},
   'sentiment_analysis': {'score': 5,
    'details': 'No news data; defaulting to neutral sentiment'}},
  'StanleyDruckenmillerAgent': {'name': 'Stanley Druckenmiller',
   'signal': 'neutral',
   'score': 6.194444444444445,
   'max_score': 10,
   'growth_momentum_analysis': {'score': 8.88888888888889,
    'details': 'Moderate revenue growth: 27.0%; Strong EPS growth: 88.7%; Not enough recent price data for momentum analysis.'},
   'sentiment_analysis': {'score': 5,
    'details': 'No news data; defaulting to neutral sentiment'},
   'insider_activity': {'score': 5,
    'details': 'No insider trades data; defaulting to neutral'},
   'risk_reward_analysis': {'score': 6.666666666666666,
    'details': 'Somewhat high debt-to-equity: 2.04'},
   'valuation_analysis': {'score': 2.5,
    'details': 'High or Very high P/E: -529.96; Attractive P/FCF: 15.88; High EV/EBIT: -714.27; High EV/EBITDA: -714.27'}},
  'WarrenBuffettAgent': {'name': 'Warren Buffett',
   'signal': 'bearish',
   'score': 2,
   'max_score': 15,
   'fundamental_analysis': {'score': 1,
    'details': 'Weak ROE of -1.1%; High debt to equity ratio of 2.0; Weak operating margin of -27.8%; Good liquidity position',
    'metrics': 'Operating Margin: -0.27847539163685703; Current Ratio: 1.4907816100289755; Debt to Equity: 2.0375661588887994; Return on Equity: -0.011347760672738507'},
   'consistency_analysis': {'score': 0,
    'details': 'Inconsistent earnings growth pattern; Total earnings growth of 87.6% over past 4 periods'},
   'moat_analysis': {'score': 0,
    'max_score': 3,
    'details': 'ROE not consistently above 27.449999999999996%; Operating margin not consistently above 18.3%'},
   'management_analysis': {'score': 1,
    'max_score': 2,
    'details': 'Company has been repurchasing shares (shareholder-friendly); No significant new stock issuance detected; No or minimal dividends paid'},
   'intrinsic_value_analysis': {'intrinsic_value': 458316898.82227015,
    'margin_of_safety': -0.8720299486828728,
    'owner_earnings': 27869500.0,
    'assumptions': {'growth_rate': 0.05,
     'discount_rate': 0.09,
     'terminal_multiple': 12,
     'projection_years': 10},
    'details': ['Intrinsic value calculated using DCF model with owner earnings']},
   'margin_of_safety': -0.8720299486828728},
  'ValuationAgent': {'name': 'Valuation',
   'signal': 'bullish',
   'confidence': 100,
   'reasoning': {'dcf_analysis': {'signal': 'bullish',
     'details': 'Value: $56,800,905,017.42, Market Cap: $3,581,438,735.90, Gap: 1486.0%, Weight: 35%'},
    'owner_earnings_analysis': {'signal': 'bullish',
     'details': 'Value: $181,272,503,854.68, Market Cap: $3,581,438,735.90, Gap: 4961.4%, Weight: 35%'}}},
  'FundamentalsAgent': {'name': 'Fundamentals',
   'signal': 'neutral',
   'confidence': 50.0,
   'reasoning': {'profitability_signal': {'signal': 'bearish',
     'details': 'ROE: -1.13%, Net Margin: -1.13%, Op Margin: -1.13%'},
    'growth_signal': {'signal': 'bullish',
     'details': 'Revenue Growth: 27.04%, Earnings Growth: 88.71%'},
    'financial_health_signal': {'signal': 'bullish',
     'details': 'Current Ratio: 1.49, D/E: 2.04'},
    'price_ratios_signal': {'signal': 'bearish',
     'details': 'P/E: -529.96, P/B: 6.01, P/S: 22.09'}}}}}

st.sidebar.title("Settings")
ticker = st.sidebar.selectbox("Ticker", list(data.keys()))
signal_filter = st.sidebar.radio("Show signals", ["All", "Bullish", "Neutral", "Bearish"])

agents = data[ticker]

# --- Agent Cards Grid ---
cols = st.columns(3)
for (name, ag), col in zip(agents.items(), cols * ((len(agents)//3)+1)):
    sig = ag.get("signal", "n/a").capitalize()
    if signal_filter != "All" and sig.lower() != signal_filter.lower():
        continue

    with col:
        st.markdown(f"### {ag['name']}")
        color = {"Bullish":"green","Neutral":"gray","Bearish":"red"}.get(sig, "black")
        st.markdown(f"<span style='color:{color};font-weight:bold'>{sig}</span>", unsafe_allow_html=True)

        # Only show a progress bar if there's a score/max_score
        if "score" in ag and ag.get("max_score"):
            pct = ag["score"] / ag["max_score"]
            st.progress(pct)

        # Fallback: if there's a confidence field instead
        elif "confidence" in ag:
            st.metric("Confidence", f"{ag['confidence']}%")

        with st.expander("Key highlights"):
            for section in [k for k in ag if k.endswith("_analysis")][:2]:
                details = ag[section].get("details", "")
                snippet = details.split(';')[0] + ("…" if ";" in details else "")
                st.write(f"**{section.replace('_',' ').title()}:** {snippet}")

# --- Deep Dive Tabs ---
tabs = st.tabs(list(ag['name'] for ag in agents.values()))
for tab, (name, ag) in zip(tabs, agents.items()):
    with tab:
        left, right = st.columns((1,2))

        with left:
            st.metric("Signal", ag.get("signal", "N/A").capitalize())
            if "score" in ag and ag.get("max_score"):
                st.metric("Score", f"{np.round(ag['score'],2)} / {ag['max_score']}")
            elif "confidence" in ag:
                st.metric("Confidence", f"{ag['confidence']}%")

            val = ag.get("valuation_analysis", {})
            if "margin_of_safety" in val:
                mos = val["margin_of_safety"] * 100
                st.metric("Margin of Safety", f"{mos:.1f}%")

        with right:
            for key, block in ag.items():
                if isinstance(block, dict) and key.endswith("_analysis"):
                    with st.expander(key.replace("_"," ").title()):
                        for k, v in block.items():
                            st.write(f"**{k.replace('_',' ').title()}:** {v}")