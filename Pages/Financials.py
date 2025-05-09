import streamlit as st
import numpy as np
import pandas as pd
import random

# Custom libraries
from Components.Fundamentals import FundementalData
from Components.AgentManager import AgentManager

### Page parameters --------------------------------------------------------------------------------

# Retrieve authenticator class with YAML credentials from SL session_state for logout widget on this page
if not st.experimental_user.is_logged_in:
    st.warning("Go to the Dashboard Page to Get Started")

### Backend functions ------------------------------------------------------------------------------  

_INDEX_CONFIG = {
    'NASDAQ':        ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 1),
    'S&P500':        ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 0),
    'RUSSELL1000':   ('https://en.wikipedia.org/wiki/Russell_1000_Index', 3, 1),
    'DOWJONES':      ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 2, 2),
}
AGENT_LIST = ['BenGrahamAgent','BillAckmanAgent','CathieWoodAgent','CharlieMungerAgent','PeterLynchAgent',
               'PhilFisherAgent','StanleyDruckenmillerAgent','WarrenBuffettAgent','ValuationAgent','FundamentalsAgent']

SPINNER_STRINGS = ["Running the Bulls...","Poking the Bear...","Buying the Dip...","Chasing the Rally...","Playing the Spread...","Fighting the Fed..."]

def get_index_tickers(indices, sample_size=10):
    all_tickers = []
    for idx in indices:
        cfg = _INDEX_CONFIG.get(idx)
        if not cfg:
            continue
        url, table_i, col_i = cfg
        try:
            df = pd.read_html(url)[table_i]
            tickers = df.iloc[:, col_i].dropna().astype(str).tolist()
        except Exception as e:
            print(f"Warning: could not fetch {idx} → {e}")
            continue
        all_tickers.extend(tickers)
    all_tickers = list(dict.fromkeys(all_tickers))
    if sample_size >= len(all_tickers):
        raise ValueError(f"Sample size ({sample_size}), cannot be greater than length of list ({len(all_tickers)}) !")
    else:
        sampled_tickers = random.sample(list(all_tickers), sample_size)

    return sampled_tickers

def run_analysis():
    # ——— compute block ———
    missing_fields = []
    if mode_selection is None:
        missing_fields.append("Mode")
    if ticker_select is None:
        missing_fields.append("Equity/Index Name")
    if agents_select is None:
        missing_fields.append("Agents")
    if period_select is None:
        missing_fields.append("Period")
    if num_workers_select is None:
        missing_fields.append("Worker Amt.")
    # Check if any fields are missing
    if missing_fields:
        st.error(f"Please fill out the following fields: {', '.join(missing_fields)}")
        st.session_state.analysis_done = False
        
        return 

    with st.spinner(f"{np.random.choice(SPINNER_STRINGS)} Please wait."):
        if mode_selection == "Multi":
            tickers = get_index_tickers(ticker_select, sample_size=sample_size_select)
        else:
            tickers = [ticker_select]
    
        data_fetcher = FundementalData(
            tickers,
            workers=num_workers_select,
            fetch_stock_price=False,
            fetch_market_cap=True
        )
        financials = data_fetcher.fetch()
        manager = AgentManager(financials, period=period_select,agents=agents_select,streamlit_progress=True)
        raw_data, summary = manager.agent_analysis()
    
        # **store** in session_state
        st.session_state.raw_data   = raw_data
        st.session_state.summary    = summary
        st.session_state.analysis_done = True

def reset_analysis():
    for k in ("analysis_done","raw_data","summary"):
        st.session_state.pop(k, None)

# ---------------
# Streamlit Layout
# ---------------

## Frontend ui -----------------------------------------------------------------------------------------------------------------------------
if st.experimental_user.is_logged_in:

    col1, col2 = st.columns([3, 1])

    with col2:
        with st.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            submit = st.button("Analyze",icon=":material/query_stats:",on_click=run_analysis)

            # Segmented control to toggle showing the ticker input
            mode_selection = st.segmented_control("Mode", ["Single", "Multi"], selection_mode="single", help="Whether to analyze a single or multiple stocks.",on_change=reset_analysis)
            if mode_selection == "Single":
                ticker_select = st.text_input("U.S. Equity",on_change=reset_analysis)
            elif mode_selection == "Multi":
                ticker_select = st.multiselect("Index",list(_INDEX_CONFIG.keys()),on_change=reset_analysis)
                sample_size_select = st.slider('Index Sample Size',1, 500)
            else:
                ticker_select = None

            st.subheader("Settings")
            period_select = st.pills('Period',options=['Annual','Quarterly'],default=['Annual'],on_change=reset_analysis)
            num_workers_select = st.slider('Worker Amt.', 10, 50, 10,on_change=reset_analysis)
            agents_select = st.multiselect("Agents",options=AGENT_LIST,default=AGENT_LIST,on_change=reset_analysis)

    with col1:
        st.subheader("Financials")
        if st.session_state.get("analysis_done", False):
            summary  = st.session_state.summary
            raw_data = st.session_state.raw_data

            # --- Summmary Table ---
            with st.container(border=True):
                st.subheader("Analysis Summary")
                st.dataframe(summary)

            # --- Agent Cards Grid ---
            with st.container(border=True):
                st.subheader("Agent Results Grid")

                results_ticker_select = st.selectbox("Ticker", list(raw_data.keys()))
                agents = raw_data[results_ticker_select]

                cols = st.columns(3)
                for (name, ag), col in zip(agents.items(), cols * ((len(agents)//3)+1)):
                    try:
                        sig = ag.get("signal", "n/a").capitalize()

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
                    except:
                        st.warning(f"Oops! Couldn't get signals for {name}. This is likely an error.")

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
                                if isinstance(block, dict):
                                    with st.expander(key.replace("_"," ").title()):
                                        for k, v in block.items():
                                            st.write(f"**{k.replace('_',' ').title()}:** {v}")


        else:
            with st.container(border=True):
                st.write("Enter the params and click the button to get results!")

# ---------------
# End of the App
# ---------------
st.write("No analysis provided on this page or site should constitute any professional investment advice or recommendations to buy, sell, or hold any investments or investment products of any kind, and should be treated as information for educational purposes to learn more about the stock market.")