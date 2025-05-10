import streamlit as st
import datetime

st.set_page_config(page_title="DeepGreen",
                   layout="wide",
                   page_icon=":material/finance_mode:"
)


def login():
    # 1) Inject CSS up front
    st.markdown(
        """
        <style>
        /* CARD STYLING */
        #login-card {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 2rem;
            max-width: 400px;
            margin: 2rem auto;
            text-align: center;
        }

        /* BUTTONS: every stButton that follows #login-card */
        #login-card ~ .stButton > button {
            margin-top: 1rem;
            font-size: 1rem;
            border-radius: 6px;
            width: 100%;
            background-color: #0078d4;
            color: #ffffff;
        }
        #login-card ~ .stButton > button:hover {
            background-color: #005a9e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not st.user.is_logged_in:
        # 2) Emit an empty div with our ID — this will precede all the stButton widgets
        st.markdown('<div id="login-card">', unsafe_allow_html=True)

        # Title inside our card
        st.markdown("### Welcome Back!", unsafe_allow_html=True)

        # 3) Loop over your providers
        providers = {
            "Microsoft": "microsoft",
            "Google":    "google",
            "GitHub":    "github",
            # just add more here later…
        }
        for name, key in providers.items():
            if st.button(f"Continue with {name}", icon=":material/login:", key=key):
                st.login(key)

        # close our card div
        st.markdown("</div>", unsafe_allow_html=True)

        st.stop()

    st.rerun()

def logout():
    if st.user.is_logged_in:
        st.logout()
        st.rerun()

# Pages
login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
financials = st.Page("Pages/Financials.py", title="Financials", icon=":material/corporate_fare:", default=True)
forecasting = st.Page("Pages/Forecasting.py", title="Forecasting", icon=":material/bolt:")
backtesting = st.Page("Pages/Backtesting.py", title="Backtesting", icon=":material/candlestick_chart:")

# Navigation structure
if st.user.is_logged_in:

    # Check token issue time for early expiration
    issued_at = datetime.datetime.fromtimestamp(st.user.iat)
    if datetime.datetime.now() - issued_at > datetime.timedelta(hours=1):
        st.warning("Session expired: logging you out.")
        st.logout()

    st.sidebar.markdown(f"### Hello, {st.user.name}")
    st.sidebar.markdown(f"### Logged in as {st.user.email}")
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Functions": [financials,forecasting, backtesting]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()