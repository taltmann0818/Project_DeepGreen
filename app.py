import streamlit as st
import datetime

st.set_page_config(page_title="DeepGreen",
                   layout="wide",
                   page_icon=":material/finance_mode:"
)


def login():
    # Inject custom CSS
    st.markdown(
        """
        <style>
        .login-card {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 2rem;
            text-align: center;
        }
        .login-card h3 {
            margin-bottom: 1.5rem;
        }
        .login-card button {
            margin: 0.5rem 0;
            font-size: 1rem;
            border-radius: 6px;
        }
        .login-card button:hover {
            opacity: 0.9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Responsive columns
    with col2:
        if not st.user.is_logged_in:  # Access user state
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            st.markdown("### Welcome Back!", unsafe_allow_html=True)

            # Provider buttons
            providers = {
                "Microsoft": "microsoft",
                "Google": "google",
                "GitHub": "github",
            }
            for name, key in providers.items():
                if st.button(
                        f"Continue with {name}",
                        icon=":material/login:",
                        use_container_width=True
                ):
                    st.login(key)  # Trigger OIDC flow
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