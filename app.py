import streamlit as st
import datetime

st.set_page_config(page_title="DeepGreen",
                   layout="wide",
                   page_icon=":material/finance_mode:"
)

def login():
    left, middle, right = st.columns(3)
    if not st.user.is_logged_in:
        with middle:
            st.subheader('Welcome back')
            if st.button("Continue with Microsoft Account", icon=":material/login:"):
                st.login("microsoft")
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
    st.sidebar.subheader(st.user.exp)

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