import streamlit as st

st.set_page_config(layout="wide")

def login():
    # All the authentication info is stored in the session_state
    if not st.experimental_user.is_logged_in:
        if st.button("Log in with Microsoft Entra ID"):
            st.login("microsoft")
        st.stop()
    else:
        st.write(f"Hello, {st.experimental_user.name}!")

    st.rerun()

def logout():
    if st.experimental_user.is_logged_in:
        st.logout()
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

training = st.Page(
    "Pages/Forecasting.py", title="Forecasting", icon=":material/bolt:", default=True
)
backtesting = st.Page("Pages/Backtesting.py", title="Backtesting", icon=":material/dashboard:")

if st.experimental_user.is_logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Functions": [training, backtesting]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()

