import streamlit as st

st.set_page_config(layout="wide")

def login():
    left, middle, right = st.columns(3)
    if not st.experimental_user.is_logged_in:
        with middle:
            st.subheader('Welcome back')
            if st.button("Continue with Microsoft Account", icon=":material/login:"):
                st.login("microsoft")
            st.stop()

    st.rerun()

def logout():
    if st.experimental_user.is_logged_in:
        st.logout()
        st.rerun()

welcome = st.Page(f"Hello, {st.experimental_user.name}",title="")
login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

training = st.Page(
    "Pages/Forecasting.py", title="Forecasting", icon=":material/bolt:", default=True
)
backtesting = st.Page(
    "Pages/Backtesting.py", title="Backtesting", icon=":material/dashboard:"
)

if st.experimental_user.is_logged_in:
    pg = st.navigation(
        {
            "": [welcome],
            "Account": [logout_page],
            "Functions": [training, backtesting]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()

