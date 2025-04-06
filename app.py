import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(layout="wide")

authenticator = stauth.Authenticate(
    dict(st.secrets['credentials']),
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days']
)

def login():
    try:
        auth = authenticator.login('main')
    except Exception as e:
        st.error(e)

    # All the authentication info is stored in the session_state
    if st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
        st.stop()
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')
        st.stop()

    st.rerun()

def logout():
    if st.session_state.get("authentication_status"):
        authenticator.logout("Logout", "unrendered")
        st.session_state["authentication_status"] = None
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

training = st.Page(
    "Pages/Forecasting.py", title="Forecasting", icon=":material/bolt:", default=True
)
backtesting = st.Page("Pages/Backtesting.py", title="Backtesting", icon=":material/dashboard:")

if st.session_state["authentication_status"]:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Functions": [training, backtesting]
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()

