import streamlit as st

st.set_page_config(layout="wide")

def login():
    st.markdown(
        """
        <style>
        /* Container to center content */
        .center-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Wrap the login content in a centered container
    st.markdown('<div class="center-container">', unsafe_allow_html=True)

    if not st.experimental_user.is_logged_in:
        with st.container(border=True):
            if st.button("Log in with Microsoft Entra ID"):
                st.login("microsoft")
        st.stop()
    
    st.markdown('</div>', unsafe_allow_html=True)

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

