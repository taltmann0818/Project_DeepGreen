import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to credentials.yml
credentials_path = os.path.join(script_dir, "credentials.yml")

# Load the config
with open(credentials_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
stauth.Hasher.hash_passwords(config['credentials'])

# Save the Hashed Credentials to our config file
with open(credentials_path, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
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
    "Pages/Training.py", title="Model Training", icon=":material/bolt:", default=True
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
