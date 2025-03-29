# Project DeepGreen

## Overview

Project DeepGreen is built using Python 3.8.8 and incorporates a variety of data science and web development libraries. The primary purpose of this application is to provide price prediction capabilities through machine learning models, with a streamlined web interface for user access and model training.

## Features

- **Machine Learning**: Leverages custom ML model, TEMPUS, for United States equities price prediction 
- **Backtesting**: Tools to evaluate model performance on historical market data and generate portfolio performance and tearsheets
- **Interactive Web Interface**: Built with Streamlit for easy navigation and use

## Requirements

Make sure you have Python 3.8.8 installed. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Project-DeepGreen.git
   cd Project-DeepGreen
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a valid `credentials.yml` file in the project root directory for authentication with the Streamlit webapp.

## Usage

1. Run the application by executing:
   ```bash
   streamlit run app.py
   ```
2. Log in using your credentials
3. Navigate between the Model Training and Backtesting pages using the sidebar menu
4. Use the Training page to create and train price prediction models
5. Use the Backtesting page to evaluate model performance

## Libraries Used

Below are the major libraries used in this project:

- **Core Libraries**: `numpy`, `pandas`
- **Machine Learning/AI**: `scikit-learn`, `pytorch`
- **Web Development**: `streamlit`, `streamlit_authenticator`
- **Visualization**: `plotly`