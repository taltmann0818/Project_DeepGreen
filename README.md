# Project DeepGreen

## Ideas
https://github.com/mert-kurttutan/torchview
https://github.com/szagoruyko/pytorchviz

## Overview

Project DeepGreen is built using Python 3.8.8 and incorporates a variety of data science and web development libraries. The primary purpose of this application is to provide stock price prediction capabilities through machine learning models, with a web interface for user access and model training.

Streamlit Deployment Link: https://projectdeepgreen-nyuq3jdtij9etdtdnvy87p.streamlit.app/

## Features

- **Machine Learning**: Leverages a custom ML model, TEMPUS, for United States equities price prediction. Model architecture is built on a hybrid of LSTMs at multiple temporal resolutions, Temporal Convolutional Network, and a Temporal Attention mechanism.
- **Backtesting**: Tools to evaluate model performance on historical market data and generate portfolio performance and tearsheets
- **Interactive Web Interface**: Built with Streamlit for easy navigation and use

# Architecture
![Screenshot 2025-04-26 at 11.31.58 AM.png](Screenshot%202025-04-26%20at%2011.31.58%E2%80%AFAM.png)

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
- **Financial Data**: `yfinance`
- **Machine Learning/AI**: `scikit-learn`, `pytorch`, `ray`
- **Web Development**: `streamlit`, `streamlit_authenticator`
- **Visualization**: `plotly`
