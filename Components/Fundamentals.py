from polygon import RESTClient
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class FundementalData:
    def __init__(self, tickers, indicator_list, years=1, prediction_window=5,**kwargs):
        """
        Initialize the FundementalData class with a ticker symbol and number of past days to fetch.
        """
        self.client = RESTClient(os.environ["POLYGON_API_KEY"])


    def get_market_cap(self, asof=None):
        try:
            if asof is None:
                asof = datetime.today().strftime("%Y-%m-%d")
            return self.client.get_ticker_details(ticker, date=asof).market_cap
        except Exception as e:
            print(f"Error getting market cap: {str(e)}")
    
    def get_close_price(self, asof=None):
        try:
            if asof is None:
                asof = datetime.today().strftime("%Y-%m-%d")
            else:
                asof = datetime.strptime(asof, "%Y-%m-%d").date()
            while True:
                if not np.is_busday(asof.strftime("%Y-%m-%d")):
                    asof -= timedelta(days=1)
                    continue
                try:
                    response = self.client.get_daily_open_close_agg(ticker, date=asof.strftime("%Y-%m-%d"))
                    if response and hasattr(response, 'close'):
                        return response.close
                    else:
                        asof -= timedelta(days=1)
                except Exception as e:
                    asof -= timedelta(days=1)
        except Exception as e:
            print(f"Error getting close price: {str(e)}")
            return None