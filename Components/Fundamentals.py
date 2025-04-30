from polygon import RESTClient
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.easter import easter
from pandas.tseries.holiday import USFederalHolidayCalendar
from concurrent.futures import ThreadPoolExecutor
import json
import logging

class FundementalData:
    def __init__(self, tickers, years=5, **kwargs):
        """
        Initialize the FundamentalData class with a ticker symbol and number of past days to fetch.
        """
        self.client = RESTClient("XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2")
        self.tickers = tickers

        self.current_date = kwargs.get('end_date', datetime.today())
        self.past_date = kwargs.get('start_date', datetime.today() - timedelta(days=years*365))
        self.fetch_market_cap = kwargs.get("fetch_market_cap", True)
        self.fetch_stock_price = kwargs.get("fetch_stock_price", True)
        self.workers = kwargs.get("workers", 20)

    def _ticker(self, ticker):
        financials = []
        market_caps = []
        close_prices = []
        for f in self.client.vx.list_stock_financials(ticker,
                                                      filing_date_lte=self.current_date.strftime("%Y-%m-%d"),
                                                      filing_date_gte=self.past_date.strftime("%Y-%m-%d")):
            financials.append(f)

        financials = pd.DataFrame(financials)
        financials["financials"] = financials["financials"].apply(
            lambda v: v if isinstance(v, dict) else json.loads(v)
        )
        flat = pd.json_normalize(
            financials["financials"].tolist()
        )
        flat_filtered = (
            flat
            .filter(like="value")
            # .dropna(axis=1, how="all")   # drop cols that are all missing
        )

        flat_filtered.index = financials.index
        financials = financials.drop(columns=["financials"]).join(flat_filtered).sort_index()

        if self.fetch_market_cap or self.fetch_stock_price:
            for val in financials['end_date'].values:
                if self.fetch_market_cap:
                    market_caps.append(self.get_market_cap(ticker, val))
                if self.fetch_stock_price:
                    close_prices.append(self.get_close_price(ticker, val))
        if self.fetch_market_cap:
            financials['market_cap'] = market_caps
        if self.fetch_stock_price:
            financials['share_price'] = close_prices
        financials['ticker'] = ticker

        return financials.fillna(0)

    def get_market_cap(self, ticker, asof=None):
        try:
            if asof is None:
                asof = datetime.today().strftime("%Y-%m-%d")
            return self.client.get_ticker_details(ticker, date=asof).market_cap
        except Exception as e:
            print(f"Error getting market cap: {str(e)}")

    def get_close_price(self, ticker, asof=None):
        if asof:
            date_np = np.datetime64(asof)
        else:
            date_np = np.datetime64(datetime.now().strftime("%Y-%m-%d"))
        busday_np = np.busday_offset(date_np,offsets=0,roll='backward',holidays=generate_us_market_holidays(2020,2030))
        date_str = str(busday_np)
        try:
            resp = self.client.get_daily_open_close_agg(ticker, date=date_str)
            return getattr(resp, "close", None)
        except Exception as err:
            logging.error(f"[get_close_price] failed for {ticker} on {date_str}: {err}")
            return None
    
    def get_fundamentals(self):
        fundementals = pd.DataFrame()
        fundementals['ticker'] = self.financial_data['ticker']
        fundementals['end_date'] = self.financial_data['end_date']
        fundementals['fiscal_period'] = self.financial_data['fiscal_period']
        fundementals['fiscal_year'] = self.financial_data['fiscal_year']
        fundementals['return_on_equity'] = self.financial_data['income_statement.net_income_loss.value'] / self.financial_data[
            'balance_sheet.equity.value']
        fundementals['operating_margin'] = self.financial_data['income_statement.operating_income_loss.value'] / self.financial_data[
            'income_statement.revenues.value']
        fundementals['debt_to_equity'] = self.financial_data['balance_sheet.liabilities.value'] / self.financial_data[
            'balance_sheet.equity.value']
        fundementals['enterprise_value'] = self.financial_data['market_cap'] + self.financial_data['balance_sheet.liabilities.value'] - \
                                           self.financial_data['balance_sheet.cash.value']
        fundementals['ebit'] = self.financial_data['income_statement.net_income_loss.value'] + self.financial_data[
            'income_statement.income_tax_expense_benefit.value'] + self.financial_data[
                                   'income_statement.interest_and_debt_expense.value']
        fundementals['ebitda'] = fundementals['ebit'] + self.financial_data[
            'income_statement.depreciation_and_amortization.value']
        fundementals['free_cash_flow'] = self.financial_data['cash_flow_statement.net_cash_flow_from_operating_activities.value']-self.financial_data['cash_flow_statement.net_cash_flow_from_investing_activities.value']
        fundementals['net_margin'] = self.financial_data['income_statement.net_income_loss.value'] / self.financial_data[
            'income_statement.revenues.value']
        self.financial_data['book_value_per_share'] = self.financial_data['balance_sheet.equity.value'] / self.financial_data[
            'income_statement.basic_average_shares.value']
        fundementals['book_value_per_share'] = self.financial_data['book_value_per_share']
        #fundementals['price_to_book_ratio'] = self.financial_data['share_price'] / self.financial_data['book_value_per_share']
        fundementals['current_ratio'] = self.financial_data['balance_sheet.current_assets.value'] / self.financial_data[
            'balance_sheet.liabilities.value']
        #fundementals['price_to_sales'] = self.financial_data['share_price'] / (
        #            self.financial_data['income_statement.revenues.value'] / self.financial_data[
        #        'income_statement.basic_average_shares.value'])
        fundementals['return_on_invested_capital'] = (self.financial_data['income_statement.net_income_loss.value'] - self.financial_data[
            'income_statement.common_stock_dividends.value']) / (self.financial_data['balance_sheet.equity.value'] + self.financial_data[
            'balance_sheet.liabilities.value'])
        fundementals['gross_margin'] = (self.financial_data['income_statement.revenues.value'] - self.financial_data[
            'income_statement.costs_and_expenses.value']) / self.financial_data['income_statement.revenues.value']
        fundementals['working_capital'] = self.financial_data['balance_sheet.current_assets.value'] - self.financial_data[
            'balance_sheet.current_liabilities.value']
        # Line items needed
        fundementals['earnings_per_share'] = self.financial_data['income_statement.basic_earnings_per_share.value']
        fundementals['revenue'] = self.financial_data['income_statement.revenues.value']
        fundementals['net_income'] = self.financial_data['income_statement.net_income_loss.value']
        fundementals['total_assets'] = self.financial_data['balance_sheet.assets.value']
        fundementals['total_liabilities'] = self.financial_data['balance_sheet.liabilities.value']
        fundementals['total_debt'] = self.financial_data['balance_sheet.liabilities.value']
        fundementals['current_assets'] = self.financial_data['balance_sheet.current_assets.value']
        fundementals['current_liabilities'] = self.financial_data['balance_sheet.current_liabilities.value']
        fundementals['dividends_and_other_cash_distributions'] = self.financial_data[
            'income_statement.preferred_stock_dividends_and_other_adjustments.value']
        fundementals['issuance_or_purchase_of_equity_shares'] = self.financial_data[
            'cash_flow_statement.net_cash_flow_from_financing_activities.value']
        fundementals['outstanding_shares'] = self.financial_data['income_statement.basic_average_shares.value']
        fundementals['capital_expenditure'] = self.financial_data[
            'cash_flow_statement.net_cash_flow_from_investing_activities.value']
        fundementals['operating_expense'] = self.financial_data['income_statement.operating_expenses.value']
        fundementals['cash_and_equivalents'] = self.financial_data['balance_sheet.cash.value']
        fundementals['shareholders_equity'] = self.financial_data['balance_sheet.equity.value']
        fundementals['research_and_development'] = self.financial_data['income_statement.research_and_development.value']
        fundementals['goodwill_and_intangible_assets'] = self.financial_data['balance_sheet.intangible_assets.value']
        fundementals['operating_income'] = self.financial_data['income_statement.operating_income_loss.value']
        fundementals['depreciation_and_amortization'] = self.financial_data[
            'income_statement.depreciation_and_amortization.value']
        fundementals['earnings_per_share'] = self.financial_data['income_statement.basic_earnings_per_share.value']
        #fundementals['share_price'] = self.financial_data['share_price']
        fundementals['market_cap'] = self.financial_data['market_cap']
        fundementals['intangible_assets'] = self.financial_data['balance_sheet.intangible_assets.value']

        return fundementals.set_index('end_date')

    def fetch(self):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            financial_dfs = ex.map(lambda t: self._ticker(t), self.tickers)
        self.financial_data = pd.concat(financial_dfs, axis=0)
        self.fundamentals = self.get_fundamentals()

        return self.fundamentals

def generate_us_market_holidays(start_year: int, end_year: int) -> np.ndarray:
    cal = USFederalHolidayCalendar()
    fed = cal.holidays(
        start=f"{start_year}-01-01",
        end  =f"{end_year  }-12-31"
    )
    good_fridays = [
        easter(yr) - timedelta(days=2)
        for yr in range(start_year, end_year + 1)
    ]
    all_holidays = (
        pd.DatetimeIndex(fed).tolist()
        + good_fridays
    )
    return np.array(all_holidays, dtype="datetime64[D]")


def search_line_items(ticker: str, line_items: list, period: str, limit: int = 5, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Filters the financial DataFrame based on ticker, period, line items, and row limit.

    Args:
        ticker (str): The ticker symbol to filter by.
        line_items (list): List of financial line item column names to select.
        period (str, optional): The fiscal period to filter by (e.g., 'annual'). Defaults to 'annual'.
        limit (int, optional): The number of rows to return. Defaults to 5.
        df (pd.DataFrame): The full financial DataFrame to filter.

    Returns:
        pd.DataFrame: The filtered and selected DataFrame.
    """

    if df is None:
        raise ValueError("You must provide a DataFrame to search.")

    # Ensure required columns are present
    required_columns = {'ticker', 'fiscal_period'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Filter by ticker and period
    filtered = df[(df['ticker'] == ticker) & (df['fiscal_period'] == period)]

    # Select the requested columns plus 'ticker' and any index column (if needed)
    columns_to_select = ['ticker'] + line_items
    columns_available = [col for col in columns_to_select if col in filtered.columns]
    missing_columns = set(columns_to_select) - set(columns_available)
    if missing_columns:
        print(f"Warning: Missing columns in DataFrame: {missing_columns}")

    filtered = filtered[columns_available].sort_index(ascending=False)

    # Apply row limit
    return filtered.head(limit)