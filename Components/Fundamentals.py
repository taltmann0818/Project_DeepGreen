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
from typing import Union

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
        try:
            financials = []
            for f in self.client.vx.list_stock_financials(ticker,filing_date_lte=self.current_date.strftime("%Y-%m-%d"),filing_date_gte=self.past_date.strftime("%Y-%m-%d")):
                financials.append(f)
    
            if not financials:
                #logging.warning(f"No financials for {ticker}; skipping.")
                return pd.DataFrame()
    
            financials = pd.DataFrame(financials)
            financials["financials"] = financials["financials"].apply(
                lambda v: v if isinstance(v, dict) else json.loads(v)
            )
            flat = pd.json_normalize(financials["financials"].tolist())
            flat_filtered = (flat.filter(like="value"))
            flat_filtered.index = financials.index
            financials = financials.drop(columns=["financials"]).join(flat_filtered).sort_index()
            
            market_caps = []
            close_prices = []
            sic_codes = []
            for asof in financials['end_date']:
                if self.fetch_market_cap:
                    market_cap, sic_code = self.get_market_cap(ticker, asof)
                    market_caps.append(market_cap)
                    sic_codes.append(sic_code)
                if self.fetch_stock_price:
                    close_prices.append(self.get_close_price(ticker, asof))
            if self.fetch_market_cap:
                financials['market_cap'] = market_caps
                financials['4digit_SIC_code'] = sic_codes
                financials['2digit_SIC_code'] = financials['4digit_SIC_code'].astype(str).str[:2]
            if self.fetch_stock_price:
                financials['share_price'] = close_prices
            financials['ticker'] = ticker
    
            return financials.fillna(0)
            
        except Exception as e:
            #logging.warning(f"[_ticker] failed for {ticker}: {e}")
            return pd.DataFrame()

    def reconstruct_q4(self):
        df = self.financial_data
        metrics = [c for c in df.columns if c.endswith('.value')]
        sub = df[df['fiscal_period'].isin(['Q1', 'Q2', 'Q3', 'FY'])].copy()
        pivot = sub.pivot_table(
            index=['ticker', 'fiscal_year'],
            columns='fiscal_period',
            values=metrics,
            aggfunc='first'
        )
        q4_rows = []
        for (ticker, year), row in pivot.iterrows():
            periods_present = set(
                sub.loc[
                    (sub['ticker'] == ticker) &
                    (sub['fiscal_year'] == year),
                    'fiscal_period'
                ].unique()
            )
            if not {'Q1', 'Q2', 'Q3', 'FY'}.issubset(periods_present):
                continue
            fy_slice = df.loc[
                (df['ticker'] == ticker) &
                (df['fiscal_year'] == year) &
                (df['fiscal_period'] == 'FY')
                ]
            if fy_slice.empty:
                continue
            fy_row = fy_slice.iloc[0]

            q1_vals = row.xs('Q1', level=1)
            q2_vals = row.xs('Q2', level=1)
            q3_vals = row.xs('Q3', level=1)
            fy_vals = row.xs('FY', level=1)
            # compute Q4 = FY – (Q1 + Q2 + Q3)
            q4_vals = fy_vals - (q1_vals + q2_vals + q3_vals)
            new = {'ticker': ticker,'cik': fy_row['cik'],'company_name': fy_row['company_name'],'fiscal_year': year,'fiscal_period': 'Q4','end_date': fy_row['end_date'],
                   'filing_date': fy_row['filing_date'], '2digit_SIC_code': fy_row['2digit_SIC_code'], '4digit_SIC_code': fy_row['4digit_SIC_code'], 'market_cap': fy_row['market_cap']}
            for metric, val in q4_vals.items():
                new[metric] = val
            q4_rows.append(new)

        if q4_rows:
            q4_df = pd.DataFrame(q4_rows)
            self.financial_data = pd.concat(
                [df, q4_df],
                ignore_index=True,
                sort=False
            )

    def get_market_cap(self, ticker, asof=None):
        try:
            if asof is None:
                asof = datetime.today().strftime("%Y-%m-%d")
            market_cap = self.client.get_ticker_details(ticker, date=asof).market_cap
            sic_code = self.client.get_ticker_details(ticker, date=asof).sic_code
            return market_cap, sic_code
        except Exception as e:
            #logging.warning(f"[get_market_cap] {ticker} @ {asof} failed: {e}")
            return e

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
            logging.warning(f"[get_close_price] {ticker} failed for {asof}: {err}")
            return pd.DataFrame()
    
    def get_fundamentals(self):
        fundementals = pd.DataFrame()
        fundementals['ticker'] = self.financial_data['ticker']
        fundementals['2digit_SIC_code'] = self.financial_data['2digit_SIC_code']
        fundementals['4digit_SIC_code'] = self.financial_data['4digit_SIC_code']
        fundementals['filing_date'] = self.financial_data['filing_date']
        fundementals['end_date'] = self.financial_data['end_date']
        fundementals['fiscal_period'] = self.financial_data['fiscal_period']
        fundementals['fiscal_year'] = self.financial_data['fiscal_year']
        fundementals['return_on_equity'] = self.financial_data['income_statement.net_income_loss.value'] / self.financial_data[
            'balance_sheet.equity.value']
        fundementals['operating_margin'] = self.financial_data['income_statement.operating_income_loss.value'] / self.financial_data[
            'income_statement.revenues.value']
        fundementals['debt_to_equity'] = self.financial_data['balance_sheet.liabilities.value'] / self.financial_data[
            'balance_sheet.equity.value']
        fundementals['debt_ratio'] = self.financial_data['balance_sheet.liabilities.value'] / self.financial_data[
            'balance_sheet.assets.value']
        fundementals['current_ratio'] = self.financial_data['balance_sheet.current_assets.value'] / self.financial_data[
            'balance_sheet.current_liabilities.value']
        fundementals['enterprise_value'] = self.financial_data['market_cap'] + self.financial_data['balance_sheet.liabilities.value'] - self.financial_data['balance_sheet.cash.value']
        fundementals['ebit'] = self.financial_data['income_statement.net_income_loss.value'] + self.financial_data[
            'income_statement.income_tax_expense_benefit.value'] + self.financial_data[
                                   'income_statement.interest_and_debt_expense.value']
        fundementals['ebitda'] = fundementals['ebit'] + self.financial_data[
            'income_statement.depreciation_and_amortization.value']
        fundementals['free_cash_flow'] = self.financial_data['cash_flow_statement.net_cash_flow_from_operating_activities.value']-self.financial_data['cash_flow_statement.net_cash_flow_from_investing_activities.value']
        fundementals['free_cash_flow_per_share'] = fundementals['free_cash_flow'] / self.financial_data[
            'income_statement.basic_average_shares.value']
        fundementals['net_margin'] = self.financial_data['income_statement.net_income_loss.value'] / self.financial_data[
            'income_statement.revenues.value']
        fundementals['book_value'] = self.financial_data['balance_sheet.assets.value'] - self.financial_data['balance_sheet.liabilities.value']
        self.financial_data['book_value_per_share'] = fundementals['book_value'] / self.financial_data[
            'income_statement.basic_average_shares.value']
        fundementals['book_value_per_share'] = self.financial_data['book_value_per_share']
        fundementals['price_to_book_ratio'] = self.financial_data['market_cap'] / fundementals['book_value']
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
        fundementals['market_cap'] = self.financial_data['market_cap']
        fundementals['intangible_assets'] = self.financial_data['balance_sheet.intangible_assets.value']

        return fundementals.set_index('end_date')

    def fetch(self):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            all_results = ex.map(lambda t: self._ticker(t), self.tickers)
        dfs = [df for df in all_results if df is not None and not df.empty]
        self.financial_data = pd.concat(dfs, axis=0)
        # *** reconstruct Q4 rows before computing fundamentals ***
        self.reconstruct_q4()
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
    if df is None:
        raise ValueError("You must provide a DataFrame to search.")
    if period == 'Q':
        quarterly = df[(df['ticker'] == ticker) &
                       (df['fiscal_period'].isin(['Q1','Q2','Q3','Q4']))]
        quarterly = quarterly.sort_values('filing_date', ascending=False)
        result = quarterly.head(4)
    else:
        result = df[(df['ticker']==ticker) & (df['fiscal_period']==period)]
        result = result.sort_index(ascending=False).head(limit)

    cols = ['ticker'] + line_items
    avail = [c for c in cols if c in result.columns]
    missing = set(cols) - set(avail)
    if missing:
        print(f"Warning: Missing columns in DataFrame: {missing}")
    return result[avail]

def get_metric_value(df: pd.DataFrame, sic_code: Union[str, int], metric: str):
    df_copy = df.copy()
    df_copy['SIC_str'] = df_copy.index.astype(str)
    lookup_code = str(sic_code)

    matches = df_copy.loc[df_copy['SIC_str'] == lookup_code, metric]
    if not matches.empty:
        val = matches.iloc[0]
    else:
        val = None

    blank_values = {None, '', 'N/A'}
    if pd.isna(val) or val in blank_values:
        val = df_copy.iloc[0][metric]

    return val