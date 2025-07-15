from polygon import RESTClient
from Components.polygon_client_patch import patch_polygon_client
patch_polygon_client(max_pool_size=50)
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
import math

class FundementalData:
    def __init__(self, tickers, days=252, **kwargs):
        """
        Initialize the FundamentalData class with a ticker symbol and number of past days to fetch.
        """
        self.client = RESTClient()
        self.tickers = tickers

        self.current_date = kwargs.get('end_date', datetime.today())
        self.past_date = kwargs.get('start_date', datetime.today() - timedelta(days))
        self.fetch_market_cap = kwargs.get("fetch_market_cap", True)
        self.fetch_stock_price = kwargs.get("fetch_stock_price", True)
        self.workers = kwargs.get("workers", 10)

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
                    market_cap, sic_code = self.get_market_cap(ticker)
                    market_caps.append(market_cap)
                    sic_codes.append(sic_code)
                if self.fetch_stock_price:
                    close_prices.append(self.get_close_price(ticker))
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
                   'filing_date': fy_row['filing_date'], '2digit_SIC_code': fy_row['2digit_SIC_code'], '4digit_SIC_code': fy_row['4digit_SIC_code'], 'market_cap': fy_row['market_cap'],'share_price': fy_row['share_price']}
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
        busday_np = str(np.busday_offset(date_np,offsets=0,roll='backward',holidays=generate_us_market_holidays(2020,2030)))
        date_str = str(busday_np)
        try:
            resp = self.client.get_daily_open_close_agg(ticker, date=date_str)
            close = getattr(resp, "close", None)
            return close if close is not None else getattr(resp, "open", None)
        except Exception as err:
            logging.warning(f"[get_close_price] {ticker} failed for {asof}: {err}")
            return pd.DataFrame()

    def _safe_div(self, num: pd.Series, den: pd.Series) -> pd.Series:
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
    
        res = num.div(den.replace(0, np.nan))
        return res.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
    def get_fundamentals(self) -> pd.DataFrame:
        fd = self.financial_data             # alias for brevity
        
        out = pd.DataFrame({
            "Ticker":         fd["ticker"],
            "filing_date":    fd["filing_date"],
            "date":           fd["end_date"],
            "fiscal_period":  fd["fiscal_period"],
            "fiscal_year":    fd["fiscal_year"],

            # profitability & leverage
            "roe":            self._safe_div(fd["income_statement.net_income_loss.value"],
                                              fd["balance_sheet.equity.value"]),
            "dte":            self._safe_div(fd["balance_sheet.liabilities.value"],
                                              fd["balance_sheet.equity.value"]),
            "debt_ratio":     self._safe_div(fd["balance_sheet.liabilities.value"],
                                              fd["balance_sheet.assets.value"]),
            "current_ratio":  self._safe_div(fd["balance_sheet.current_assets.value"],
                                              fd["balance_sheet.current_liabilities.value"]),
            "op_mrg":         self._safe_div(fd["income_statement.operating_income_loss.value"],
                                              fd["income_statement.revenues.value"]),
            "net_mrg":        self._safe_div(fd["income_statement.net_income_loss.value"],
                                              fd["income_statement.revenues.value"]),
            "gross_mrg":      self._safe_div(
                                   fd["income_statement.revenues.value"]
                                   - fd["income_statement.costs_and_expenses.value"],
                                   fd["income_statement.revenues.value"]),
        })

        # enterprise-value multiples
        enterprise_value = (
              fd["market_cap"]
            + fd["balance_sheet.liabilities.value"]
            - fd["balance_sheet.cash.value"]
        )
        ebitda = (
              fd["income_statement.net_income_loss.value"]
            + fd["income_statement.income_tax_expense_benefit.value"]
            + fd["income_statement.interest_and_debt_expense.value"]
            + fd["income_statement.depreciation_and_amortization.value"]
        )
        out["ev_ebitda"] = self._safe_div(enterprise_value, ebitda)

        # ROIC
        out["roic"] = self._safe_div(
            fd["income_statement.net_income_loss.value"]
            - fd["income_statement.common_stock_dividends.value"],
            fd["balance_sheet.equity.value"] + fd["balance_sheet.liabilities.value"],
        )

        # per-share metrics
        book_value = (
              fd["balance_sheet.assets.value"]
            - fd["balance_sheet.intangible_assets.value"]
            - fd["balance_sheet.liabilities.value"]
        )
        shares = fd["income_statement.basic_average_shares.value"]
        out["bv_per_share"]  = self._safe_div(book_value, shares)

        free_cash_flow = (
              fd["cash_flow_statement.net_cash_flow_from_operating_activities.value"]
            - fd["cash_flow_statement.net_cash_flow_from_investing_activities.value"]
        )
        out["fcf_per_share"] = self._safe_div(free_cash_flow, shares)

        out["eps"]          = fd["income_statement.basic_earnings_per_share.value"]

        # valuation ratios
        fd["share_price"] = (
            fd["share_price"]
            .apply(lambda v: np.nan if isinstance(v, pd.DataFrame) and v.empty else v)
            .pipe(pd.to_numeric, errors="coerce") 
        )
        out["pe_ratio"]   = self._safe_div(fd["share_price"], out["eps"])
        sales_per_sh     = self._safe_div(fd["income_statement.revenues.value"], fd["share_price"])
        out["ps_ratio"]  = self._safe_div(fd["share_price"], sales_per_sh)
        out["pb_ratio"]  = self._safe_div(fd["share_price"], out["bv_per_share"])
        out["pfcf_ratio"] = self._safe_div(fd["share_price"], out["fcf_per_share"])

        # Graham number (set to 0 if eps * bv_per_share is negative)
        graham_product = 22.5 * out["eps"] * out["bv_per_share"]
        out["graham_number"] = np.where(graham_product <= 0.0, 0.0, np.sqrt(graham_product))

        # interest-coverage
        interest_exp = fd["income_statement.interest_expense_operating.value"].abs()
        out["interest_coverage"] = self._safe_div(
            fd["income_statement.net_income_loss.value"]
          + fd["income_statement.income_tax_expense_benefit.value"]
          + fd["income_statement.interest_and_debt_expense.value"],
            interest_exp,
        )

        # final tidy-up
        return out

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
    
    sic_code = result['4digit_SIC_code'][0] if result['2digit_SIC_code'][0] == '73' else result['2digit_SIC_code'][0]

    cols = ['ticker'] + line_items
    avail = [c for c in cols if c in result.columns]
    missing = set(cols) - set(avail)
    if missing:
        print(f"Warning: Missing columns in DataFrame: {missing}")
    return result[avail], sic_code

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