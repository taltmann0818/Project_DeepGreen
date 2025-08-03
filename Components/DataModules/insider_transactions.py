import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class InsiderTransactions:

    @staticmethod
    def _fetch_insiders_for_tickers(tickers, data_fetcher):

        insiders = data_fetcher.get_insider_transactions_data(tickers)

        return insiders

    @staticmethod
    def build_insider_daily(df_raw: pd.DataFrame,
                            prices: pd.DataFrame,
                            index_prices: pd.DataFrame = None
                           ) -> pd.DataFrame:
        df = df_raw.copy()
        df["direction"] = np.where(df["Text"].str.contains("Purchase", case=False), 1, np.where(df["Text"].str.contains("Sale", case=False), -1, np.nan))
        df["action"] = np.where(df["Text"].str.contains("Purchase", case=False), 'buy', np.where(df["Text"].str.contains("Sale", case=False), 'sell', np.nan))
        df["signed_shares"] = df["direction"] * df["Shares"]
        df["signed_value"]  = df["direction"] * df["Value"]
        df["avg_price"]  = df["signed_value"] / df["signed_shares"]
        df["exec"] = np.where(df["Position"].str.contains("Chief Executive Officer|Chief Financial Offier", case=False), 1,0)

        daily = (
            df
            .groupby(["Ticker", "date"])
            .agg(
                net_inside_shares=("signed_shares", "sum"),
                net_inside_value =("signed_value",  "sum"),
                inside_buyers_ct=("Insider", lambda x: x[df.loc[x.index, "direction"]==1].nunique()),
                inside_sellers_ct=("Insider", lambda x: x[df.loc[x.index, "direction"]==-1].nunique()),
                exec_action_flag=("exec", lambda x: x[df.loc[x.index, "exec"]==1].nunique())
            )
            .reset_index()
        )

        daily = prices.merge(
            daily,
            on=['date', 'Ticker'],
            how='left',
            validate='many_to_one',          # sanity‑check analyst_daily uniqueness
        ).fillna(0)

        roll_vals = [1,5,20]
        daily = daily.sort_values(['Ticker', 'date'])
        for col in roll_vals:
            daily[f'net_inside_value_{col}d'] = (
                daily
                .groupby('Ticker')['net_inside_value']
                .transform(lambda s: s.rolling(col).sum())
            ).fillna(0.0)

        roll_vals = [1,5,20]
        daily = daily.sort_values(['Ticker', 'date'])
        for col in roll_vals:
            daily[f'net_inside_shares_{col}d'] = (
                daily
                .groupby('Ticker')['net_inside_shares']
                .transform(lambda s: s.rolling(col).sum())
            ).fillna(0.0)

        roll_vals = [1,5,10,20]
        for col in roll_vals:
            daily[f'inside_buyers_{col}d'] = (
                daily
                .groupby('Ticker')['inside_buyers_ct']
                .transform(lambda s: s.rolling(col).sum())
            ).fillna(0.0)
        daily['clust_insider_buy_flag_10d'] = np.where(daily["inside_buyers_10d"] >= 3.0, 1,0)

        roll_vals = [1,5,10,20]
        for col in roll_vals:
            daily[f'inside_sellers_{col}d'] = (
                daily
                .groupby('Ticker')['inside_sellers_ct']
                .transform(lambda s: s.rolling(col).sum())
            ).fillna(0.0)
        daily['clust_insider_sell_flag_10d'] = np.where(daily["inside_sellers_10d"] >= 3.0, 1,0)

        daily['net_shares_pct_float'] = daily['net_inside_shares'] / daily['share_count']

        return daily

    @staticmethod
    def add_insider_indicators(df, data_fetcher, indicator_list):
        # Check if any news indicators are requested
        insider_indicators = [
            'net_inside_shares','net_inside_value', 'inside_buyers_ct', 'inside_sellers_ct','exec_action_flag',
            'net_inside_value_1d', 'net_inside_value_5d', 'net_inside_value_20d',
            'net_inside_shares_1d','net_inside_shares_5d', 'net_inside_shares_20d',
            'inside_buyers_1d', 'inside_buyers_5d', 'inside_buyers_20d',
            'inside_sellers_1d', 'inside_sellers_5d', 'inside_sellers_20d',
            'clust_insider_sell_flag_10d','clust_insider_buy_flag_10d','net_shares_pct_float'
        ]
        needed_features = [ind for ind in insider_indicators if ind in indicator_list]

        if not needed_features:
            return df

        # Get unique tickers
        df_reset = df.reset_index()
        unique_tickers = df_reset['Ticker'].unique()

        insiders = InsiderTransactions._fetch_insiders_for_tickers(unique_tickers, data_fetcher)

        df_reset['close_t-1'] = df_reset.groupby('Ticker')['Close'].shift(1)
        df_reset['date'] = pd.to_datetime(df_reset['date']).dt.date
        insider_daily = InsiderTransactions.build_insider_daily(insiders, df_reset[['close_t-1','date','Ticker','share_count']])

        # Merge only the requested indicators
        df = df_reset.merge(
            insider_daily[['Ticker', 'date'] + [col for col in needed_features if col in insider_daily.columns]],
            left_on=['date', 'Ticker'],
            right_on=['date', 'Ticker'],
            how='left'
        ).set_index('date')

        for indicator in needed_features:
            if indicator in insider_daily.columns:
                df[indicator] = df[indicator].fillna(0.0)

        return df
