import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def days_since(series: pd.Series) -> pd.Series:
    counter = 0
    out = []
    for flag in series:
        if flag:  # reset at an upgrade or downgrade
            counter = 0
        else:
            counter += 1
        out.append(counter)
    return pd.Series(out, index=series.index)

class AnalystUpdates:

    @staticmethod
    def _fetch_analysts_for_tickers(tickers, data_fetcher):

        analysts = data_fetcher.get_analyst_data(tickers)

        return analysts

    @staticmethod
    def build_analyst_daily(df_broker: pd.DataFrame,
                            prices: pd.Series,
                            ) -> pd.DataFrame:
        """
        df: analyst events with ['date','ticker','toGrade','fromGrade',
                                 'action','priceTarget','prevPriceTarget']
        price_df: daily close prices
        """
        GRADE_MAP = {
            # bullish
            "strong buy": 5, "buy": 4, "overweight": 4, "outperform": 4, "market outperform": 4,
            "sector outperform": 4,
            # neutral
            "equal-weight": 3, "hold": 3, "neutral": 3, "in-line": 3,
            "market perform": 3, "sector perform": 3, "perform": 3,
            "peer perform": 3, "sector weight": 3,
            # bearish
            "strong sell": 1, "sell": 2, "underperform": 2, "underweight": 2, "reduce": 2
        }
        df = df_broker.assign(
            grade_to=df_broker['ToGrade'].str.lower().map(GRADE_MAP),
            grade_from=df_broker['FromGrade'].str.lower().map(GRADE_MAP),
        )
        df['grade_delta'] = df['grade_to'] - df['grade_from']
        df['action_dir'] = df['Action'].map({'up': 1, 'down': -1}).fillna(0)

        # last close (t‑1) to avoid look‑ahead
        df = df.merge(prices[['close_t-1', 'date', 'Ticker']], on=['date', 'Ticker'])
        df['implied_upside'] = df['currentPriceTarget'] / df['close_t-1'] - 1
        df['pt_change'] = df.groupby(['Ticker', 'Firm'])['currentPriceTarget'].pct_change().fillna(0)

        df.sort_values(['Ticker', 'date'], inplace=True)
        df = df.rename(columns={'currentPriceTarget': 'pt'})
        # 3) daily aggregation ----------------------------------------------------
        df['w'] = 1.0
        agg_funs = {
            'grade_to': ['mean'],
            'implied_upside': ['mean'],
            'pt': ['mean', 'std', 'min', 'max'],
            'pt_change': ['mean'],
            'grade_delta': ['sum'],
            'action_dir': lambda s: (s == 1).sum() - (s == -1).sum(),
            'action_dir': 'sum',
            'w': 'sum'  # analyst breadth
        }
        daily = df.groupby(['date', 'Ticker']).agg(agg_funs)
        daily.columns = ['_'.join(c).strip() for c in daily.columns]
        daily = daily.rename(columns={'w_sum': 'analysts_action_sum', 'action_dir_<lambda>': 'action_dir_lambda'})
        daily['pt_range_pct'] = (daily['pt_max'] - daily['pt_min']) / daily['pt_mean']
        daily['pt_cv'] = daily['pt_std'] / daily['pt_mean']

        daily = prices.merge(
            daily,
            on=['date', 'Ticker'],
            how='left',
            validate='many_to_one',  # sanity‑check analyst_daily uniqueness
        ).fillna(0)

        daily['event_flag'] = (daily['action_dir_sum'] != 0).astype(int)
        daily['days_since_event'] = (
            daily
            .groupby('Ticker')['event_flag']
            .apply(days_since)
            .reset_index(level=0, drop=True)
        )

        ema_cols = ['grade_delta_sum', 'implied_upside_mean', 'pt_change_mean']
        daily = daily.sort_values(['Ticker', 'date'])
        for col in ema_cols:
            daily[f'{col}_ema30'] = (
                daily
                .groupby('Ticker')[col]
                .transform(lambda s: s.ewm(span=30, adjust=False).mean())
            )
        daily[f'action_sum_ema5'] = (
            daily
            .groupby('Ticker')['analysts_action_sum']
            .transform(lambda s: s.ewm(span=5, adjust=False).mean())
        )

        return daily.reset_index().drop('index', axis=1)

    @staticmethod
    def add_analyst_indicators(df, data_fetcher, indicator_list):
        # Check if any news indicators are requested
        analyst_indicators = [
            'grade_to_mean', 'implied_upside_mean', 'pt_mean', 'pt_min',
            'pt_max', 'grade_delta_sum', 'action_dir_sum', 'analysts_action_sum',
            'pt_range_pct','event_flag','days_since_event',
            'grade_delta_sum_ema30','implied_upside_mean_ema30','pt_change_mean_ema30',
            'action_sum_ema5'
        ]
        needed_features = [ind for ind in analyst_indicators if ind in indicator_list]

        if not needed_features:
            return df

        # Get unique tickers
        df_reset = df.reset_index()
        unique_tickers = df_reset['Ticker'].unique()

        insiders = AnalystUpdates._fetch_analysts_for_tickers(unique_tickers, data_fetcher)

        df_reset['close_t-1'] = df_reset.groupby('Ticker')['Close'].shift(1)
        df_reset['date'] = pd.to_datetime(df_reset['date']).dt.date
        analyst_daily = AnalystUpdates.build_analyst_daily(insiders, df_reset[['close_t-1', 'date', 'Ticker']])

        # Merge only the requested indicators
        df = df_reset.merge(
            analyst_daily[['Ticker', 'date'] + [col for col in needed_features if col in analyst_daily.columns]],
            left_on=['date', 'Ticker'],
            right_on=['date', 'Ticker'],
            how='left'
        ).set_index('date')

        for indicator in needed_features:
            if indicator in analyst_daily.columns:
                df[indicator] = df[indicator].fillna(0.0)

        return df