
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class ShortSales:

    @staticmethod
    def _fetch_shorts_for_ticker(ticker, data_fetcher, start_date, end_date):
        try:
            shorts_data = data_fetcher.get_shorts_for_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            return ticker, shorts_data
        except Exception as e:
            print(f"Warning: Could not fetch shorts data for {ticker}: {e}")
            shorts_data = None
            return ticker, shorts_data

    @staticmethod
    def add_short_features(raw):
        """Add engineered short-selling features for TFT ingestion."""
        df = raw.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['Ticker', 'date'])

        # --- 1. Level ratios ------------------------------------------------------
        df['short_vol_ratio'] = df['short_volume'] / df['total_volume']
        df['nasdaq_carteret_ratio'] = df['nasdaq_carteret_short_volume'] / df['total_volume']
        df['nyse_ratio'] = df['nyse_short_volume'] / df['total_volume']
        df['exempt_ratio'] = df['exempt_volume'] / df['short_volume'].replace(0, np.nan)
        df['non_exempt_ratio'] = df['non_exempt_volume'] / df['total_volume']

        # --- 2. Rolling stats & deltas -------------------------------------------
        g = df.groupby('Ticker')
        for k in (5, 10, 20):
            r = g['short_vol_ratio']
            df[f'svr_ma_{k}'] = r.transform(lambda x: x.rolling(k).mean())
            df[f'svr_std_{k}'] = r.transform(lambda x: x.rolling(k).std(ddof=0))
            df[f'svr_z_{k}'] = (df['short_vol_ratio'] - df[f'svr_ma_{k}']) / df[f'svr_std_{k}']
            df[f'svr_pctchg_{k}'] = r.transform(lambda x: x.pct_change(k))

        # --- 3. Liquidity-adjusted days-to-cover ---------------------------------
        df['avg_vol_20'] = g['total_volume'].transform(lambda x: x.rolling(20).mean())
        df['days_to_cover_est'] = df['short_volume'] / df['avg_vol_20']

        # --- 4. Event / threshold flags ------------------------------------------
        df['high_short_flag'] = (
                (df['short_vol_ratio'] > 0.20) &
                (df['days_to_cover_est'] > 3)
        ).astype(int)

        df['exempt_spike_flag'] = (
            df.groupby('Ticker')['exempt_ratio']
            .transform(lambda x: (x > x.quantile(0.95)).astype(int))
        )

        # Keep only useful columns for TFT
        keep = ['Ticker', 'date'] + [c for c in df.columns if c not in raw.columns]
        return df[keep].fillna(0.0)

    @staticmethod
    def add_shorts_indicators(df, data_fetcher, indicator_list):
        # Check if any news indicators are requested
        shorts_indicators = [
            'short_vol_ratio', 'nasdaq_carteret_ratio', 'nyse_ratio', 'exempt_ratio', 'non_exempt_ratio',
            'svr_ma_5', 'svr_std_5', 'svr_z_5','svr_pctchg_5',
            'svr_ma_10','svr_std_10','svr_z_10','svr_pctchg_10',
            'svr_ma_20', 'svr_std_20', 'svr_z_20', 'svr_pctchg_20',
            'avg_vol_20', 'days_to_cover_est', 'high_short_flag', 'exempt_spike_flag',
        ]
        needed_features = [ind for ind in shorts_indicators if ind in indicator_list]

        if not needed_features:
            return df

        # Get unique tickers
        df_reset = df.reset_index()
        unique_tickers = df_reset['Ticker'].unique()

        start_date = data_fetcher.start_date
        end_date = data_fetcher.end_date

        # Fetch news data for all tickers
        all_data = []
        with ThreadPoolExecutor(max_workers=40) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    ShortSales._fetch_shorts_for_ticker,
                    ticker, data_fetcher, start_date, end_date
                ): ticker
                for ticker in unique_tickers
            }

            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                try:
                    ticker, news_data = future.result()
                    if not news_data.empty:
                        all_data.append(news_data)
                except Exception as exc:
                    # Create empty data for failed ticker
                    empty_data = pd.DataFrame()
                    all_data.append(empty_data)

        if all_data:
            # Combine all news data
            combined_data = pd.concat(all_data, ignore_index=False)
            combined_data = combined_data.reset_index()

            shorts_with_features = ShortSales.add_short_features(combined_data)

            df_reset = df.reset_index()
            if 'date' in df_reset.columns:
                df_reset['date'] = pd.to_datetime(df_reset['date']).dt.tz_localize(None)

            # Merge only the requested indicators
            df = df_reset.merge(
                shorts_with_features[['Ticker', 'date'] + [col for col in needed_features if col in shorts_with_features.columns]],
                left_on=['date', 'Ticker'],
                right_on=['date', 'Ticker'],
                how='left'
            ).set_index('date')

            for indicator in needed_features:
                if indicator in shorts_with_features.columns:
                    df[indicator] = df[indicator].fillna(0.0)

        return df

