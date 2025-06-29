"""
Market news module for TickerData.
Contains news sentiment analysis and feature engineering.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class MarketNews:
    """Collection of market news indicators and sentiment analysis"""

    @staticmethod
    def _fetch_news_for_ticker(ticker, data_fetcher, start_date, end_date, full_dates):
        """
        Helper method to fetch news data for a single ticker for concurrent execution.

        Parameters:
        -----------
        ticker : str
            Ticker symbol to fetch news for
        data_fetcher : DataFetcher
            DataFetcher instance to get news data
        start_date : datetime
            Start date for news data
        end_date : datetime
            End date for news data
        full_dates : pd.DatetimeIndex
            Full date range for alignment

        Returns:
        --------
        tuple
            (ticker, news_data) tuple
        """
        try:
            news_data = data_fetcher.get_news_for_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                full_dates=full_dates
            )
            return ticker, news_data
        except Exception as e:
            print(f"Warning: Could not fetch news data for {ticker}: {e}")
            # Create empty news data for this ticker
            empty_news = pd.DataFrame(0,
                                      index=full_dates,
                                      columns=["positive", "neutral", "negative"])
            empty_news.index.name = "date"
            empty_news["Ticker"] = ticker
            return ticker, empty_news

    @staticmethod
    def add_news_indicators(df, data_fetcher, indicator_list):
        """
        Add news sentiment indicators to the dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            The main dataframe to add indicators to
        data_fetcher : DataFetcher
            DataFetcher instance to get news data
        indicator_list : set
            Set of indicators to calculate

        Returns:
        --------
        pd.DataFrame
            DataFrame with news sentiment indicators added
        """
        # Check if any news indicators are requested
        news_indicators = [
            'sent_count_T0', 'sent_mean_T0', 'sent_vol_T0', 'sent_sum_3d', 
            'buzz_score_T0', 'pos_ratio_T0', 'neg_ratio_T0'
        ]
        needed_news = [ind for ind in news_indicators if ind in indicator_list]

        if not needed_news:
            return df

        # Get unique tickers - handle both column and index cases
        if 'Ticker' in df.columns:
            unique_tickers = df['Ticker'].unique()
        elif isinstance(df.index, pd.MultiIndex) and 'Ticker' in df.index.names:
            unique_tickers = df.index.get_level_values('Ticker').unique()
        else:
            # Try to reset index to access Ticker
            df_reset = df.reset_index()
            if 'Ticker' in df_reset.columns:
                unique_tickers = df_reset['Ticker'].unique()
            else:
                print("Warning: Could not find Ticker information in DataFrame")
                return df

        start_date = data_fetcher.start_date
        end_date = data_fetcher.end_date

        # Create full date range for alignment
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Fetch news data for all tickers
        all_news_data = []
        with ThreadPoolExecutor(max_workers=40) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    MarketNews._fetch_news_for_ticker,
                    ticker, data_fetcher, start_date, end_date, full_dates
                ): ticker
                for ticker in unique_tickers
            }

            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                try:
                    ticker, news_data = future.result()
                    if not news_data.empty:
                        all_news_data.append(news_data)
                except Exception as exc:
                    ticker = future_to_ticker[future]
                    print(f'News data fetch for ticker {ticker} generated an exception: {exc}')
                    # Create empty news data for failed ticker
                    empty_news = pd.DataFrame(0,
                                              index=full_dates,
                                              columns=["positive", "neutral", "negative"])
                    empty_news.index.name = "date"
                    empty_news["Ticker"] = ticker
                    all_news_data.append(empty_news)

        if all_news_data:
            # Combine all news data
            combined_news = pd.concat(all_news_data, ignore_index=False)
            combined_news = combined_news.reset_index()

            # Ensure timezone consistency before processing
            if 'date' in combined_news.columns:
                # Handle mixed timezone scenarios more robustly
                try:
                    # First, try to convert any timezone-aware values to naive
                    if combined_news['date'].dtype == 'object':
                        # Convert each value individually to handle mixed types
                        normalized_dates = []
                        for date_val in combined_news['date']:
                            if pd.isna(date_val):
                                normalized_dates.append(pd.NaT)
                            else:
                                # Convert to datetime if not already
                                dt = pd.to_datetime(date_val) if not isinstance(date_val, pd.Timestamp) else date_val
                                # Remove timezone if present
                                if hasattr(dt, 'tz') and dt.tz is not None:
                                    dt = dt.tz_localize(None)
                                normalized_dates.append(dt)
                        combined_news['date'] = normalized_dates
                    else:
                        # Already datetime type, just handle timezone
                        combined_news['date'] = pd.to_datetime(combined_news['date'])
                        if hasattr(combined_news['date'].dtype, 'tz') and combined_news['date'].dt.tz is not None:
                            combined_news['date'] = combined_news['date'].dt.tz_localize(None)
                except Exception as e:
                    print(f"Warning: Issue with date conversion: {e}")
                    # Fallback: force conversion by removing timezone info from all values
                    combined_news['date'] = pd.to_datetime(combined_news['date'], errors='coerce').dt.tz_localize(None)

            # Apply sentiment feature engineering
            news_with_features = MarketNews.add_news_features_from_counts(combined_news)

            # Merge with main dataframe
            news_with_features['date'] = pd.to_datetime(news_with_features['date']).dt.tz_localize(None)
            news_with_features = news_with_features.set_index('date')

            # Merge only the requested indicators
            for indicator in needed_news:
                if indicator in news_with_features.columns:
                    indicator_data = news_with_features[['Ticker', indicator]].reset_index()
                    df = df.reset_index().merge(
                        indicator_data, 
                        left_on=['date', 'Ticker'], 
                        right_on=['date', 'Ticker'], 
                        how='left'
                    ).set_index('date')
                    df[indicator] = df[indicator].fillna(0.0)

        return df

    @staticmethod
    def add_news_features_from_counts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer daily sentiment features from *counts* of positive, negative,
        and neutral news headlines.

        Required columns: ['date','ticker','pos_sum','neg_sum','neu_sum'] where
          *_sum are INT counts per (ticker, day).
        """
        # Ensure proper data types before sorting
        df = df.copy()

        # Convert Ticker to string to avoid categorical issues
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].astype(str)

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Sort with proper error handling
        try:
            df = df.sort_values(["Ticker", "date"])
        except TypeError:
            # If sorting fails, try converting categorical columns to strings
            for col in ["Ticker", "date"]:
                if col in df.columns and hasattr(df[col], 'cat'):
                    df[col] = df[col].astype(str)
            df = df.sort_values(["Ticker", "date"])

        # ------------------------------------------------------------------
        # 1) Total buzz and bullish-bearish imbalance ratio
        # ------------------------------------------------------------------
        df["sent_count_T0"] = df[["positive", "negative", "neutral"]].sum(axis=1)

        # imbalance ratio  (pos − neg) / total;  0 if no news
        df["sent_mean_T0"] = (
                (df["positive"] - df["negative"])
                / df["sent_count_T0"].replace(0, np.nan)
        ).fillna(0.0)

        # ------------------------------------------------------------------
        # 2) Rolling metrics
        # ------------------------------------------------------------------
        # 3-day cumulative imbalance (persistence of tone)
        df["sent_sum_3d"] = (
            df.groupby("Ticker")["sent_mean_T0"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum())
        )

        # 5-day volatility of imbalance ratio (disagreement)
        df["sent_vol_T0"] = (
            df.groupby("Ticker")["sent_mean_T0"]
            .transform(lambda x: x.rolling(5, min_periods=2).std())
            .fillna(0.0)
        )

        # abnormal buzz z-score vs. 30-day baseline
        roll = df.groupby("Ticker")["sent_count_T0"]
        mean30 = roll.transform(lambda x: x.rolling(30, min_periods=5).mean())
        std30 = roll.transform(lambda x: x.rolling(30, min_periods=5).std())
        df["buzz_score_T0"] = ((df["sent_count_T0"] - mean30) / std30).fillna(0.0)

        # ------------------------------------------------------------------
        # 3) Optional extra ratios (if you want more features)
        # ------------------------------------------------------------------
        df["pos_ratio_T0"] = (
                df["positive"] / df["sent_count_T0"].replace(0, np.nan)
        ).fillna(0.0)
        df["neg_ratio_T0"] = (
                df["negative"] / df["sent_count_T0"].replace(0, np.nan)
        ).fillna(0.0)

        # Fill any residual NA with zero so they qualify as "known" features
        sentiment_cols = [
            "sent_count_T0", "sent_mean_T0", "sent_vol_T0",
            "sent_sum_3d", "buzz_score_T0",
            "pos_ratio_T0", "neg_ratio_T0"
        ]
        df[sentiment_cols] = df[sentiment_cols].fillna(0.0)

        return df
