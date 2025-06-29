"""
Calendar and earnings indicators module for TickerData.
Handles date-based features and earnings-related indicators.
"""

import pandas as pd
import numpy as np


class CalendarEarnings:
    """Handles calendar-based and earnings-related indicators"""
    
    @staticmethod
    def add_calendar_indicators(df):
        """
        Add calendar-based indicators to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calendar indicators added
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek

        # Day of month (1-31)
        df['day_of_month'] = df.index.day

        # Days to month end
        # Get the last day of each month for each date
        month_ends = df.index.to_period('M').end_time
        current_dates = df.index
        df['days_to_month_end'] = (month_ends - current_dates).days

        return df

    @staticmethod
    def get_earnings_dates_for_ticker(ticker, data_fetcher=None):
        """
        Get earnings dates for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        data_fetcher : DataFetcher, optional
            DataFetcher instance (for future implementation)
            
        Returns:
        --------
        list or None
            List of earnings dates, or None if not available
            
        Note:
        -----
        This is a placeholder implementation. In a real implementation, 
        you would fetch earnings dates from an earnings calendar API 
        or financial data provider.
        """
        try:
            # Placeholder: In a real implementation, you would fetch earnings dates
            # from an earnings calendar API or financial data provider
            # For now, return None to indicate no earnings data available
            return None
        except Exception as e:
            print(f"Warning: Could not fetch earnings dates for {ticker}: {e}")
            return None

    @staticmethod
    def add_earnings_dummy(df, earnings_dates_dict=None, lookforward_days=10):
        """
        Add earnings dummy indicator (0/1 flag for next N trading days before earnings).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with datetime index and 'Ticker' column
        earnings_dates_dict : dict, optional
            Dictionary mapping tickers to lists of earnings dates
        lookforward_days : int, default=10
            Number of days before earnings to flag
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with earnings dummy indicator added
        """
        # Initialize earnings dummy column
        df['earnings_dummy_10d'] = 0

        if earnings_dates_dict:
            for ticker, earnings_dates in earnings_dates_dict.items():
                if earnings_dates:
                    ticker_mask = df['Ticker'] == ticker
                    ticker_df = df[ticker_mask].copy()

                    for earnings_date in earnings_dates:
                        earnings_date = pd.to_datetime(earnings_date)
                        # Find dates within lookforward_days before earnings
                        start_date = earnings_date - pd.Timedelta(days=lookforward_days)
                        end_date = earnings_date

                        # Set dummy to 1 for dates in the range
                        date_mask = (ticker_df.index >= start_date) & (ticker_df.index <= end_date)
                        df.loc[ticker_mask & date_mask, 'earnings_dummy_10d'] = 1

        return df

    @staticmethod
    def add_calendar_earnings_indicators(df, data_fetcher, indicator_list):
        """
        Add calendar and earnings indicators to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The main dataframe to add indicators to
        data_fetcher : DataFetcher
            DataFetcher instance (for future earnings data fetching)
        indicator_list : set
            Set of indicators to calculate
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calendar and earnings indicators added
        """
        # Process calendar indicators
        calendar_indicators = ['day_of_week', 'day_of_month', 'days_to_month_end']
        needed_calendar = [ind for ind in calendar_indicators if ind in indicator_list]
        if needed_calendar:
            df = CalendarEarnings.add_calendar_indicators(df)

        # Process earnings dummy indicator
        if 'earnings_dummy_10d' in indicator_list:
            # Get unique tickers and fetch earnings dates
            unique_tickers = df['Ticker'].unique()
            earnings_dates_dict = {}
            for ticker in unique_tickers:
                earnings_dates = CalendarEarnings.get_earnings_dates_for_ticker(ticker, data_fetcher)
                if earnings_dates:
                    earnings_dates_dict[ticker] = earnings_dates

            # Add earnings dummy indicator
            df = CalendarEarnings.add_earnings_dummy(df, earnings_dates_dict)

        return df

    @staticmethod
    def get_calendar_statistics(df):
        """
        Get statistics about calendar patterns in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with calendar indicators
            
        Returns:
        --------
        dict
            Dictionary with calendar statistics
        """
        stats = {}
        
        if 'day_of_week' in df.columns:
            stats['day_of_week_counts'] = df['day_of_week'].value_counts().sort_index()
            
        if 'day_of_month' in df.columns:
            stats['day_of_month_stats'] = {
                'mean': df['day_of_month'].mean(),
                'std': df['day_of_month'].std(),
                'min': df['day_of_month'].min(),
                'max': df['day_of_month'].max()
            }
            
        if 'days_to_month_end' in df.columns:
            stats['days_to_month_end_stats'] = {
                'mean': df['days_to_month_end'].mean(),
                'std': df['days_to_month_end'].std(),
                'min': df['days_to_month_end'].min(),
                'max': df['days_to_month_end'].max()
            }
            
        if 'earnings_dummy_10d' in df.columns:
            stats['earnings_dummy_rate'] = df['earnings_dummy_10d'].mean()
            
        return stats

    @staticmethod
    def analyze_calendar_effects(df, return_column='Close'):
        """
        Analyze calendar effects on returns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with calendar indicators and price data
        return_column : str, default='Close'
            Column to calculate returns from
            
        Returns:
        --------
        dict
            Dictionary with calendar effect analysis
        """
        if return_column not in df.columns:
            return {}
            
        # Calculate returns
        df_copy = df.copy()
        df_copy['returns'] = df_copy.groupby('Ticker')[return_column].pct_change()
        
        analysis = {}
        
        # Day of week effects
        if 'day_of_week' in df_copy.columns:
            dow_effects = df_copy.groupby('day_of_week')['returns'].agg(['mean', 'std', 'count'])
            dow_effects.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            analysis['day_of_week_effects'] = dow_effects
            
        # Month end effects
        if 'days_to_month_end' in df_copy.columns:
            # Group by proximity to month end (last 5 days vs others)
            df_copy['near_month_end'] = df_copy['days_to_month_end'] <= 5
            month_end_effects = df_copy.groupby('near_month_end')['returns'].agg(['mean', 'std', 'count'])
            analysis['month_end_effects'] = month_end_effects
            
        # Earnings effects
        if 'earnings_dummy_10d' in df_copy.columns:
            earnings_effects = df_copy.groupby('earnings_dummy_10d')['returns'].agg(['mean', 'std', 'count'])
            analysis['earnings_effects'] = earnings_effects
            
        return analysis