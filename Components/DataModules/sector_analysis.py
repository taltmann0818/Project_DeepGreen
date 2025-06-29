"""
Sector analysis module for TickerData.
Handles SIC code mapping and sector-related functionality.
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class SectorAnalysis:
    """Handles sector classification and analysis based on SIC codes"""
    
    @staticmethod
    def get_sic_to_sector_mapping():
        """Get the mapping from SIC code ranges to sector names"""
        return {
            # Technology
            range(3570, 3580): 'Technology',  # Computer and office equipment
            range(3600, 3700): 'Technology',  # Electronic equipment
            range(7370, 7380): 'Technology',  # Computer programming and data processing

            # Financial Services
            range(6000, 6100): 'Financial',   # Banking
            range(6200, 6300): 'Financial',   # Security and commodity brokers
            range(6300, 6400): 'Financial',   # Insurance carriers
            range(6700, 6800): 'Financial',   # Holding and investment offices

            # Healthcare
            range(2830, 2840): 'Healthcare',  # Drugs
            range(3840, 3850): 'Healthcare',  # Surgical and medical instruments
            range(8000, 8100): 'Healthcare',  # Health services

            # Energy
            range(1300, 1400): 'Energy',      # Oil and gas extraction
            range(2900, 3000): 'Energy',      # Petroleum refining

            # Consumer Discretionary
            range(2300, 2400): 'Consumer_Discretionary',  # Apparel
            range(3700, 3800): 'Consumer_Discretionary',  # Transportation equipment
            range(5000, 5200): 'Consumer_Discretionary',  # Wholesale trade
            range(5200, 5600): 'Consumer_Discretionary',  # Retail trade

            # Consumer Staples
            range(2000, 2100): 'Consumer_Staples',  # Food products
            range(5400, 5500): 'Consumer_Staples',  # Food stores

            # Industrials
            range(1500, 1800): 'Industrials',  # Construction
            range(3300, 3400): 'Industrials',  # Primary metal industries
            range(3400, 3500): 'Industrials',  # Fabricated metal products
            range(3500, 3600): 'Industrials',  # Industrial machinery

            # Materials
            range(1000, 1500): 'Materials',    # Mining
            range(2600, 2700): 'Materials',    # Paper and allied products
            range(2800, 2900): 'Materials',    # Chemicals

            # Utilities
            range(4900, 5000): 'Utilities',    # Electric, gas, and sanitary services

            # Real Estate
            range(6500, 6600): 'Real_Estate',  # Real estate

            # Communication Services
            range(4800, 4900): 'Communication',  # Communications
        }

    @staticmethod
    def map_sic_to_sector(sic_code):
        """
        Map a SIC code to its corresponding sector.
        
        Parameters:
        -----------
        sic_code : int or str
            The SIC code to map
            
        Returns:
        --------
        str
            The sector name, or 'Other' if not found, or 'Unknown' if invalid
        """
        if sic_code is None:
            return 'Unknown'
            
        try:
            # Convert SIC code to integer for range comparison
            sic_code_int = int(sic_code)
            sic_to_sector = SectorAnalysis.get_sic_to_sector_mapping()
            
            for sic_range, sector_name in sic_to_sector.items():
                if isinstance(sic_range, range) and sic_code_int in sic_range:
                    return sector_name
                    
            return 'Other'  # Default sector for valid SIC codes not in our mapping
            
        except (ValueError, TypeError):
            # If SIC code can't be converted to int, mark as Unknown
            return 'Unknown'

    @staticmethod
    def _process_ticker(ticker, data_fetcher):
        """
        Helper method to process a single ticker for concurrent execution.

        Parameters:
        -----------
        ticker : str
            Ticker symbol to process
        data_fetcher : DataFetcher
            DataFetcher instance to get SIC codes

        Returns:
        --------
        tuple
            (ticker, sector) tuple
        """
        sic_code = data_fetcher.get_sic_code_for_ticker(ticker)
        sector = SectorAnalysis.map_sic_to_sector(sic_code)
        return ticker, sector

    @staticmethod
    def create_sector_indicator(tickers, data_fetcher, max_workers=None):
        """
        Create sector indicator based on SIC codes for a list of tickers using ThreadPoolExecutor.

        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        data_fetcher : DataFetcher
            DataFetcher instance to get SIC codes
        max_workers : int, optional
            Maximum number of worker threads. If None, defaults to min(32, len(tickers) + 4)

        Returns:
        --------
        dict
            Dictionary mapping tickers to their sectors
        """
        ticker_sectors = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(SectorAnalysis._process_ticker, ticker, data_fetcher): ticker
                for ticker in tickers
            }

            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                try:
                    ticker, sector = future.result()
                    ticker_sectors[ticker] = sector
                except Exception as exc:
                    ticker = future_to_ticker[future]
                    #print(f'Ticker {ticker} generated an exception: {exc}')
                    ticker_sectors[ticker] = 'Unknown'

        return ticker_sectors


    @staticmethod
    def add_sector_indicators(df, data_fetcher, indicator_list):
        """
        Add sector-related indicators to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The main dataframe to add indicators to
        data_fetcher : DataFetcher
            DataFetcher instance to get SIC codes
        indicator_list : set
            Set of indicators to calculate
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with sector indicators added
        """
        # Process SIC sector indicator
        # Process SIC sector indicator
        if 'sic_sector' in indicator_list:
            # Check if Ticker is in columns or index
            if 'Ticker' in df.columns:
                unique_tickers = df['Ticker'].unique()
                ticker_sectors = SectorAnalysis.create_sector_indicator(unique_tickers, data_fetcher)
                df['sic_sector'] = df['Ticker'].map(ticker_sectors)
            elif isinstance(df.index, pd.MultiIndex) and 'Ticker' in df.index.names:
                # Ticker is part of a MultiIndex
                unique_tickers = df.index.get_level_values('Ticker').unique()
                ticker_sectors = SectorAnalysis.create_sector_indicator(unique_tickers, data_fetcher)
                # Create the mapping using the index level
                df['sic_sector'] = df.index.get_level_values('Ticker').map(ticker_sectors)
            elif df.index.name == 'Ticker':
                # Ticker is a single-level index
                unique_tickers = df.index.unique()
                ticker_sectors = SectorAnalysis.create_sector_indicator(unique_tickers, data_fetcher)
                df['sic_sector'] = df.index.map(ticker_sectors)
            else:
                # Fallback: try to reset index to find Ticker
                temp_df = df.reset_index()
                if 'Ticker' in temp_df.columns:
                    unique_tickers = temp_df['Ticker'].unique()
                    ticker_sectors = SectorAnalysis.create_sector_indicator(unique_tickers, data_fetcher)
                    df['sic_sector'] = temp_df['Ticker'].map(ticker_sectors).values
                else:
                    # If we still can't find Ticker, skip this indicator
                    print("Warning: Could not find 'Ticker' column or index level. Skipping sic_sector indicator.")
        return df

    @staticmethod
    def get_sector_statistics(df, sector_column='sic_sector'):
        """
        Get statistics about sector distribution in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing sector information
        sector_column : str, default='sic_sector'
            Name of the column containing sector information
            
        Returns:
        --------
        pd.Series
            Series with sector counts
        """
        if sector_column in df.columns:
            return df[sector_column].value_counts()
        else:
            return pd.Series(dtype=int)

    @staticmethod
    def get_sector_performance(df, sector_column='sic_sector', return_column='Close'):
        """
        Calculate performance metrics by sector.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing sector and price information
        sector_column : str, default='sic_sector'
            Name of the column containing sector information
        return_column : str, default='Close'
            Name of the column to calculate returns from
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with sector performance metrics
        """
        if sector_column not in df.columns or return_column not in df.columns:
            return pd.DataFrame()
            
        # Calculate returns
        df_copy = df.copy()
        df_copy['returns'] = df_copy.groupby('Ticker')[return_column].pct_change()
        
        # Group by sector and calculate metrics
        sector_stats = df_copy.groupby(sector_column)['returns'].agg([
            'mean', 'std', 'count'
        ]).round(4)
        
        sector_stats.columns = ['avg_return', 'volatility', 'observations']
        
        return sector_stats