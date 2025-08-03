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
    def _process_ticker(data_fetcher, ticker):
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
        details = data_fetcher._get_details_for_ticker(ticker)
        sector = SectorAnalysis.map_sic_to_sector(details.get("sic_code"))
        if isinstance(details, dict):
            details["sic_sector"] = sector
        else:
            setattr(details, "sic_sector", sector)
        return ticker, details

    @staticmethod
    def create_detail_indicator(tickers, data_fetcher, max_workers=50):
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
        out = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(SectorAnalysis._process_ticker, data_fetcher, tk): tk
                for tk in tickers
            }
            for fut in as_completed(futures):
                try:
                    tk, details = fut.result()
                    out[tk] = details
                except Exception as exc:
                    ticker = futures[fut]
                    out[ticker] = {'sic_sector': 'Unknown','asset_type': 'Unknown', 'sic_code': 'Unknown', 'employees': 0, 'share_count': 0}

        return out


    @staticmethod
    def add_detail_indicators(df, data_fetcher, indicator_list):
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
        temp_df = df.reset_index()
        if 'Ticker' in temp_df.columns:
            unique_tickers = temp_df['Ticker'].unique()
            details = SectorAnalysis.create_detail_indicator(unique_tickers, data_fetcher)
            details_df = pd.DataFrame(details).T.reset_index().rename(columns={'index':'Ticker'})
            temp_df = temp_df.merge(details_df, on='Ticker')

            temp_df = temp_df[temp_df['asset_type'] == 'CS']
            temp_df = temp_df[temp_df['sic_sector'] != 'Unknown']

        else:
            print("Warning: Could not find 'Ticker' column or index level. Skipping sic_sector indicator.")
            
        return temp_df