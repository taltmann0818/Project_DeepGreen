import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import hashlib
import json

# Add project root to path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from Components.TickerData import TickerData

class DataModule:
    """Simple DataModule for Tempus v2 model"""

    def __init__(self, config: dict = None,
                 data_path: str = 'raw_data_4k.csv', batch_size: int = 256, 
                 max_prediction_length: int = 3, max_encoder_length: int = 30, 
                 days: int = 90, prediction_window: int = 3, 
                 num_workers: Optional[int] = None, use_cache: bool = True, 
                 cache_dir: str = "data_cache",
                 sample_size: int = 100):

        self.config = config
        self.sample_size = sample_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.days = days
        self.prediction_window = prediction_window
        self.num_workers = num_workers or os.cpu_count()
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Tempus v2 specific indicators
        self.indicator_list = ['ema_20', 'ema_50', 'ema_100', 'stoch_rsi_14', 'macd', 'hmm_state', 'Close']

    def _generate_cache_key(self):
        """Generate a unique cache key based on parameters"""
        params = {
            'days': self.days,
            'prediction_window': self.prediction_window,
            'indicators': sorted(self.indicator_list),
            'max_encoder_length': self.max_encoder_length,
            'max_prediction_length': self.max_prediction_length
        }

        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    def _get_cache_paths(self):
        """Get cache file paths"""
        cache_key = self._generate_cache_key()
        return {
            'data': self.cache_dir / f"tempus_v2_data_{cache_key}.parquet",
            'metadata': self.cache_dir / f"tempus_v2_metadata_{cache_key}.json"
        }

    def _is_cache_valid(self):
        """Check if cache files exist and are valid"""
        cache_paths = self._get_cache_paths()

        if not all(path.exists() for path in cache_paths.values()):
            return False

        try:
            # Check if metadata is valid
            with open(cache_paths['metadata'], 'r') as f:
                metadata = json.load(f)

            # Verify cache is for current parameters
            expected_key = self._generate_cache_key()
            return metadata.get('cache_key') == expected_key

        except Exception:
            return False

    def _save_to_cache(self, data: pd.DataFrame):
        """Save processed data to cache"""
        if not self.use_cache:
            return

        cache_paths = self._get_cache_paths()

        try:
            # Save data
            data.to_parquet(cache_paths['data'])

            # Save metadata
            metadata = {
                'cache_key': self._generate_cache_key(),
                'created_at': pd.Timestamp.now().isoformat(),
                'days': self.days,
                'prediction_window': self.prediction_window,
                'indicators': self.indicator_list,
                'data_shape': data.shape
            }

            with open(cache_paths['metadata'], 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Cached data to {cache_paths['data']}")

        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")

    def _load_from_cache(self):
        """Load processed data from cache"""
        if not self.use_cache or not self._is_cache_valid():
            return None

        cache_paths = self._get_cache_paths()

        try:
            data = pd.read_parquet(cache_paths['data'])
            print(f"Loaded cached data from {cache_paths['data']}")
            return data
        except Exception as e:
            print(f"Warning: Could not load from cache: {e}")
            return None

    def prepare_data(self, raw_data=None):
        """Prepare the data for training/inference"""
        # Try to load from cache first
        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

        try:
            if raw_data is not None:
                data_retriever = TickerData(
                    indicator_list=self.indicator_list,
                    days=self.days,
                    prediction_window=self.prediction_window,
                    prediction_mode=True,
                    sample_size=self.sample_size
                )
                processed_data = data_retriever.add_features(df=raw_data)
                processed_data = data_retriever.merge_data(df=processed_data)
                
            else:
                # Use TickerData to process the data
                processed_data = TickerData(
                    indicator_list=self.indicator_list,
                    days=self.days,
                    prediction_window=self.prediction_window,
                    prediction_mode=True,
                    sample_size=self.sample_size
                ).process_all()

            # Handle MultiIndex properly
            if isinstance(processed_data.index, pd.MultiIndex):
                # Reset MultiIndex and handle the level names
                processed_data = processed_data.reset_index()

            # Ensure Ticker column is properly formatted as strings
            if 'Ticker' in processed_data.columns:
                processed_data['Ticker'] = processed_data['Ticker'].astype(str)

            self._save_to_cache(processed_data)
            return processed_data

        except Exception as e:
            print(f"Error processing data: {e}")
            return None

def main():
    """Example usage"""
    datamodule = DataModule(days=90, use_cache=True)
    data = datamodule.prepare_data()

    if data is not None:
        print(f"Processed data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(data.head())
    else:
        print("No data available")


if __name__ == "__main__":
    main()