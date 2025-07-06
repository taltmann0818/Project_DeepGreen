import pandas as pd
import numpy as np
from typing import Tuple, Optional
import torch
import hashlib
from pathlib import Path
import json
import os
from Components.TickerData import TickerData
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Components.TickerData import TickerData

class DataModule:
    """
    Data module for TFT model training that handles data loading, preprocessing,
    and creation of PyTorch Lightning dataloaders.
    """

    def __init__(
        self,
        config: dict = None,
        batch_size: int = None,
        max_prediction_length: int = None,
        max_encoder_length: int = None,
        days: int = 252,
        prediction_window: int = 3,
        num_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: str = "data_cache",
        sample_size: int = 100,
    ):
        self.config = config

        # Use config values with fallback to parameters
        self.batch_size = batch_size or config.get("BATCH_SIZE", 256)
        self.sample_size = sample_size
        self.max_prediction_length = max_prediction_length or config.get("DECODER_LEN", 3)
        self.max_encoder_length = max_encoder_length or config.get("ENCODER_LEN", 30)
        self.days = days
        self.prediction_window = prediction_window
        self.num_workers = num_workers or max(1, os.cpu_count() // 2)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Get feature columns from config
        self.static_categoricals = config.get("STATIC_CATS", [])
        self.time_varying_known_reals = config.get("TV_KNOWN_REAL", [])
        self.time_varying_unknown_reals = config.get("TV_UNKNOWN_REAL", [])

        # Model path from config
        self.model_path = config.get("ONNX_MODEL_PATH", "")

        # Additional columns needed for processing (target and identifier)
        self.indicator_list = ['Close'] + self.static_categoricals + self.time_varying_known_reals + self.time_varying_unknown_reals

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on data path and processing parameters."""
        # Get file modification time for cache invalidation
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        cache_params = {
            "current_date": current_date,
            "days": self.days,
            "prediction_window": self.prediction_window,
            "indicator_list": sorted(self.indicator_list),  # Sort for consistency
            "max_prediction_length": self.max_prediction_length,
            "max_encoder_length": self.max_encoder_length
        }

        # Create hash from parameters
        cache_str = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        return f"processed_data_{cache_hash}"

    def _get_cache_paths(self) -> Tuple[Path, Path]:
        """Get cache file paths for data and metadata."""
        cache_key = self._generate_cache_key()
        data_path = self.cache_dir / f"{cache_key}.parquet"
        meta_path = self.cache_dir / f"{cache_key}_meta.json"
        return data_path, meta_path

    def _is_cache_valid(self) -> bool:
        """Check if cached data exists and is valid."""
        if not self.use_cache:
            return False

        data_path, meta_path = self._get_cache_paths()

        # Check if both files exist
        if not (data_path.exists() and meta_path.exists()):
            return False

        try:
            # Load and validate metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # Check if cache key matches
            expected_key = self._generate_cache_key()
            return metadata.get("cache_key") == expected_key

        except (json.JSONDecodeError, KeyError):
            return False

    def _save_to_cache(self, data: pd.DataFrame):
        """Save processed data to cache."""
        if not self.use_cache:
            return

        data_path, meta_path = self._get_cache_paths()

        try:
            # Save data as Parquet
            data.to_parquet(data_path, index=False)

            # Save metadata
            metadata = {
                "cache_key": self._generate_cache_key(),
                "data_shape": data.shape,
                "columns": list(data.columns),
                "created_at": pd.Timestamp.now().isoformat(),
                "processing_params": {
                    "days": self.days,
                    "prediction_window": self.prediction_window,
                    "max_prediction_length": self.max_prediction_length,
                    "max_encoder_length": self.max_encoder_length
                }
            }

            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Data cached to: {data_path}")

        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load processed data from cache."""
        if not self.use_cache or not self._is_cache_valid():
            return None

        data_path, meta_path = self._get_cache_paths()

        try:
            data = pd.read_parquet(data_path)
            print(f"Loaded cached data from: {data_path}")
            print(f"Cached data shape: {data.shape}")
            return data

        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None

    def prepare_data(self, raw_data=None) -> pd.DataFrame:
        """Prepare the data for training/inference"""
        # Try to load from cache first
        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

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

        # CRITICAL: Filter out tickers with insufficient data
        min_length = self.max_encoder_length + self.max_prediction_length
        print(f"Filtering tickers with at least {min_length} days of data...")

        ticker_counts = processed_data.groupby('Ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_length].index

        print(f"Before filtering: {len(ticker_counts)} tickers")
        print(f"After filtering: {len(valid_tickers)} tickers (>= {min_length} days)")
        print(f"Removed {len(ticker_counts) - len(valid_tickers)} tickers with insufficient data")

        if len(valid_tickers) == 0:
            raise ValueError(
                f"No tickers have at least {min_length} days of data. Consider reducing encoder_length or max_prediction_length.")

        # Filter the data to only include valid tickers
        processed_data = processed_data[processed_data['Ticker'].isin(valid_tickers)]

        # Create time index for the dataset
        processed_data["time_idx"] = processed_data.groupby("Ticker").cumcount()
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        processed_data = processed_data.sort_values(["Ticker", "date"]).reset_index(drop=False)

        # Save to cache
        self._save_to_cache(processed_data)
        return processed_data


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