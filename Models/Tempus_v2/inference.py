import os
import sys
import json
import pandas as pd
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add current model directory to path for local imports
model_dir = Path(__file__).parent
sys.path.insert(0, str(model_dir))

try:
    from datamodule import TFTDataModule
except ImportError:
    try:
        # Try importing from current directory
        import importlib.util
        spec = importlib.util.spec_from_file_location("datamodule", model_dir / "datamodule.py")
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)
        TFTDataModule = datamodule_module.TFTDataModule
    except Exception as e:
        print(f"Warning: Could not import TFTDataModule: {e}")
        TFTDataModule = None


class TempusV2Inference:
    """Inference class for Tempus v2 model"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)

        self.model_dir = model_dir
        self.model_path = model_dir / "Tempus_v2.onnx"
        self.meta_path = model_dir / "Tempus_v2_meta.json"
        self.cache_dir = model_dir / "inference_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Load metadata
        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Initialize ONNX session
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name

    def get_cache_path(self, data_hash: str) -> Path:
        """Get cache file path for given data hash"""
        return self.cache_dir / f"inference_{data_hash}.parquet"

    def generate_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data to use as cache key"""
        import hashlib
        data_str = str(data.index.min()) + str(data.index.max()) + str(len(data))
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def load_cached_inference(self, data_hash: str) -> Optional[pd.DataFrame]:
        """Load cached inference results if available"""
        cache_path = self.get_cache_path(data_hash)
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Warning: Could not load cache {cache_path}: {e}")
        return None

    def save_inference_cache(self, data_hash: str, results: pd.DataFrame):
        """Save inference results to cache"""
        cache_path = self.get_cache_path(data_hash)
        try:
            results.to_parquet(cache_path)
            print(f"Cached inference results to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save cache {cache_path}: {e}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features according to model metadata"""
        required_features = self.metadata['features']

        # Check if all required features are present
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select and order features according to metadata
        feature_data = data[required_features].copy()
        return feature_data

    def predict(self, data: pd.DataFrame, window_size: int = 20, use_cache: bool = True) -> pd.DataFrame:
        """
        Run inference on the provided data

        Args:
            data: DataFrame with features and Ticker column
            window_size: Size of the sliding window for predictions
            use_cache: Whether to use cached results if available

        Returns:
            DataFrame with predictions, indexed by date
        """
        # Generate cache key
        data_hash = self.generate_data_hash(data)

        # Try to load from cache
        if use_cache:
            cached_results = self.load_cached_inference(data_hash)
            if cached_results is not None:
                print(f"Loaded cached inference results for {self.metadata['name']}")
                return cached_results

        # Prepare features
        feature_data = self.prepare_features(data)

        predictions = []
        tickers = []
        dates = []

        print(f"Running inference for {self.metadata['name']} on {len(data)} samples...")

        for i in range(window_size, len(feature_data)):
            date = feature_data.index[i]
            ticker = data['Ticker'].iloc[i] if 'Ticker' in data.columns else 'UNKNOWN'

            # Get feature window (excluding Ticker column)
            feature_window = feature_data.iloc[i - window_size:i].values.astype(np.float32)

            # Add batch dimension: shape = (1, window_size, num_features)
            input_window = np.expand_dims(feature_window, axis=0)

            # Run inference
            output = self.session.run(None, {self.input_name: input_window})
            prediction = float(output[0].squeeze())  # Extract scalar prediction

            predictions.append(prediction)
            tickers.append(ticker)
            dates.append(date)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Ticker': tickers,
            'Predicted': predictions
        }, index=dates)

        # Cache results
        if use_cache:
            self.save_inference_cache(data_hash, results_df)

        return results_df

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.metadata['name'],
            'features': self.metadata['features'],
            'model_path': str(self.model_path),
            'num_features': len(self.metadata['features'])
        }


def main():
    """Example usage"""
    inference = TempusV2Inference()

    if TFTDataModule is None:
        print("TFTDataModule not available - cannot run example")
        return

    # Load data using datamodule
    try:
        datamodule = TFTDataModule(years=1, use_cache=True)
        datamodule.prepare_data()
        data = datamodule.get_inference_data()

        if data is not None:
            # Run inference
            results = inference.predict(data, window_size=20)
            print(f"Generated {len(results)} predictions")
            print(results.head())
        else:
            print("No data available for inference")
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()
