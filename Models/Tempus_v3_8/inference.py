import onnxruntime as ort
from pytorch_forecasting import TimeSeriesDataSet
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

def load_model_config(config_path: str = None) -> dict:
    """
    Load model configuration from JSON metadata file.

    Args:
        config_path: Path to the JSON config file. If None, looks for 
                    Tempus_v3.8_meta.json in the current directory.

    Returns:
        Dictionary containing model configuration
    """
    if config_path is None:
        # Look for the metadata file in the current script's directory
        script_dir = Path(__file__).parent
        config_path = script_dir / "Tempus_v3.8_meta.json"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded model configuration from: {config_path}")
    return config

class TFTDataModule:
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
        years: int = 5,
        prediction_window: int = 3,
        num_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: str = "data_cache"
    ):
        # Load config if not provided
        if config is None:
            config = load_model_config()

        self.config = config

        # Use config values with fallback to parameters
        self.batch_size = batch_size or config.get("batch_size", 256)
        self.max_prediction_length = max_prediction_length or config.get("decoder_length", 3)
        self.max_encoder_length = max_encoder_length or config.get("encoder_length", 30)
        self.years = years
        self.prediction_window = prediction_window
        self.num_workers = num_workers or max(1, os.cpu_count() // 2)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Get feature columns from config
        self.feature_cols = config.get("features", [])
        self.static_categoricals = config.get("static_categoricals", [])
        self.time_varying_known_reals = config.get("time_varying_known_reals", [])
        self.time_varying_unknown_reals = config.get("time_varying_unknown_reals", [])

        # Model path from config
        self.model_path = config.get("model_path", "Tempus_v3_8_fp8.onnx")

        # Additional columns needed for processing (target and identifier)
        self.processing_cols = self.feature_cols

        # Initialize datasets
        self.training_data = None
        self.training_dataset = None
        self.validation_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on data path and processing parameters."""
        # Get file modification time for cache invalidation
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        cache_params = {
            "current_date": current_date,
            "years": self.years,
            "prediction_window": self.prediction_window,
            "feature_cols": sorted(self.feature_cols),  # Sort for consistency
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
                    "years": self.years,
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

    def prepare_data(self) -> pd.DataFrame:
        """Load and preprocess raw data with caching support."""
        print("Loading and preprocessing data...")

        # Try to load from cache first
        cached_data = self._load_from_cache()
        if cached_data is not None:
            self.training_data = cached_data
            return cached_data

        print("Cache miss or disabled. Processing data from scratch...")

        # Load raw data with all required columns (features + Close + Ticker)
        training_data = TickerData(
            indicator_list=self.processing_cols,
            years=self.years,
            prediction_window=self.prediction_window,
            prediction_mode=True,
        ).process_all()

        # Handle MultiIndex properly
        if isinstance(training_data.index, pd.MultiIndex):
            # Reset MultiIndex and handle the level names
            training_data = training_data.reset_index()

        # Ensure Ticker column is properly formatted as strings
        if 'Ticker' in training_data.columns:
            training_data['Ticker'] = training_data['Ticker'].astype(str)

        tickers = training_data['Ticker'].unique()
        tickers = np.random.choice(tickers, 100)
        training_data = training_data[training_data['Ticker'].isin(tickers)]

        # CRITICAL: Filter out tickers with insufficient data
        min_length = self.max_encoder_length + self.max_prediction_length
        print(f"Filtering tickers with at least {min_length} days of data...")

        ticker_counts = training_data.groupby('Ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= min_length].index

        print(f"Before filtering: {len(ticker_counts)} tickers")
        print(f"After filtering: {len(valid_tickers)} tickers (>= {min_length} days)")
        print(f"Removed {len(ticker_counts) - len(valid_tickers)} tickers with insufficient data")

        if len(valid_tickers) == 0:
            raise ValueError(
                f"No tickers have at least {min_length} days of data. Consider reducing encoder_length or max_prediction_length.")

        # Filter the data to only include valid tickers
        training_data = training_data[training_data['Ticker'].isin(valid_tickers)]

        # Create time index for the dataset
        training_data["time_idx"] = training_data.groupby("Ticker").cumcount()
        training_data = training_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        training_data = training_data.sort_values(["Ticker", "date"]).reset_index(drop=True)

        # Save to cache
        self._save_to_cache(training_data)

        self.training_data = training_data
        print(f"Data prepared successfully. Shape: {training_data.shape}")
        return training_data

# ---------------------------------------------------------------------
# 1.  CONFIG-BASED CONSTANTS  
# ---------------------------------------------------------------------
def get_model_constants(config: dict = None):
    """
    Get model constants from configuration.

    Args:
        config: Model configuration dictionary. If None, loads from JSON.

    Returns:
        Dictionary containing model constants
    """
    if config is None:
        config = load_model_config()

    return {
        'ENCODER_LEN': config.get("encoder_length", 30),
        'DECODER_LEN': config.get("decoder_length", 3),
        'GROUP_IDS': ["Ticker"],  # column(s) uniquely identifying a series
        'TARGET': "Close",  # regression target used in training
        'TIME_IDX': "time_idx",  # any sequential int index
        'STATIC_CATS': config.get("static_categoricals", []),
        'STATIC_REALS': [],  # static real covariates
        'TV_UNKNOWN_REAL': config.get("time_varying_unknown_reals", []),
        'TV_KNOWN_REAL': config.get("time_varying_known_reals", []),
        'TV_UNKNOWN_CAT': [],  # e.g. lagged categorical
        'ALLOW_MISSING': True,
        'ONNX_MODEL_PATH': config.get("model_path", "Tempus_v3_8_fp8.onnx"),
        'EXEC_PROVIDER': "CPUExecutionProvider",  # or "CUDAExecutionProvider"
        'BATCH_SIZE': config.get("batch_size", 256),
        'NUM_WORKERS': 0  # >0 if Linux
    }

# ---------------------------------------------------------------------
# 2.  PREP OOS DATA  ---------------------------------------------------
# ---------------------------------------------------------------------
def prepare_dataset(oos_df: pd.DataFrame, config: dict = None) -> TimeSeriesDataSet:
    """
    Create a TimeSeriesDataSet with *exactly* the same schema as training.

    Args:
        oos_df: Out-of-sample DataFrame
        config: Model configuration dictionary. If None, loads from JSON.

    Returns:
        TimeSeriesDataSet configured for inference
    """
    constants = get_model_constants(config)

    dataset = TimeSeriesDataSet(
        oos_df,
        time_idx=constants['TIME_IDX'],
        target=constants['TARGET'],
        group_ids=constants['GROUP_IDS'],
        min_encoder_length=constants['ENCODER_LEN'] // 2,
        max_encoder_length=constants['ENCODER_LEN'],
        min_prediction_length=constants['DECODER_LEN'],
        max_prediction_length=constants['DECODER_LEN'],
        time_varying_known_reals=constants['TV_KNOWN_REAL'],
        time_varying_unknown_reals=constants['TV_UNKNOWN_REAL'],
        static_categoricals=constants['STATIC_CATS'],
        allow_missing_timesteps=constants['ALLOW_MISSING'],
        add_relative_time_idx=False,   # keep plain ints if you used them
        target_normalizer=None         # already baked into ONNX graph
    )
    return dataset


# ---------------------------------------------------------------------
# 3.  RUN INFERENCE  ---------------------------------------------------
# ---------------------------------------------------------------------
def onnx_predict(model_path: str,
                 dataset: TimeSeriesDataSet,
                 batch_size: int = 256,
                 provider: str = "CPUExecutionProvider",
                 config: dict = None) -> pd.DataFrame:
    """
    Iterate through a TimeSeriesDataSet, send batches to ONNX Runtime
    and stitch forecasts back into a tidy DataFrame.

    Args:
        model_path: Path to the ONNX model file
        dataset: TimeSeriesDataSet for inference
        batch_size: Batch size for inference
        provider: ONNX execution provider
        config: Model configuration dictionary

    Returns
    -------
    DataFrame with columns
        ['Ticker', 'time_idx', 't+1', 't+2', ... 't+DECODER_LEN']
    """
    constants = get_model_constants(config)
    # ––––––––––––––– 3.1 create session –––––––––––––––
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = True
    session = ort.InferenceSession(model_path,
                                   sess_options=sess_options,
                                   providers=[provider])

    # names come out in arbitrary order → safe to query
    input_names = {i.name for i in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    # ––––––––––––––– 3.2 dataloader –––––––––––––––
    loader = dataset.to_dataloader(
        train=False,  # disables data augmentations
        batch_size=batch_size,
        shuffle=False,
        num_workers=constants['NUM_WORKERS'],
        drop_last=False  # keep trailing smaller batches
    )

    # mapping from PTF tensor names -> ONNX input names
    alias = {
        "encoder_cont":     "enc_cont",
        "encoder_cat":      "enc_cat",
        "decoder_cont":     "dec_cont",
        "decoder_cat":      "dec_cat",
        "encoder_lengths":  "enc_len",
        "decoder_lengths":  "dec_len",
        "target_scale":     "target_scale",
    }

    preds, groups, times = [], [], []

    for batch in loader:
        x, _ = batch

        # Fix sequence lengths validation - get actual sequence dimensions from tensors
        enc_seq_len = x["encoder_cont"].shape[1]
        dec_seq_len = x["decoder_cont"].shape[1]

        # Validate and fix encoder lengths
        if "encoder_lengths" in x:
            x["encoder_lengths"].fill_(enc_seq_len)

        # Validate and fix decoder lengths
        if "decoder_lengths" in x:
            x["decoder_lengths"].fill_(dec_seq_len)

        ort_inputs = {
            alias.get(k, k): v.detach().cpu().numpy()
            for k, v in x.items()
            if alias.get(k, k) in input_names
        }

        # Additional validation for sequence lengths in ONNX inputs
        #if "enc_len" in ort_inputs:
        #    ort_inputs["enc_len"] = np.clip(ort_inputs["enc_len"], 1, enc_seq_len - 1)
        #if "dec_len" in ort_inputs:
        #    ort_inputs["dec_len"] = np.clip(ort_inputs["dec_len"], 1, dec_seq_len - 1)

        try:
            # forward pass
            batch_pred = session.run([output_name], ort_inputs)[0]  # (B, dec_len, n_targets)

            # Fix the reshape issue - handle 3D output properly
            if batch_pred.ndim == 3:
                # If output is (batch_size, seq_len, n_targets), we need to reshape appropriately
                batch_size_actual, seq_len, n_targets = batch_pred.shape
                # Take the last timestep prediction or reshape based on your model's output format
                if n_targets == 1:
                    # Single target case - take all timesteps
                    batch_pred = batch_pred.squeeze(-1)  # (batch_size, seq_len)
                else:
                    # Multiple targets - you may need to select specific targets or reshape
                    # For now, let's take the first target across all timesteps
                    batch_pred = batch_pred[:, :, 0]  # (batch_size, seq_len)
            elif batch_pred.ndim == 2:
                # Already in correct format (batch_size, seq_len)
                pass
            else:
                raise ValueError(f"Unexpected output shape: {batch_pred.shape}")

            # keep identifiers to re-assemble
            preds.append(batch_pred)  # (B, dec_len)
            groups.append(x["groups"].squeeze().cpu().numpy())  # (B,)
            times.append(x["decoder_time_idx"].cpu().numpy())  # (B, dec_len)

        except Exception as e:
            print(f"Error in batch inference: {e}")
            continue

        # Check if we have any successful predictions
    if not preds:
        raise RuntimeError("No successful predictions were made. Check your data and model compatibility.")

    # ––––––––––––––– 3.3 concat & tidy –––––––––––––––
    preds_arr = np.concatenate(preds, axis=0)  # (N, dec_len)
    groups_arr = np.concatenate(groups, axis=0)  # (N,)
    times_arr = np.concatenate(times, axis=0)  # (N, dec_len)

    # Ensure we have 2D data for DataFrame creation
    if preds_arr.ndim != 2:
        raise ValueError(f"Predictions array should be 2D, got shape: {preds_arr.shape}")

    n_horizons = preds_arr.shape[1]

    # Create DataFrame with proper 2D structure
    df_data = {f"t+{i + 1}": preds_arr[:, i] for i in range(n_horizons)}
    df_data["group_id"] = groups_arr

    # Handle time indices properly - take the first time index for each sample
    if times_arr.ndim == 2:
        df_data["time_idx"] = times_arr[:, 0]  # Take first timestep
    else:
        df_data["time_idx"] = times_arr

    df = pd.DataFrame(df_data)

    # Melt to long format
    value_cols = [f"t+{i + 1}" for i in range(n_horizons)]
    df = df.melt(
        id_vars=["group_id", "time_idx"],
        value_vars=value_cols,
        var_name="horizon",
        value_name="prediction"
    ).sort_values(["group_id", "time_idx", "horizon"], ignore_index=True)

    return df


# ---------------------------------------------------------------------
# 4.  MAIN INFERENCE FUNCTION  ----------------------------------------
# ---------------------------------------------------------------------
def run_inference(config_path: str = None, external_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main inference function that loads config, prepares data, and runs prediction.

    Args:
        config_path: Path to the JSON config file. If None, uses default.
        external_df: External DataFrame to use for inference. If None, uses TFTDataModule to fetch data.

    Returns:
        DataFrame with predictions
    """
    # Load configuration
    config = load_model_config(config_path)
    constants = get_model_constants(config)

    # Initialize data module
    data_module = TFTDataModule(config=config)

    # Get data - either from external source or fetch using data module
    if external_df is not None:
        print("Using provided external DataFrame for inference")
        oos_df = external_df
    else:
        print("Fetching data using TFTDataModule...")
        oos_df = data_module.prepare_data()

    # Ensure we have a consecutive int index per series
    if constants['TIME_IDX'] not in oos_df.columns:
        print(f"Creating {constants['TIME_IDX']} column...")
        oos_df[constants['TIME_IDX']] = (
            oos_df.groupby(constants['GROUP_IDS']).cumcount().astype("int32")
        )

    # Prepare dataset for inference
    print("Preparing dataset for inference...")
    dataset = prepare_dataset(oos_df, config)

    # Get model path (relative to script directory)
    script_dir = Path(__file__).parent
    model_path = script_dir / constants['ONNX_MODEL_PATH']

    # Run inference
    print(f"Running inference with model: {model_path}")
    predictions = onnx_predict(
        str(model_path), 
        dataset,
        batch_size=constants['BATCH_SIZE'],
        provider=constants['EXEC_PROVIDER'],
        config=config
    )

    print(f"Inference completed. Generated {len(predictions)} predictions.")
    return predictions

# ---------------------------------------------------------------------
# 5.  EXAMPLE USAGE  ---------------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Run inference with automatic config loading and data fetching
    predictions = run_inference()

    # Store results
    predictions.to_parquet("predictions.parquet")
    print("Predictions saved to predictions.parquet")
    print("\nFirst few predictions:")
    print(predictions.head())
