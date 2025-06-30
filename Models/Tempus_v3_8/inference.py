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

    tickers_sorted = sorted(oos_df["Ticker"].unique())
    id2ticker = {i: t for i, t in enumerate(tickers_sorted)}

    # We'll need `oos_df` later to get the date, so keep a trimmed copy
    date_lookup = (
        oos_df.loc[:, ["Ticker", "time_idx", "date"]]
              .drop_duplicates()             # one row per (Ticker, time_idx)
              .set_index(["Ticker", "time_idx"])
    )

    # Get model path (relative to script directory)
    model_path = constants['ONNX_MODEL_PATH']

    # Run inference
    print(f"Running inference with model: {model_path}")
    predictions = onnx_predict(
        str(model_path), 
        dataset,
        batch_size=constants['BATCH_SIZE'],
        provider=constants['EXEC_PROVIDER'],
        config=config
    )

    predictions["Ticker"] = predictions["group_id"].map(id2ticker)

    # Re-attach the calendar date that belongs to the *first* decoder
    # position (the one stored in `time_idx`)
    predictions = (
        predictions
        .merge(
            date_lookup.reset_index(),
            on=["Ticker", "time_idx"],
            how="left"
        )
        .drop(columns="group_id")            # no longer useful
        .loc[:, ["Ticker", "date", "time_idx",
                 "horizon", "prediction"]]   # tidy column order
        .sort_values(["Ticker", "date", "horizon"])
        .reset_index(drop=True)
    )

    print(f"Inference completed. Generated {len(predictions)} predictions.")
    return predictions

# ---------------------------------------------------------------------
# 5.  EXAMPLE USAGE  ---------------------------------------------------
# ---------------------------------------------------------------------
predictions = run_inference()

# Store results
predictions.to_parquet("predictions.parquet")
print("Predictions saved to predictions.parquet")
print("\nFirst few predictions:")
print(predictions.head())