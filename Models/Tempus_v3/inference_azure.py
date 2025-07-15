"""
Tempus v3 Azure ML Batch Endpoint Inference Script

This script is designed to run inference with the Tempus v3 TFT model on Azure ML batch endpoints.
It implements the standard Azure ML batch inference pattern with init() and run(mini_batch) functions.

Azure ML Batch Endpoint Usage:
The script automatically uses init() and run(mini_batch) functions when deployed as an Azure ML batch endpoint.
- init(): Called once per worker to load the model and initialize Azure Storage client
- run(mini_batch): Processes chunks of parquet files and returns output paths

Environment Variables (for Azure ML batch endpoint):
- AZURE_STORAGE_CONNECTION_STRING: Azure Storage connection string
- AZURE_STORAGE_CONTAINER_NAME: Container name (defaults to 'batch-inference')

Legacy Usage:
1. Azure Batch Mode (legacy):
   python inference_azure.py --input_data input.parquet --output_data predictions.parquet 
                            --connection_string "your_connection_string" --container_name "your_container"

2. Local Testing Mode:
   python inference_azure.py

3. Test Azure Functions:
   python inference_azure.py --test

4. Test Batch Endpoint Functions:
   python inference_azure.py --test-batch

Requirements:
- The ONNX model file (tft_20250628_122613_6d872b_fp8.onnx) must be in the same directory
- config.yaml must be in the same directory
- Input data must be a parquet file created using the DataModule from datamodule.py
- Azure Storage connection string and container must be accessible (for Azure deployment)

The script expects input data in the format produced by DataModule.prepare_data() with columns:
- Ticker: Stock ticker symbol
- date: Date column
- time_idx: Time index for the time series
- Close: Target variable (closing price)
- Various technical indicators and features as defined in config.yaml
"""

import onnxruntime as ort
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
import yaml
import os
import warnings
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
from azure.storage.blob import BlobServiceClient
warnings.filterwarnings("ignore")

def get_azure_blob_client(connection_string: str, container_name: str) -> BlobServiceClient:
    """Initialize Azure Blob Service Client"""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return blob_service_client

def download_blob_to_dataframe(blob_service_client: BlobServiceClient, container_name: str, blob_name: str) -> pd.DataFrame:
    """Download parquet file from Azure Storage and return as DataFrame"""
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download blob content
        blob_data = blob_client.download_blob().readall()

        # Read parquet data from bytes
        import io
        df = pd.read_parquet(io.BytesIO(blob_data))
        print(f"Successfully downloaded {blob_name} from container {container_name}")
        print(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Failed to download blob {blob_name}: {str(e)}")

def upload_dataframe_to_blob(blob_service_client: BlobServiceClient, container_name: str, blob_name: str, df: pd.DataFrame):
    """Upload DataFrame as parquet file to Azure Storage"""
    try:
        # Convert DataFrame to parquet bytes
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        # Upload to blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        print(f"Successfully uploaded {blob_name} to container {container_name}")
        print(f"Uploaded data shape: {df.shape}")
    except Exception as e:
        raise Exception(f"Failed to upload blob {blob_name}: {str(e)}")

class Tempusv3Inference:
    """Inference class for Tempus v3 model"""
    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).parent)) / "onnx_bundle"

        cfg_path = self.model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"'config.yaml' not found in {self.model_dir}. "
                "Verify the model asset bundles it."
            )

        with open(cfg_path, "r", encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh) or {}

        self.session = None  # cache

        # convenience: keep constants handy
        self.constants = self.get_model_constants()

        # these will be filled later
        self.data: pd.DataFrame | None = None
        self.id2ticker: dict[int, str] | None = None
        self.date_lookup: pd.DataFrame | None = None

    def _build_session(self):
        if self.session is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                self.model_dir / self.constants["ONNX_MODEL_PATH"],
                providers=providers,
                sess_options=opts,
            )
        return self.session

    def get_model_constants(self) -> Dict[str, Any]:
        """Translate YAML keys into the constants the pipeline expects."""
        c = self.config  # shorthand
        return {
            "ENCODER_LEN":         c["dataset_parameters"]["value"]["max_encoder_length"],
            "DECODER_LEN":         c["dataset_parameters"]["value"]["min_prediction_length"],
            "GROUP_IDS":           ["Ticker"],
            "TARGET":              "Close",
            "TIME_IDX":            "time_idx",
            "STATIC_CATS":         c["dataset_parameters"]["value"]["static_categoricals"],
            #"STATIC_REALS":        c.get("static_reals", []),
            "TV_UNKNOWN_REAL":     c["dataset_parameters"]["value"]["time_varying_unknown_reals"],
            "TV_KNOWN_REAL":       c["dataset_parameters"]["value"]["time_varying_known_reals"],
            #"TV_UNKNOWN_CAT":      c.get("time_varying_unknown_categoricals", []),
            "ALLOW_MISSING":       bool(c.get("allow_missing", True)),
            "ONNX_MODEL_PATH":     str(c["onnx_model_name"]["value"]),
            "EXEC_PROVIDER":       "CPUExecutionProvider",
            "BATCH_SIZE":          int(c["batch_size"]["value"]),
            "NUM_WORKERS":         int(c.get("num_workers", 0)),
            "SAMPLE_SIZE":         100,
            # Anything else you put in YAML is still available via self.config
        }

    def prepare_features(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        """
        Build a TimeSeriesDataSet that mirrors the training schema.
        """
        self.data = df.copy()
        const = self.constants

        # Ensure we have a consecutive int index per series
        if const["TIME_IDX"] not in self.data.columns:
            self.data[const["TIME_IDX"]] = (
                self.data.groupby(const["GROUP_IDS"]).cumcount().astype("int32")
            )

        target_normalizer = EncoderNormalizer(
            method="standard",
            center=True,
            transformation="log",
        )

        dataset = TimeSeriesDataSet(
            self.data,
            time_idx=const['TIME_IDX'],
            target=const['TARGET'],
            group_ids=const['GROUP_IDS'],
            min_encoder_length=const['ENCODER_LEN'] // 2,
            max_encoder_length=const['ENCODER_LEN'],
            min_prediction_length=const['DECODER_LEN'],
            max_prediction_length=const['DECODER_LEN'],
            time_varying_known_reals=const['TV_KNOWN_REAL'],
            time_varying_unknown_reals=const['TV_UNKNOWN_REAL'],
            static_categoricals=const['STATIC_CATS'],
            allow_missing_timesteps=const['ALLOW_MISSING'],
            add_relative_time_idx=False,
            add_encoder_length=True,
            target_normalizer=None,
            predict_mode=False
        )
        return dataset

    def predict(
        self,
        data: pd.DataFrame,
        quantiles: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    ) -> pd.DataFrame:
        """
        Run ONNX-exported TFT on a `TimeSeriesDataSet` and return **quantile**
        forecasts in tidy (long) format.

        Returns columns:
            Ticker, date, horizon, quantile, prediction
        """
        const = self.constants
        tft_dataset = self.prepare_features(data)
        # ––––––––––––––– 1 Create ONNX session –––––––––––––––
        session = self._build_session()
        print(f"Loaded ONNX Model: {const['ONNX_MODEL_PATH']}")

        input_names = {i.name for i in session.get_inputs()}
        output_name = session.get_outputs()[0].name

        # ––––––––––––––– 2 Dataloader –––––––––––––––
        loader = tft_dataset.to_dataloader(
            train=False,
            batch_size=const['BATCH_SIZE'],
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        # map PTF names → ONNX names
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
        for batch_idx, batch in enumerate(
                tqdm(loader,
                     total=len(loader),  # lets tqdm know how many batches to expect
                     desc="Running ONNX inference",
                     leave=True)
        ):
            try:
                x, _ = batch
                ort_inputs = {alias.get(k, k): v.detach().cpu().numpy()
                              for k, v in x.items()
                              if alias.get(k, k) in input_names}
                out = session.run([output_name], ort_inputs)[0]

                # out can be (B, T, Q)  or  (B, Q, T)
                if out.ndim != 3:
                    raise ValueError(f"ONNX output must be 3-D, got {out.shape}")

                b, d1, d2 = out.shape
                dec_len = tft_dataset.max_prediction_length
                if d2 == dec_len:                 # (B, Q, T)
                    out = out.transpose(0, 2, 1)  # (B, T, Q) → transpose
                elif d1 != dec_len:               # neither dim matches decoder lengt
                    raise ValueError(
                        f"Cannot locate decoder length {dec_len} in ONNX output {out.shape}"
                    )

                preds.append(out)
                groups.append(x["groups"].squeeze().cpu().numpy())
                times.append(x["decoder_time_idx"].cpu().numpy())

            except Exception as e:
                print(f"Warning: Skipping batch {batch_idx} due to error: {e}")
                continue

        if not preds:
            raise RuntimeError("No successful predictions were made. Check your data and model compatibility.")

        # ––––––––––––––– 3 concat & tidy –––––––––––––––
        preds_arr = np.concatenate(preds, axis=0)
        groups_arr = np.concatenate(groups, axis=0) 
        times_arr = np.concatenate(times, axis=0) 

        N, T, Q = preds_arr.shape
        if Q != len(quantiles):
            raise ValueError(
                f"Model output {Q} quantiles vs. requested {len(quantiles)}: {quantiles}"
            )

        # one row per (group, time_idx[h], horizon=h+1, q)
        flat_len   = N * T * Q
        group_id   = np.repeat(groups_arr, T * Q)
        time_idx   = np.repeat(times_arr.reshape(-1), Q)
        horizon    = np.tile(np.repeat(np.arange(1, T + 1), Q), N)
        quantile_r = np.tile(np.array(quantiles), N * T)
        prediction = preds_arr.reshape(-1)

        # build id → ticker mapping the first time we predict
        if self.id2ticker is None:
            tickers_sorted = sorted(self.data["Ticker"].unique())
            self.id2ticker = {i: t for i, t in enumerate(tickers_sorted)}

        if self.date_lookup is None:
            self.date_lookup = (
                self.data.loc[:, ["Ticker", const["TIME_IDX"], "date", "Close"]]
                .drop_duplicates()
                .set_index(["Ticker", const["TIME_IDX"]])
            )

        predictions = (
            pd.DataFrame(
                {
                    "group_id": group_id,
                    "time_idx": time_idx,
                    "horizon": horizon,
                    "quantile": quantile_r,
                    "prediction": prediction,
                }
            )
            .assign(Ticker=lambda d: d["group_id"].map(self.id2ticker))
            .merge(
                self.date_lookup.reset_index(),
                on=["Ticker", "time_idx"],
                how="left",
            )
            .drop(columns="group_id")
            .loc[:, ["Ticker", "date", "time_idx", "horizon", "quantile", "prediction", "Close"]]
            .sort_values(["Ticker", "date", "horizon"])
            .reset_index(drop=True)
        )

        h3 = predictions.query("horizon == 3")

        # ------------------------------------------------------------------
        # 3 )  Wide-format by quantile
        # ------------------------------------------------------------------
        pivot = (
            h3.pivot_table(index=["date", "Ticker"],
                           columns="quantile",
                           values="prediction")
               .rename_axis(None, axis=1)
        )

        result = pd.DataFrame(index=pivot.index)
        result["Predicted"]    = pivot[np.median(quantiles)]    # median
        result["q_low"]        = pivot[np.min(quantiles)]       # 2 % quantile
        result["q_high"]       = pivot[np.max(quantiles)]       # 98 % quantile
        result["sigma_daily"] = (result["q_high"] - result["q_low"]) / (2 * norm.ppf(0.98)) / np.sqrt(3)
        result = result.reset_index().merge(predictions[['Ticker','date','Close']], on=["Ticker", "date"], how="left").drop_duplicates()
        result['pred_return'] = (result['Predicted'] - result['Close']) / result['Close'] # Updating to use implied returns instead of median quantile raw prediction
        result['q_low'] = (result['q_low'] - result['Close']) / result['Close'] # Updating to use implied returns instead of low quantile raw prediction
        result['q_high'] = (result['q_high'] - result['Close']) / result['Close'] # Updating to use implied returns instead of high quantile raw prediction

        return result.set_index('date') #.drop(columns='Close')


# Global variables for Azure ML batch endpoint
inference_model = None
blob_service_client = None
container_name = None

def init():
    """
    Initialize the model and Azure Storage client once when the batch endpoint starts.
    This function is called once per worker process in Azure ML batch endpoints.
    """
    global inference_model, blob_service_client, container_name

    try:
        print("Initializing Tempus v3 inference model for Azure ML batch endpoint...")

        # Initialize the inference model
        model_root = os.getenv("AZUREML_MODEL_DIR")  # <-- new
        inference_model = Tempusv3Inference(model_dir=model_root)
        print("✓ Model loaded successfully")

        # Initialize Azure Storage client using environment variables
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        container_name = os.environ.get('AZURE_STORAGE_CONTAINER_NAME', 'batch-inference')

        if connection_string:
            blob_service_client = get_azure_blob_client(connection_string, container_name)
            print(f"✓ Azure Storage client initialized for container: {container_name}")
        else:
            print("Warning: AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
            blob_service_client = None

        print("Initialization completed successfully!")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise


def init():
    """
    Initialize the model and Azure Storage client once when the batch endpoint starts.
    This function is called once per worker process in Azure ML batch endpoints.
    """
    global inference_model, blob_service_client, container_name

    try:
        print("Initializing Tempus v3 inference model for Azure ML batch endpoint...")

        # Initialize the inference model
        model_root = os.getenv("AZUREML_MODEL_DIR", Path(__file__).parent)
        print(f"Model root directory: {model_root}")

        # Debug: List contents of model directory
        model_path = Path(model_root)
        if model_path.exists():
            print(f"Contents of {model_path}:")
            for item in model_path.iterdir():
                print(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")

        inference_model = Tempusv3Inference(model_dir=model_root)
        print("✓ Model loaded successfully")

        # Initialize Azure Storage client using environment variables
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        container_name = os.environ.get('AZURE_STORAGE_CONTAINER_NAME', 'batch-inference')

        if connection_string:
            blob_service_client = get_azure_blob_client(connection_string, container_name)
            print(f"✓ Azure Storage client initialized for container: {container_name}")
        else:
            print("Warning: AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
            blob_service_client = None

        print("Initialization completed successfully!")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run(mini_batch):
    """
    Process a mini-batch of input files and return output file paths.

    Args:
        mini_batch: List of input file paths (parquet files)

    Returns:
        List of output file paths where predictions are saved
    """
    global inference_model, blob_service_client, container_name

    if inference_model is None:
        raise RuntimeError("Model not initialized. Call init() first.")

    output_paths = []

    for input_file_path in mini_batch:
        try:
            infile = Path(input_file_path)

            if infile.exists():
                input_data = pd.read_parquet(infile)

            elif blob_service_client and container_name:
                input_data = download_blob_to_dataframe(
                    blob_service_client,
                    container_name,
                    infile.name,  # just the filename
                )
            else:
                raise FileNotFoundError(f"{infile} not found locally or in Blob")
        except Exception as e:
            print(f"❌ {input_file_path}: {e}")
            output_paths.append(None)  # marks this item as failed

        print(f"Loaded data with shape: {input_data.shape}")

        # Validate input data
        if input_data.empty:
            print(f"Warning: Empty input data for {input_file_path}")
            continue

        # Run inference
        predictions = inference_model.predict(input_data)
        print(f"Generated {len(predictions)} predictions")

        # Validate predictions
        if predictions.empty:
            print(f"Warning: Empty predictions for {input_file_path}")
            continue

        # Azure ML creates an output folder path for you
        out_dir = os.environ.get("AZUREML_BI_OUTPUT_PATH", ".")
        output_path = Path(out_dir) / f"predictions_{infile.name}"
        predictions.to_parquet(output_path, index=False)

        output_paths.append(str(output_path))
        print(f"✓ Saved predictions to: {output_path}")

    return output_paths


def run(mini_batch):
    """
    Process a mini-batch of input files and return output file paths.

    Args:
        mini_batch: List of input file paths (parquet files)

    Returns:
        List of output file paths where predictions are saved
    """
    global inference_model, blob_service_client, container_name

    if inference_model is None:
        raise RuntimeError("Model not initialized. Call init() first.")

    output_paths = []

    for input_file_path in mini_batch:
        try:
            print(f"Processing file: {input_file_path}")

            # Extract filename from path for output naming
            input_filename = Path(input_file_path).name
            output_filename = f"predictions_{input_filename}"

            # Read the parquet file
            input_data = download_blob_to_dataframe(blob_service_client, container_name, input_filename)
            print(f"Loaded data with shape: {input_data.shape}")

            # Run inference
            predictions = inference_model.predict(input_data)
            print(f"Generated {len(predictions)} predictions")

            # Upload to Azure Storage
            upload_dataframe_to_blob(blob_service_client, container_name, output_filename, predictions)
            output_path = f"azureml://datastores/workspaceblobstore/paths/{container_name}/{output_filename}"

            output_paths.append(output_path)
            print(f"✓ Saved predictions to: {output_path}")

        except Exception as e:
            print(f"Error processing {input_file_path}: {str(e)}")
            # Continue processing other files in the batch
            continue

    return output_paths
