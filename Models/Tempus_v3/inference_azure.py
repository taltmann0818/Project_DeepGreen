# New script

# Scoring script for Azure ML Batch Endpoint – Tempus v3
# --------------------------------------------------------

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from azure.storage.blob import BlobServiceClient
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
from scipy.stats import norm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Azure helpers
# -----------------------------------------------------------------------------

def get_azure_blob_client(connection_string: str) -> BlobServiceClient:
    """Return an authenticated Azure BlobServiceClient."""
    return BlobServiceClient.from_connection_string(connection_string)

def upload_dataframe_to_blob(
        blob_service_client: BlobServiceClient,
        container_name: str,
        blob_name: str,
        df: pd.DataFrame,
):
    """Upload *df* as a parquet file to `container_name/blob_name`."""
    import io

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(buffer.getvalue(), overwrite=True)
    print(f"✓ Uploaded {blob_name} ({df.shape}) to container {container_name}")


# -----------------------------------------------------------------------------
# Inference wrapper
# -----------------------------------------------------------------------------

class Tempusv3Inference:
    """Inference class for Tempus v3 model"""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) / "onnx_bundle"
        cfg_path = self.model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.yaml in {self.model_dir}")
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh) or {}
        self.const = self._translate_config()
        self.session: Optional[ort.InferenceSession] = None
        self.data: Optional[pd.DataFrame] = None
        self.id2ticker: Optional[dict[int, str]] = None
        self.date_lookup: Optional[pd.DataFrame] = None

    def _build_session(self) -> ort.InferenceSession:
        if self.session is None:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            model_path = self.model_dir / self.const["ONNX_MODEL_PATH"]
            print(f"[onnx] loading {model_path.relative_to(self.model_dir.parent)} …")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        return self.session

    def _translate_config(self) -> Dict[str, Any]:
        c = self.config
        return {
            "ENCODER_LEN": c["dataset_parameters"]["value"]["max_encoder_length"],
            "DECODER_LEN": c["dataset_parameters"]["value"]["min_prediction_length"],
            "GROUP_IDS": ["Ticker"],
            "TARGET": "Close",
            "TIME_IDX": "time_idx",
            "STATIC_CATS": c["dataset_parameters"]["value"]["static_categoricals"],
            "TV_UNKNOWN_REAL": c["dataset_parameters"]["value"]["time_varying_unknown_reals"],
            "TV_KNOWN_REAL": c["dataset_parameters"]["value"]["time_varying_known_reals"],
            "ALLOW_MISSING": bool(c.get("allow_missing", True)),
            "ONNX_MODEL_PATH": str(c["onnx_model_name"]["value"]),
            "BATCH_SIZE": int(c["batch_size"]["value"]),
        }

    def prepare_features(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        """
        Build a TimeSeriesDataSet that mirrors the training schema.
        """
        self.data = df.copy()
        const = self.const
        if const["TIME_IDX"] not in self.data.columns:
            self.data[const["TIME_IDX"]] = self.data.groupby(const["GROUP_IDS"]).cumcount().astype("int32")

        return TimeSeriesDataSet(
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
        const = self.const
        tft_dataset = self.prepare_features(data)
        session = self._build_session()
        loader = tft_dataset.to_dataloader(train=False, batch_size=const['BATCH_SIZE'], shuffle=False, num_workers=0,
                                           drop_last=False)

        if len(loader) == 0:
            raise ValueError("Dataloader is empty – check input schema vs. training schema.")
        print(f"[predict] dataset len={len(tft_dataset):,}, batches={len(loader):,}")

        # map PTF names → ONNX names
        alias = {
            "encoder_cont": "enc_cont",
            "encoder_cat": "enc_cat",
            "decoder_cont": "dec_cont",
            "decoder_cat": "dec_cat",
            "encoder_lengths": "enc_len",
            "decoder_lengths": "dec_len",
            "target_scale": "target_scale",
        }
        input_names = {i.name for i in session.get_inputs()}
        output_name = session.get_outputs()[0].name

        preds, groups, times = [], [], []
        for batch_idx, batch in enumerate(loader):
            try:
                x, _ = batch
                ort_inputs = {alias.get(k, k): v.detach().cpu().numpy() for k, v in x.items() if
                              alias.get(k, k) in input_names}
                out = session.run([output_name], ort_inputs)[0]

                # out can be (B, T, Q)  or  (B, Q, T)
                if out.ndim != 3:
                    raise ValueError(f"ONNX output must be 3-D, got {out.shape}")

                b, d1, d2 = out.shape
                dec_len = tft_dataset.max_prediction_length
                if d2 == dec_len:  # (B, Q, T)
                    out = out.transpose(0, 2, 1)  # (B, T, Q) → transpose
                elif d1 != dec_len:  # neither dim matches decoder lengt
                    raise ValueError(f"Cannot locate decoder length {dec_len} in ONNX output {out.shape}")

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
        flat_len = N * T * Q
        group_id = np.repeat(groups_arr, T * Q)
        time_idx = np.repeat(times_arr.reshape(-1), Q)
        horizon = np.tile(np.repeat(np.arange(1, T + 1), Q), N)
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

        print(f"[predict] Creating predictions DataFrame...")
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

        print(f"[predict] Building final result DataFrame...")
        result = pd.DataFrame(index=pivot.index)
        result["Predicted"] = pivot[np.median(quantiles)]  # median
        result["q_low"] = pivot[np.min(quantiles)]  # 2 % quantile
        result["q_high"] = pivot[np.max(quantiles)]  # 98 % quantile
        result["sigma_daily"] = (result["q_high"] - result["q_low"]) / (2 * norm.ppf(0.98)) / np.sqrt(3)
        result = result.reset_index().merge(predictions[['Ticker', 'date', 'Close']], on=["Ticker", "date"],
                                            how="left").drop_duplicates()
        result['pred_return'] = (result['Predicted'] - result['Close']) / result[
            'Close']  # Updating to use implied returns instead of median quantile raw prediction
        result['q_low'] = (result['q_low'] - result['Close']) / result[
            'Close']  # Updating to use implied returns instead of low quantile raw prediction
        result['q_high'] = (result['q_high'] - result['Close']) / result[
            'Close']  # Updating to use implied returns instead of high quantile raw prediction

        print(f"[predict] Final result shape: {result.shape}")

        return result


# -----------------------------------------------------------------------------
# Azure ML Batch Endpoint glue code
# -----------------------------------------------------------------------------

def init():
    """Initialise model"""
    global model, out_dir

    model = Tempusv3Inference(model_dir=os.getenv("AZUREML_MODEL_DIR"))
    out_dir = Path(os.getenv("AZUREML_BI_OUTPUT_PATH", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[init] ✓ model ready – outputs →", out_dir)


def run(mini_batch: List[str]):
    outputs: List[str] = []

    for file_path in mini_batch:
        in_path = Path(file_path)
        try:
            # 1) Read the parquet
            df_in = pd.read_parquet(file_path)

            # 2) Inference
            preds = model.predict(df_in)

            # 3) Write predictions
            out_path = out_dir / f"{in_path.stem}_predictions.parquet"
            preds.to_parquet(out_path, index=False)
            outputs.append(str(out_path))
            print(f"[run] {out_path.name} ({preds.shape})")

        except Exception as exc:
            # 4) Capture per‑file error so the mini‑batch still *succeeds*
            err_path = out_dir / f"{in_path.stem}.error.txt"
            err_path.write_text(str(exc))
            outputs.append(str(err_path))
            print(f"[run] \u274c {in_path.name}: {exc}")

    return outputs