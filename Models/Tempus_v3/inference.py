import onnxruntime as ort
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import hashlib
import sys
from scipy.stats import norm
from pathlib import Path
import yaml
import os
import warnings
from typing import Dict, Any, Optional
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# Add current model directory to path for local imports
model_dir = Path(__file__).parent
sys.path.insert(0, str(model_dir))

def _find_model(pickle_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]   # ../..
    for path in project_root.rglob(pickle_name):
        return path
    raise FileNotFoundError(f"{pickle_name} not found anywhere under {project_root}")

try:
    from datamodule import DataModule
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("datamodule", model_dir / "datamodule.py")
    datamodule_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datamodule_module)
    DataModule = datamodule_module.DataModule

class Tempusv3Inference:
    """Inference class for Tempus v3 model"""
    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent

        # ── 1. locate & load YAML ──────────────────────────────────────────────
        try:
            cfg_path = next(self.model_dir.glob("config.yaml"))
        except StopIteration:
            raise FileNotFoundError(
                f"No 'config.yaml' found in {self.model_dir}. "
                "Place a config file in that directory or pass a different path."
            )

        with open(cfg_path, "r", encoding="utf-8") as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh) or {}

        # convenience: keep constants handy
        self.constants = self.get_model_constants()

        # these will be filled later
        self.data: pd.DataFrame | None = None
        self.id2ticker: dict[int, str] | None = None
        self.date_lookup: pd.DataFrame | None = None

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
            "EXEC_PROVIDER":       c.get("execution_provider", "CPUExecutionProvider"),
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
            target_normalizer=target_normalizer,
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
        model_path = _find_model(const["ONNX_MODEL_PATH"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_path,
            providers=["TensorrtExecutionProvider", "CPUExecutionProvider"],
            sess_options=opts
        )
        print(f"Loaded ONNX Model: {const['ONNX_MODEL_PATH']}")

        input_names = {i.name for i in session.get_inputs()}
        output_name = session.get_outputs()[0].name

        # ––––––––––––––– 2 Dataloader –––––––––––––––
        loader = tft_dataset.to_dataloader(
            train=False,
            batch_size=const['BATCH_SIZE'],
            shuffle=False,
            num_workers=const['NUM_WORKERS'],
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


def main():
    """Example usage"""
    inference = Tempusv3Inference()

    if DataModule is None:
        print("DataModule not available - cannot run example")
        return

    # Load data using datamodule
    try:
        datamodule = DataModule(config=inference.constants)
        data = datamodule.prepare_data()

        if data is not None:
            # Run inference
            print("Beginning Inference")
            results = inference.predict(data)
            results.to_parquet("predictions.parquet")
            print(f"Generated {len(results)} predictions")
            print(results.head())
        else:
            print("No data available for inference")
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()
