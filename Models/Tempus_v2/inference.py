import os
import sys
import yaml
import pandas as pd
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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

class Tempusv2Inference:
    """Inference class for Tempus v2 model"""
    def __init__(self):
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

        self.data: pd.DataFrame | None = None

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features according to model metadata"""
        required_features = self.constants['TV_KNOWN_REAL'] + ['Ticker','date']

        # Check if all required features are present
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select and order features according to metadata
        feature_data = data[required_features].copy()
        return feature_data

    def get_model_constants(self) -> Dict[str, Any]:
        """Translate YAML keys into the constants the pipeline expects."""
        c = self.config  # shorthand
        return {
            "TV_KNOWN_REAL":       c["dataset_parameters"]["value"]["time_varying_unknown_reals"],
            "ONNX_MODEL_PATH":     str(c["onnx_model_name"]["value"]),
            "WINDOW_SIZE":         str(c["window_size"]["value"]),
            "EXEC_PROVIDER":       c.get("execution_provider", "CPUExecutionProvider"),
            "BATCH_SIZE":          int(c["batch_size"]["value"]),
            "NUM_WORKERS":         int(c.get("num_workers", 0)),
            "SAMPLE_SIZE":         100,
        }

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run ONNX-exported model on dateset and return 1-point prediction.

        Returns columns:
            Ticker, date, prediction
        """
        const = self.constants
        feature_data = self.prepare_features(data)
        window_size = int(const['WINDOW_SIZE'])
        # ––––––––––––––– 1 Create ONNX session –––––––––––––––
        model_path = _find_model(const["ONNX_MODEL_PATH"])
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        print(f"Loaded ONNX Model: {const['ONNX_MODEL_PATH']}")
        input_name = session.get_inputs()[0].name

        preds, groups, times = [], [], []
        total_iterations = len(feature_data) - window_size
        for i in tqdm(range(window_size, len(feature_data)),
                      desc="Processing predictions",
                      total=total_iterations):
            # Get feature window (excluding Ticker column)
            values = feature_data.drop(columns=['Ticker','date']).values.astype(np.float32)

            input_window = values[i - window_size:i]

            input_window = np.expand_dims(input_window, axis=0)

            # Run inference
            output = session.run(None, {input_name: input_window})

            preds.append(float(output[0].squeeze()))
            groups.append(feature_data['Ticker'].iloc[i])
            times.append(feature_data['date'][i])

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Ticker': groups,
            'Predicted': preds
        }, index=pd.DatetimeIndex(times))

        return results_df

def main():
    """Example usage"""
    inference = Tempusv2Inference()

    if DataModule is None:
        print("DataModule not available - cannot run example")
        return

    # Load data using datamodule
    try:
        datamodule = DataModule(config=inference.constants)
        data = datamodule.prepare_data()

        if data is not None:
            # Run inference
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
