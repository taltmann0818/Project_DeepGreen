import os
import sys
import yaml
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
    import importlib.util
    spec = importlib.util.spec_from_file_location("datamodule", model_dir / "datamodule.py")
    datamodule_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datamodule_module)
    TFTDataModule = datamodule_module.TFTDataModule

class TempusV2Inference:
    """Inference class for Tempus v2 model"""
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

        self.data: pd.DataFrame | None = None

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

    def get_model_constants(self) -> Dict[str, Any]:
        """Translate YAML keys into the constants the pipeline expects."""
        c = self.config  # shorthand
        return {
            "TV_KNOWN_REAL":       c["dataset_parameters"]["value"]["time_varying_unknown_reals"],
            "ONNX_MODEL_PATH":     str(c["onnx_model_name"]["value"]),
            "WINDOW_SIZE":         str(c["window_size"]["value"]),
            "EXEC_PROVIDER":       c.get("execution_provider", "CPUExecutionProvider"),
            "BATCH_SIZE":          int(c["batch_size"]["value"]),
            "NUM_WORKERS":         int(c.get("num_workers", 30)),
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
        # ––––––––––––––– 1 Create ONNX session –––––––––––––––
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        session = ort.InferenceSession(
            const["ONNX_MODEL_PATH"],
            sess_options=sess_options,
            providers=[const['EXEC_PROVIDER']]
        )
        input_name = session.get_inputs()[0].name

        preds, groups, times = [], [], []
        
        for i in range(const["WINDOW_SIZE"], len(feature_data)):
            # Get feature window (excluding Ticker column)
            feature_window = feature_data.iloc[i - window_size:i].values.astype(np.float32)

            # Add batch dimension: shape = (1, window_size, num_features)
            input_window = np.expand_dims(feature_window, axis=0)

            # Run inference
            output = self.session.run(None, {input_name: input_window})

            preds.append(float(output[0].squeeze()))
            groups.append(feature_data['Ticker'].iloc[i])
            times.append(feature_data.index[i])

        # Create results DataFrame
        results_df = pd.DataFrame({
            'Ticker': groups,
            'Predicted': preds
        }, index=times)

        return results_df

def main():
    """Example usage"""
    inference = TempusV2Inference()

    if TFTDataModule is None:
        print("TFTDataModule not available - cannot run example")
        return

    # Load data using datamodule
    try:
        datamodule = TFTDataModule(config=inference.constants)
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
