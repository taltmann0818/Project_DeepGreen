import os
import argparse
import mlflow
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import hashlib
from Models.ResMLP.resmlp_lightning import ResMLPLightning
from datamodule import DataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
import sys

# Enable optimizations
#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True
#torch.backends.cuda.enable_flash_sdp(True)

class ResNetTrainer:
    """
    Modular TFT trainer with MLFlow and WandB integration for rapid experimentation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_module = None
        self.model = None
        self.trainer = None
        self.wandb_logger = None

        # Setup directories
        self.setup_directories()

        # Setup logging
        self.setup_logging()

    def setup_directories(self):
        """Create necessary directories."""
        Path("ckpts").mkdir(exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("plots").mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup WandB and MLFlow logging."""
        # WandB Logger
        numb = True
        if numb:
            wandb_config = self.config.copy()
            # Add notes to WandB config if provided
            if self.config.get("notes"):
                wandb_config["experiment_notes"] = self.config["notes"]

            self.wandb_logger = WandbLogger(
                project=self.config.get("wandb_project", "resNET-equities"),
                name=self.config.get("experiment_name", "resNET_experiment"),
                log_model=True,
                config=wandb_config,
                notes=self.config.get("notes", "")
            )

        # MLFlow setup
        if self.config.get("use_mlflow", True):
            azure_uri = (
                "azureml://eastus.api.azureml.ms/mlflow/v1.0/"
                "subscriptions/6d500bb9-25c2-4821-a253-a860046398df/"
                "resourceGroups/Project_DeepGreen/providers/"
                "Microsoft.MachineLearningServices/workspaces/DeepGreen"
            )

            # `set_tracking_uri` lives at the top level in full MLflow and
            # under `mlflow.tracking` in stripped-down builds.  We try the
            # top level first and fall back gracefully.
            try:
                mlflow.set_tracking_uri(azure_uri)
            except AttributeError:
                from mlflow.tracking import set_tracking_uri as _set_tracking_uri
                _set_tracking_uri(azure_uri)
            experiment_name = self.config.get("mlflow_experiment", "resNET_quant")
            try:
                mlflow.set_experiment(experiment_name)
            except AttributeError:
                from mlflow.tracking import set_experiment as _set_experiment
                _set_experiment(experiment_name)

    def prepare_data(self):
        """Prepare data using CrossSectionalDataModule."""
        self.data_module = DataModule(
            batch_size=self.config["batch_size"],
            feature_cols=self.config.get("feature_cols", ["Close"]),
            days=self.config.get("days", 252),
            sample_size=self.config.get("sample_size", 100),
            prediction_window=self.config.get("prediction_window", 3),
            num_workers=self.config.get("num_workers", 4),
            #use_cache=self.config.get("use_cache", True),
            cache_dir=self.config.get("cache_dir", "data_cache")
        )
        print("Data preparation completed!")

    def create_model(self):
        """Create ResNET model from dataset"""
        return ResMLPLightning(n_features=len(self.data_module.feature_cols),
                       lr=self.config["learning_rate"], wd=self.config["weight_decay"], d=self.config["width"],
                       n_blocks=self.config["blocks"], dropout=self.config["dropout"])

    def setup_trainer(self):
        """Setup PyTorch Lightning trainer with callbacks."""
        callbacks = []

        # Model checkpoint callback
        checkpoint_cb = ModelCheckpoint(
            dirpath="ckpts",
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        callbacks.append(checkpoint_cb)

        # Early stopping callback
        if self.config.get("early_stopping", True):
            early_stop_cb = EarlyStopping(
                monitor="val_loss",
                min_delta=self.config.get("early_stopping_min_delta", 1e-5),
                patience=self.config.get("early_stopping_patience", 15),
                verbose=False,
                mode="min"
            )
            callbacks.append(early_stop_cb)

        # Learning rate monitor
        lr_logger = LearningRateMonitor()
        callbacks.append(lr_logger)

        # Setup trainer
        trainer_kwargs = {
            "max_epochs": self.config["epochs"],
            "accelerator": self.config.get("accelerator", "gpu"),
            "precision": self.config.get("precision", "bf16-mixed"),
            "enable_model_summary": True,
            "gradient_clip_val": self.config["gradient_clip"],
            "accumulate_grad_batches": self.config["accumulate_grad_batches"],
            #"val_check_interval": self.config.get("val_check_interval", 0.5),
            "enable_checkpointing": True,
            "deterministic": False,
            "callbacks": callbacks,
        }

        # Add WandB logger if enabled
        if self.wandb_logger:
            trainer_kwargs["logger"] = self.wandb_logger

        self.trainer = pl.Trainer(**trainer_kwargs)
        return self.trainer

    def train(self):
        """Train the model."""
        if self.data_module is None:
            self.prepare_data()

        if self.model is None:
            self.model = self.create_model()

        if self.trainer is None:
            self.setup_trainer()

        print("Starting model training...")

        # Start MLFlow run
        with mlflow.start_run(run_name=self.config.get("experiment_name", "tft_experiment")) as run:
            # Log parameters
            mlflow.log_params(self.config)

            # Log notes as a tag if provided
            if self.config.get("notes"):
                mlflow.set_tag("notes", self.config["notes"])
                mlflow.set_tag("experiment_context", self.config["notes"])

            # Train model
            self.trainer.fit(
                self.model,
                datamodule=self.data_module,
            )

            # Log model to MLFlow
            #mlflow.pytorch.log_model(self.model, artifact_path="pt_model")

            print("Model training completed!")
            return self.trainer.checkpoint_callback.best_model_path

    def evaluate_model(self, best_model_path: str):
        """Evaluate the trained model and calculate metrics."""
        print("Evaluating model...")

        # Load best model
        best_model = ResMLPLightning.load_from_checkpoint(best_model_path)

        # Get validation dataloader
        val_dataloader = self.data_module.val_dataloader()

        # Calculate metrics
        metrics = {
            "MAE": MeanAbsoluteError(),
            "RMSE": MeanSquaredError(squared=False),  # RMSE is MSE with squared=False
            "MAPE": MeanAbsolutePercentageError()
        }
        device = torch.device('cpu')
        best_model = best_model.to(device)
        best_model.eval()

        with torch.no_grad():
            for x, y in val_dataloader:
                out = best_model(x)
                if isinstance(out, dict) and "prediction" in out:
                    y_hat = out["prediction"]
                elif hasattr(out, "prediction"):
                    y_hat = out.prediction
                elif isinstance(out, (tuple, list)):
                    y_hat = out[0]
                else:
                    y_hat = out

                # Ensure both tensors are 1D for metric calculation
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze()
                if y.dim() > 1:
                    y = y.squeeze()

                for metric in metrics.values():
                    metric.update(y_hat, y)

        # Compute final metrics
        results = {name: metric.compute().item() for name, metric in metrics.items()}

        print(f"Model Evaluation Results:")
        for name, value in results.items():
            print(f"  {name}: {value:.4f}")

        # Log metrics to MLFlow
        if self.config.get("use_mlflow", True):
            mlflow.log_metrics(results)

        return results, best_model

    def plot_predictions(self, model, num_examples: int = 5):
        """Generate prediction plots for visualization."""
        print("Generating prediction plots...")

        val_dataloader = self.data_module.val_dataloader()
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()

        plots = []
        examples_plotted = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_dataloader):
                if examples_plotted >= num_examples:
                    break

                out = model(x)
                if isinstance(out, dict) and "prediction" in out:
                    y_hat = out["prediction"]
                elif hasattr(out, "prediction"):
                    y_hat = out.prediction
                elif isinstance(out, (tuple, list)):
                    y_hat = out[0]
                else:
                    y_hat = out

                # Ensure proper dimensions for plotting
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze()
                if y.dim() > 1:
                    y = y.squeeze()

                # Create plot for each sample in batch
                batch_size = min(x.size(0), num_examples - examples_plotted)
                for i in range(batch_size):
                    plt.figure(figsize=(10, 6))
                    plt.plot(y[i].cpu().numpy(), label='Actual', marker='o')
                    plt.plot(y_hat[i].cpu().numpy(), label='Predicted', marker='s')
                    plt.title(f'Prediction vs Actual - Example {examples_plotted + 1}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True)

                    plot_path = f"plots/prediction_example_{examples_plotted + 1}.png"
                    plt.savefig(plot_path)
                    plt.close()

                    plots.append(plot_path)
                    examples_plotted += 1

                    if examples_plotted >= num_examples:
                        break

        return plots

    def export_onnx(self, model, model_name: str = "tft_model.onnx"):
        """Export model to ONNX format."""
        print("Exporting model to ONNX...")
        # Get sample batch
        batch, _ = next(iter(self.data_module.val_dataloader()))

        # Move to CPU
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        model.freeze()

        onnx_path = f"TFT/Models/{self.config.get('onnx_model_name','model.onnx')}"

        torch.onnx.export(
            model,
            batch.to("cpu"),
            onnx_path,
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=17,
            dynamo=True
        )

        # Log ONNX model to MLFlow
        if self.config.get("use_mlflow", True):
            mlflow.log_artifact(onnx_path)

        # Log ONNX model to W&B Model Registry
        if self.wandb_logger:
            import wandb

            # Create model artifact with proper registry naming and versioning
            model_artifact = wandb.Artifact(
                name="TEMPUS_resnet",
                type="model",
                description="ResNET model for financial time series forecasting",
                metadata={
                    **self.config,
                    "model_type": "Deep Learning Residual neural network with Multilayer Perceptron",
                    "framework": "pytorch_lighting",
                    "export_format": "onnx",
                    "target": "Close"
                }
            )

            # Add the ONNX model file
            model_artifact.add_file(onnx_path, name=self.config.get("onnx_model_name","model.onnx"))

            # Log the artifact to the model registry
            # This will automatically version the model
            wandb.log_artifact(model_artifact)

            # Also link it to the model registry for easy access
            try:
                # Create or update model registry entry
                wandb.run.link_artifact(model_artifact, "model-registry/TEMPUS_resNET")
                print("Model logged to registry as TEMPUS_resNET with automatic versioning")
            except Exception as e:
                print(f"Warning: Could not link to model registry: {e}")
                print("Model artifact logged successfully without registry linking")

        print(f"ONNX model exported to: {onnx_path}")
        return onnx_path

    def run_full_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        print("Starting full ResNET training pipeline...")

        # Train model
        best_model_path = self.train()

        # Evaluate model
        metrics, best_model = self.evaluate_model(best_model_path)

        # Generate plots
        prediction_plots = self.plot_predictions(best_model, num_examples=self.config.get("num_prediction_plots", 5))

        # Export ONNX
        onnx_path = self.export_onnx(best_model, f"{self.config.get('experiment_name','tft_model')}.onnx")

        print("Full pipeline completed successfully!")

        return {
            "best_model_path": best_model_path,
            "metrics": metrics,
            "prediction_plots": prediction_plots,
            "onnx_path": onnx_path
        }


def generate_auto_experiment_name(config: Dict[str, Any]) -> str:
    """Generate automatic experiment name based on configuration and timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a short hash of key parameters
    key_params = {
        "lr": config.get("learning_rate", 0.002),
        "d": config.get("depth", 256),
        "b": config.get("blocks", 8),
        "dropout": config.get("dropout", 0.1),
        "epochs": config.get("epochs", 10)
    }

    # Create hash from parameters
    param_str = "_".join([f"{k}{v}" for k, v in key_params.items()])
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]

    return f"resNET_{timestamp}_{param_hash}"


def get_default_config():
    """Get default configuration for ResNET training."""
    return {
        # Model parameters
        "epochs": 20,
        "width": 384,
        "blocks": 8,
        "learning_rate": 1e-3,  # Reduced for better stability
        "weight_decay": 1e-3,   # Reduced from 1e-2 for better stability
        "dropout": 0.1,
        "gradient_clip": 1.0,   # Increased for better gradient stability
        "accumulate_grad_batches": 4,

        # Data parameters
        "days": 1260,
        "prediction_window": 3,
        "use_cache": True,
        "cache_dir": "data_cache",
        "batch_size": 64,       # Increased for more stable gradients
        "sample_size": 2000,

        # Training parameters
        "accelerator": "gpu",
        "precision": "bf16-mixed",      # Changed to full precision for numerical stability
        "optimizer": "AdamW",   # Changed from Adafactor for better stability
        "val_check_interval": 1,
        "early_stopping": True,
        "early_stopping_patience": 10,  # Increased patience
        "early_stopping_min_delta": 1e-4,  # Increased min delta
        "reduce_on_plateau_patience": 5,

        # Logging parameters
        "use_wandb": True,
        "use_mlflow": True,
        "wandb_project": "resNET-equities",
        "mlflow_experiment": "/Shared/resnet_experiments",
        "mlflow_uri": "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/6d500bb9-25c2-4821-a253-a860046398df/resourceGroups/Project_DeepGreen/providers/Microsoft.MachineLearningServices/workspaces/DeepGreen",
        "experiment_name": "tft_experiment",
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ResNET model with MLFlow and WandB logging")

    # Data parameters
    parser.add_argument("--no-cache", action="store_true", help="Disable data caching")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Directory for data cache")

    # Logging parameters
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--auto-name", action="store_true", help="Force auto-generation of experiment name")
    parser.add_argument("--notes", type=str, default="", help="Experiment notes/context for logging")
    parser.add_argument("--wandb-project", type=str, default="resNET-equities", help="WandB project name")
    parser.add_argument("--mlflow-experiment", type=str, default="resNet-quant", help="MLFlow experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLFlow logging")
    #parser.add_argument("--onnx-name", type=str, default="tft_model.onnx", help="ONNX model filename")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Get default config and update with command line arguments
    config = get_default_config()

    # Update config with command line arguments
    config.update({
        "use_cache": not args.no_cache,
        "cache_dir": args.cache_dir,
        "wandb_project": args.wandb_project,
        "mlflow_experiment": args.mlflow_experiment,
        "use_wandb": not args.no_wandb,
        "use_mlflow": not args.no_mlflow,
        #"onnx_model_name": args.onnx_name,
        "notes": args.notes
    })

    # Handle experiment naming (auto-generate if not provided or if auto-name is requested)
    if args.auto_name or args.experiment_name is None:
        config["experiment_name"] = generate_auto_experiment_name(config)
    else:
        config["experiment_name"] = args.experiment_name
    config["onnx_model_name"] = f"{config['experiment_name']}.onnx"

    print("ResNET Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create trainer and run pipeline
    trainer = ResNetTrainer(config)
    results = trainer.run_full_pipeline()

    print("\nTraining Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
