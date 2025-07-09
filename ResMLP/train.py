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
from resmlp_lightning import ResMLP
from datamodule_cs      import CrossSectionDM
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_forecasting import MAE, RMSE, MAPE, SMAPE
import sys

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


dm = CrossSectionDM(root="features_cs", batch_size=1)    # 1 macro-batch/step


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
        if self.config.get("use_wandb", True):
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
        """Prepare data using TFTDataModule."""
        print("Preparing data...")
        self.data_module = TFTDataModule(
            data_path=self.config.get("data_path", "raw_data_4k.csv"),
            batch_size=self.config["batch_size"],
            max_prediction_length=self.config.get("max_prediction_length", 3),
            max_encoder_length=self.config.get("max_encoder_length", 30),
            years=self.config.get("years", 5),
            prediction_window=self.config.get("prediction_window", 3),
            num_workers=self.config.get("num_workers", None),
            use_cache=self.config.get("use_cache", True),
            cache_dir=self.config.get("cache_dir", "data_cache")
        )

        # Setup datasets and dataloaders
        self.data_module.setup_dataloaders()
        print("Data preparation completed!")

    def create_model(self):
        """Create ResNET model from dataset"""
        training_dataset = self.data_module.get_training_dataset()

        self.model = ResMLP(n_features=len(training_dataset.feats),
                       lr=self.config["learning_rate"], wd=self.config["weight_decay"], d=self.config["width"],
                       n_blocks=self.config["blocks"], dropout=self.config["dropout"])

        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")

        return self.model

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
            "val_check_interval": self.config.get("val_check_interval", 0.25),
            "enable_checkpointing": True,
            "deterministic": False,
            "callbacks": callbacks,
        }

        # Add WandB logger if enabled
        if self.wandb_logger:
            trainer_kwargs["logger"] = self.wandb_logger

        self.trainer = Trainer(**trainer_kwargs)
        return self.trainer

    def train(self):
        """Train the model."""
        if self.data_module is None:
            self.prepare_data()

        if self.model is None:
            self.create_model()

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

            # Get dataloaders
            train_dataloader, val_dataloader = self.data_module.get_dataloaders()

            # Train model
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            # Log model to MLFlow
            #mlflow.pytorch.log_model(self.model, artifact_path="pt_model")

            print("Model training completed!")
            return self.trainer.checkpoint_callback.best_model_path

    def evaluate_model(self, best_model_path: str):
        """Evaluate the trained model and calculate metrics."""
        print("Evaluating model...")

        # Load best model
        best_model = ResMLP.load_from_checkpoint(best_model_path)

        # Get validation dataloader
        _, val_dataloader = self.data_module.get_dataloaders()

        # Calculate metrics
        metrics = {"MAE": MAE(), "RMSE": RMSE(), "MAPE": MAPE(), "SMAPE": SMAPE()}
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

                y_hat_single = y_hat[..., 0]
                for metric in metrics.values():
                    metric.update(y_hat_single, y)

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
        """Plot prediction examples."""
        print(f"Generating {num_examples} prediction plots...")

        _, val_dataloader = self.data_module.get_dataloaders()

        # Get raw predictions
        raw = model.predict(
            val_dataloader,
            mode="raw",
            return_x=True,
            trainer_kwargs={"accelerator": "cpu"}
        )

        x, out = raw.x, raw.output

        if isinstance(out.get("prediction"), list):
            out["prediction"] = torch.stack(out["prediction"], dim=1)

        # Generate plots
        plot_paths = []
        prediction_data = []

        for idx in range(min(num_examples, out["prediction"].size(0))):
            fig = model.plot_prediction(
                x,
                out,
                idx=idx,
                add_loss_to_title=True,
                show_future_observed=True
            )

            plot_path = f"plots/prediction_{idx}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plot_paths.append(plot_path)
            plt.close(fig)

            # Extract prediction data for wandb table
            try:
                # Get actual and predicted values for this example
                if hasattr(x, 'get') and 'target' in x:
                    actual_values = x['target'][idx].cpu().numpy() if hasattr(x['target'][idx], 'cpu') else x['target'][idx]
                else:
                    actual_values = None

                predicted_values = out["prediction"][idx].cpu().numpy() if hasattr(out["prediction"][idx], 'cpu') else out["prediction"][idx]

                # Create data row for this prediction
                pred_row = {
                    "example_id": idx,
                    "predicted_values": predicted_values.tolist() if hasattr(predicted_values, 'tolist') else predicted_values,
                    "plot_path": plot_path
                }

                if actual_values is not None:
                    pred_row["actual_values"] = actual_values.tolist() if hasattr(actual_values, 'tolist') else actual_values

                prediction_data.append(pred_row)
            except Exception as e:
                print(f"Warning: Could not extract prediction data for example {idx}: {e}")

        # Log plots to MLFlow
        if self.config.get("use_mlflow", True):
            for plot_path in plot_paths:
                mlflow.log_artifact(plot_path)

        # Log plots and prediction data to WandB
        if self.wandb_logger:
            import wandb

            # Log individual plot images
            for i, plot_path in enumerate(plot_paths):
                wandb.log({f"prediction_plot_{i}": wandb.Image(plot_path)})

            # Create and log prediction table
            if prediction_data:
                try:
                    # Create table columns
                    columns = ["example_id", "predicted_values"]
                    if "actual_values" in prediction_data[0]:
                        columns.append("actual_values")
                    columns.append("plot_image")

                    # Create table data
                    table_data = []
                    for row in prediction_data:
                        table_row = [
                            row["example_id"],
                            str(row["predicted_values"])[:100] + "..." if len(str(row["predicted_values"])) > 100 else str(row["predicted_values"])
                        ]
                        if "actual_values" in row:
                            table_row.append(str(row["actual_values"])[:100] + "..." if len(str(row["actual_values"])) > 100 else str(row["actual_values"]))
                        table_row.append(wandb.Image(row["plot_path"]))
                        table_data.append(table_row)

                    # Create and log table
                    predictions_table = wandb.Table(columns=columns, data=table_data)
                    wandb.log({"predictions_table": predictions_table})
                    print("Prediction data logged to wandb table")
                except Exception as e:
                    print(f"Warning: Could not create wandb predictions table: {e}")

        print(f"Prediction plots saved to: {plot_paths}")
        return plot_paths

    def export_onnx(self, model, model_name: str = "tft_model.onnx"):
        """Export model to ONNX format."""
        print("Exporting model to ONNX...")

        _, val_dataloader = self.data_module.get_dataloaders()

        # Get sample batch
        batch = next(iter(val_dataloader))
        sample_x, sample_y = batch

        # Move to CPU
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        model.freeze()

        batch_cpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_x.items()}

        example_inputs = (
            batch_cpu["encoder_cont"],
            batch_cpu["encoder_cat"],
            batch_cpu["decoder_cont"],
            batch_cpu["decoder_cat"],
            batch_cpu["encoder_lengths"],
            batch_cpu["decoder_lengths"],
            batch_cpu["target_scale"],
        )

        onnx_path = f"TFT/Models/{self.config.get('onnx_model_name','model.onnx')}"

        dynamic_axes = {
            "enc_cont": {0: "batch", 1: "enc_time"},
            "enc_cat": {0: "batch", 1: "enc_time"},
            "dec_cont": {0: "batch", 1: "dec_time"},
            "dec_cat": {0: "batch", 1: "dec_time"},
            "enc_len": {0: "batch"},
            "dec_len": {0: "batch"},
            "target_scale": {0: "batch"},
            "output": {0: "batch", 1: "dec_time"},
        }

        torch.onnx.export(
            model,
            example_inputs,
            onnx_path,
            input_names=["enc_cont", "enc_cat", "dec_cont", "dec_cat", "enc_len", "dec_len", "target_scale"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )

        # Log ONNX model to MLFlow
        if self.config.get("use_mlflow", True):
            mlflow.log_artifact(onnx_path)

        # Log ONNX model to W&B Model Registry
        if self.wandb_logger:
            import wandb

            # Create model artifact with proper registry naming and versioning
            model_artifact = wandb.Artifact(
                name="TEMPUS_tft",
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

    return f"tft_{timestamp}_{param_hash}"


def get_default_config():
    """Get default configuration for TFT training."""
    return {
        # Model parameters
        "epochs": 15,
        "width": 384,
        "blocks": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "dropout": 0.1,
        "gradient_clip": 0.5,
        "accumulate_grad_batches": 4,

        # Data parameters
        "years": 5,
        "prediction_window": 3,
        "use_cache": True,
        "cache_dir": "data_cache",

        # Training parameters
        "accelerator": "gpu",
        "precision": "bf16-mixed",
        "optimizer": "Adafactor",
        "val_check_interval": 0.25,
        "early_stopping": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-5,
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

