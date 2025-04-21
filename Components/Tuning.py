
import ray
from ray import tune
from ray.air import session
#from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import os
from functools import partial

# Define a training function for Ray Tune
def train_model(config, input_size=None, train_data=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module = DataModule(train_data, window_size=config["window_size"], batch_size=config["batch_size"])
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    test_loader = data_module.test_loader
    config["input_size"] = data_module.num_features

    # Create model with the hyperparameter configuration
    model = TEMPUS(config,scaler=data_module.scaler).to(device)
    # Define loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

    # Training loop
    for epoch in range(10):  # Limit epochs for tuning
        # Training
        model.train()
        train_loss, train_rmse, train_mape = model._train_epoch(train_loader, criterion, optimizer)
        # Validation phase
        val_loss, val_rmse, val_mape = model.evaluate(val_loader, criterion)
        # Test phase
        test_loss, test_rmse, test_mape = model.evaluate(test_loader, criterion)
        # Update learning rate
        scheduler.step(val_loss)

        # Report metrics to Ray Tune
        session.report({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "epoch": epoch
        })

# %%
# Initialize Ray
#ray.init(num_gpus=1)

# Define the hyperparameter search space
config = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "hidden_size": tune.choice([32, 64, 128, 256]),
    "num_layers": tune.choice([1, 2, 3]),
    "dropout": tune.uniform(0.1, 0.5),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "window_size": tune.choice([5, 10, 20, 50, 100, 200]),
    "clip_size": tune.uniform(0.0, 1.0),
    "attention_heads": tune.choice([4, 16, 32, 64, 128])
}

# Configure the ASHA scheduler
scheduler = ASHAScheduler(
    max_t=10,  # Maximum number of epochs
    grace_period=1,
    reduction_factor=2
)

# Set up the tuner
tuner = tune.Tuner(
    tune.with_resources(
        partial(
            train_model,
            train_data=None,
        ),
        resources={"cpu": 2, "gpu": 1}  # Adjust based on your hardware
    ),
    tune_config=tune.TuneConfig(
        metric="mape",
        mode="min",
        scheduler=scheduler,
        num_samples=50,  # Number of hyperparameter combinations to try
        trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id[:4]}"
    ),
    param_space=config
)

# Run the hyperparameter search
results = tuner.fit()

# %%
# Get the best hyperparameters
best_result = results.get_best_result("mape", "min")
best_config = best_result.config
print("Best config:", best_config)

# Extract the best hyperparameters
best_lr = best_config["lr"]
best_hidden_size = best_config["hidden_size"]
best_num_layers = best_config["num_layers"]
best_dropout = best_config["dropout"]
best_weight_decay = best_config["weight_decay"]
best_batch_size = best_config["batch_size"]
best_window_size = best_config["window_size"]
best_gradclip_size = best_config["clip_size"]
best_attention_heads = best_config["attention_heads"]

# Plot results
df_results = results.get_dataframe()

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

# Plot learning rate vs validation accuracy
plt.subplot(2, 3, 1)
plt.scatter(df_results["config/lr"], df_results["mape"])
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Mean Average Percentage Error")

# Plot hidden size vs validation accuracy
plt.subplot(2, 3, 2)
plt.scatter(df_results["config/hidden_size"], df_results["mape"])
plt.xlabel("Hidden Size")
plt.ylabel("Mean Average Percentage Error")

# Plot num_layers vs validation accuracy
plt.subplot(2, 3, 3)
plt.scatter(df_results["config/num_layers"], df_results["mape"])
plt.xlabel("Number of Layers")
plt.ylabel("Mean Average Percentage Error")

# Plot dropout vs validation accuracy
plt.subplot(2, 3, 4)
plt.scatter(df_results["config/dropout"], df_results["mape"])
plt.xlabel("Dropout Rate")
plt.ylabel("Mean Average Percentage Error")

# Plot weight decay vs validation accuracy
plt.subplot(2, 3, 5)
plt.scatter(df_results["config/weight_decay"], df_results["mape"])
plt.xscale("log")
plt.xlabel("Weight Decay")
plt.ylabel("Mean Average Percentage Error")

# Plot batch size vs validation accuracy
plt.subplot(2, 3, 6)
plt.scatter(df_results["config/batch_size"], df_results["mape"])
plt.xlabel("Batch Size")
plt.ylabel("Mean Average Percentage Error")

plt.tight_layout()
plt.show()