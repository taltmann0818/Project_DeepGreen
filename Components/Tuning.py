
import ray
from ray import tune
from ray.air import session
#from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import os
from functools import partial

# Define a training function for Ray Tune
def train_lstm(config, input_size=None, train_data=None, val_data=None, test_data=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    data_module = DataModule(train_data, window_size=config["window_size"], batch_size=config["batch_size"])
    train_loader = data_module.train_loader
    test_loader = data_module.test_loader
    input_size = data_module.num_features

    # Create model with the hyperparameter configuration
    model = TEMPUS({
        "input_size": input_size,
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "dropout": config["dropout"],
        "device": device
    }).to(device)

    # Set up loss and optimizer
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Training loop
    for epoch in range(10):  # Limit epochs for tuning
        # Training
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).to(device).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        test_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).to(device).squeeze()
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate RMSE and MAPE
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100 # Avoid division by zero by adding a small epsilon

        # Report metrics to Ray Tune
        session.report({
            "train_loss": train_loss / len(train_loader),
            "test_loss": test_loss / len(test_loader),
            "rmse": rmse,
            "mape": mape,
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
    "window_size": tune.choice([10, 20, 50, 100, 200]),
    "grad_clip_norm": tune.uniform(0.0, 1.0)
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
            train_lstm,
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
best_gradclip_size = best_config["grad_clip_norm"]

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