# CEEMD-TFT: CompleteEnsemble Empirical Mode Decomposition with Temporal Fusion Transformer

This repository contains an implementation of a deep learning model that combines CompleteEnsemble Empirical Mode Decomposition (CEEMD) with a Temporal Fusion Transformer (TFT) for multi-horizon stock price forecasting.

## Model Architecture

The CEEMD-TFT model consists of two main components:

1. **CEEMD Decomposition**: Decomposes stock time-series data into multiple Intrinsic Mode Functions (IMFs) representing different frequency components.
2. **Temporal Fusion Transformer**: A state-of-the-art architecture for multi-horizon time series forecasting that combines variable selection, gated residual networks, LSTM layers, and multi-head attention.

### Key Features

- **Multi-scale Pattern Learning**: By decomposing time series into IMFs and processing them through the TFT, the model can capture patterns at different time scales.
- **Variable Selection**: The model uses variable selection networks to identify the most salient features for forecasting.
- **Gated Residual Networks**: Enable efficient information flow with skip connections and gating layers.
- **Time-dependent Processing**: Uses LSTMs for local processing and multi-head attention for integrating information from any timestep.
- **3-Day Forecasting**: The model is designed to forecast stock prices for the next 3 days (t+1, t+2, t+3).
- **Directional Accuracy Loss**: Incorporates a custom loss function that encourages the model to correctly predict the direction of price movements, not just their magnitude.

## Model Components

### CEEMD Decomposer

The CEEMD decomposer breaks down a time series into its constituent IMFs, which represent oscillations at different time scales. This allows the model to capture both short-term and long-term patterns in the data.

### Gated Residual Network (GRN)

GRNs are a key building block of the TFT architecture. They enable efficient information flow through:
- Skip connections that allow information to bypass layers
- Gating mechanisms that control how much information flows through each path
- Layer normalization for stable training

### Variable Selection Network (VSN)

The VSN helps the model focus on the most important features for forecasting by:
- Processing each input variable with its own GRN
- Computing variable importance weights
- Creating a weighted combination of processed variables

### Temporal Fusion Transformer (TFT)

The TFT combines:
- Static covariate encoders
- Variable selection networks for time-dependent features
- LSTM layers for local processing
- Multi-head attention for long-range dependencies
- Position-wise feed-forward networks

### Directional Accuracy Loss

The model uses a custom loss function that combines Mean Squared Error (MSE) with a directional accuracy component:

- **MSE Loss**: Measures the squared difference between predicted and actual values
- **Directional Component**: Encourages the model to correctly predict the sign of price changes
- **Combined Loss**: `loss = (1 - alpha) * mse_loss + alpha * directional_loss`

The directional component penalizes the model when its directional accuracy falls below the target accuracy. This helps the model learn to predict not just the magnitude of price movements but also their direction, which is crucial for trading strategies.

## Usage

### Installation

```bash
# Install required packages
pip install torch pandas numpy matplotlib PyEMD
```

### Example

```python
from Components.TickerData import TickerData
from Components.CEEMD_TFT_Model import CEEMD_TFT_Model

# Define configuration
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_size": 128,
    "lstm_layers": 2,
    "dropout": 0.1,
    "num_heads": 4,
    "forecast_horizon": 3,  # 3 days forecast
    "window_size": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_imfs": 5,
    "patience": 10,
    "num_epochs": 50,
    "noise_std": 0.05,
    "trials": 50,
    "static_dim": 0,
    "time_varying_categorical_dim": 1,
    "time_varying_real_dim": 5,
    "dir_alpha": 0.2,  # Weight for directional accuracy component
    "target_accuracy": 0.55  # Target directional accuracy
}

# Fetch and preprocess data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
indicators = ['ema_20', 'ema_50', 'stoch_rsi14', 'macd', 'hmm_state']
ticker_data = TickerData(tickers, indicators, years=2)
data, _ = ticker_data.process_all()

# Initialize the model
model = CEEMD_TFT_Model(config)

# Preprocess data with CEEMD decomposition
categorical_cols = ['hmm_state']
processed_data, scaler = model.preprocess_data(
    data, 
    target_col='shifted_prices', 
    categorical_cols=categorical_cols
)

# Create datasets
train_loader, val_loader, test_loader = model.create_datasets(
    processed_data,
    val_size=0.15,
    test_size=0.15,
    target_col='shifted_prices',
    categorical_cols=categorical_cols
)

# Train the model
history = model.train(train_loader, val_loader, test_loader)

# Make predictions
predictions, targets = model.predict(test_loader)

# Save the model
model.save_model('Models/ceemd_tft_model.pt')
```

### Full Example

For a complete working example, see the `ceemd_tft_example.py` script.

## Model Configuration

The model can be configured with the following parameters:

- `device`: Device to run the model on ("cpu" or "cuda")
- `hidden_size`: Size of hidden layers
- `lstm_layers`: Number of LSTM layers
- `dropout`: Dropout rate for regularization
- `num_heads`: Number of attention heads
- `forecast_horizon`: Number of future time steps to predict (default: 3)
- `window_size`: Size of the input window
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimization
- `weight_decay`: Weight decay for regularization
- `num_imfs`: Number of IMFs to extract from CEEMD
- `patience`: Patience for early stopping
- `num_epochs`: Maximum number of training epochs
- `noise_std`: Standard deviation of noise for CEEMD
- `trials`: Number of trials for CEEMD ensemble
- `static_dim`: Dimension of static features
- `time_varying_categorical_dim`: Dimension of categorical time-varying features
- `time_varying_real_dim`: Dimension of real-valued time-varying features
- `dir_alpha`: Weight for the directional accuracy component in the loss function (default: 0.2)
- `target_accuracy`: Target directional accuracy percentage (default: 0.55)

## References

1. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting.
2. Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011). A complete ensemble empirical mode decomposition with adaptive noise. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
