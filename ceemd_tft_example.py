import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Components.TickerData import TickerData
from Components.CEEMD_TFT_Model import CEEMD_TFT_Model

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    indicators = ['Ticker','ema_20', 'ema_50', 'ema_100', 'stoch_rsi14', 'macd', 'b_percent', 'hmm_state', 'dmd', 'hurst_100',
                  'perm_entropy_50', 'close_denoised_L1', 'fractal_50', 'cci', 'cmf', 'keltner_upper', 'keltner_lower', 'nasdaq_rsi', 'nasdaq_returns',
                  'shifted_prices']

    # Define configuration
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_size": 128,
        "lstm_layers": 2,
        "dropout": 0.1,
        "num_heads": 8,
        "forecast_horizon": 3,  # 3 days forecast
        "window_size": 32,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_imfs": 0,
        "patience": 10,
        "num_epochs": 50,
        "noise_std": 0.05,
        "trials": 50,  # Reduced for faster computation
        "static_dim": 0,
        "time_varying_categorical_dim": 1,  # For hmm_state
        "time_varying_real_dim": len(indicators) - 3,  # For technical indicators
        "dir_alpha": 0.1,  # Weight for directional accuracy component
        "target_accuracy": 0.80,  # Target directional accuracy

        # Ada Lovelace specific optimizations
        "use_amp": True,
        "compile_model": False,  # Use torch.compile for better performance

    }

    print(f"Using device: {config['device']}")

    # Fetch and preprocess data
    print("Fetching stock data...")
    data, raw_stock_data = pd.read_csv('training_data.csv'), pd.read_csv('raw_stock_data.csv')
    data = data.set_index(data['date']).drop(columns=['date'])
    unique_tickers = pd.Series(data["Ticker"].dropna().unique())
    sampled_tickers = unique_tickers.sample(n=2000, random_state=42).tolist()
    data = data[data["Ticker"].isin(sampled_tickers)].dropna()

    data = data[indicators]

    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns}")

    # Initialize the CEEMD-TFT model
    model = CEEMD_TFT_Model(config)
    if config["compile_model"] and hasattr(torch, 'compile'):
        model.model = torch.compile(model.model, backend="inductor",mode="default")

    # Preprocess data with CEEMD decomposition
    print("Applying CEEMD decomposition...")
    categorical_cols = ['hmm_state']
    processed_data, scaler = model.preprocess_data(
        data, 
        target_col='shifted_prices', 
        categorical_cols=categorical_cols
    )

    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed columns: {processed_data.columns}")

    # Create datasets
    print("Creating datasets...")
    train_loader, val_loader, test_loader = model.create_datasets(
        processed_data,
        val_size=0.15,
        test_size=0.15,
        target_col='shifted_prices',
        categorical_cols=categorical_cols
    )

    # Train the model
    print("Training the model...")
    history = model.train(train_loader, val_loader, test_loader)

    # Plot training history
    plt.figure(figsize=(15, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()

    # Plot directional accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_dir_acc'], label='Train Dir Acc')
    plt.plot(history['val_dir_acc'], label='Validation Dir Acc')
    plt.plot(history['test_dir_acc'], label='Test Dir Acc')
    plt.axhline(y=config['target_accuracy'] * 100, color='r', linestyle='--', label=f'Target ({config["target_accuracy"] * 100}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Directional Accuracy (%)')
    plt.title('Directional Accuracy History')
    plt.legend()

    # Make predictions
    print("Making predictions...")
    predictions, targets = model.predict(test_loader)

    # Plot predictions for the first sample
    plt.subplot(2, 2, 3)
    plt.plot(targets[0], label='Actual')
    plt.plot(predictions[0], label='Predicted')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('3-Day Forecast Example')
    plt.legend()

    # Plot directional changes
    plt.subplot(2, 2, 4)
    # Calculate day-to-day changes
    actual_changes = np.diff(targets[0])
    predicted_changes = np.diff(predictions[0])
    days = np.arange(len(actual_changes))

    plt.bar(days - 0.2, actual_changes, width=0.4, label='Actual Changes', color='blue', alpha=0.6)
    plt.bar(days + 0.2, predicted_changes, width=0.4, label='Predicted Changes', color='orange', alpha=0.6)

    # Calculate directional accuracy for this sample
    correct_directions = (np.sign(actual_changes) == np.sign(predicted_changes)).mean() * 100
    plt.title(f'Price Changes (Dir. Accuracy: {correct_directions:.1f}%)')
    plt.xlabel('Day')
    plt.ylabel('Price Change')
    plt.legend()

    plt.tight_layout()
    plt.savefig('ceemd_tft_results.png')
    plt.show()

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # Calculate directional accuracy
    pred_signs = np.sign(predictions[:, 1:] - predictions[:, :-1])
    target_signs = np.sign(targets[:, 1:] - targets[:, :-1])
    dir_acc = np.mean(pred_signs == target_signs) * 100

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test Directional Accuracy: {dir_acc:.2f}%")

    # Save the model
    model.save_model('Models/ceemd_tft_model.pt')
    print("Model saved to Models/ceemd_tft_model.pt")

    # Example of how to load and use the model for inference
    print("\nExample of model inference:")

    # Load the model
    loaded_model = CEEMD_TFT_Model(config)
    loaded_model.load_model('Models/ceemd_tft_model.pt')

    # Get a sample from the test set
    sample_inputs, sample_targets = next(iter(test_loader))

    # Move to device
    for key in sample_inputs:
        sample_inputs[key] = sample_inputs[key].to(config['device'])

    # Make prediction
    loaded_model.model.eval()
    with torch.no_grad():
        sample_predictions = loaded_model.model(sample_inputs)

    # Convert to numpy
    sample_predictions = sample_predictions.cpu().numpy()
    sample_targets = sample_targets.numpy()

    # Print results
    print("Sample predictions (3-day forecast):")
    for i in range(min(3, len(sample_predictions))):
        print(f"Sample {i+1}:")
        print(f"  Predicted: {sample_predictions[i]}")
        print(f"  Actual: {sample_targets[i]}")
        print(f"  Error: {np.abs(sample_predictions[i] - sample_targets[i])}")
        print()

if __name__ == "__main__":
    main()
