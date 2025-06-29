import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.cuda.amp import autocast, GradScaler
from transformer_engine.common.recipe import Format, DelayedScaling
import transformer_engine.pytorch as te
import copy

import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TEMPUS(nn.Module):
    """
    TEMPUS is a streamlined neural network model designed for efficient temporal data processing.
    It combines a single bidirectional LSTM with a Temporal Convolutional Network (TCN) and
    a multi-head attention mechanism for capturing temporal dependencies at different scales.

    The architecture is optimized for Ada Lovelace hardware with tensor cores, using
    mixed precision training and efficient tensor operations.

    :ivar device: The device to execute the model on (e.g., 'cpu', 'cuda').
    :ivar hidden_size: Number of hidden units used in LSTM and other layers.
    :ivar num_layers: Number of layers in the LSTM module.
    :ivar input_size: Number of input features per timestep.
    :ivar dropout: Dropout probability for regularization.
    :ivar clip_size: Gradient clipping threshold to prevent exploding gradients.
    :ivar tcn_kernel_size: Kernel size for the TCN layer.
    :ivar attention_heads: Number of attention heads used in the multi-head attention.
    :ivar learning_rate: Learning rate for the optimizer.
    :ivar weight_decay: Weight decay for L2 regularization in the optimizer.
    :ivar scaler: Feature scaler for the input data (optional).
    :type device: Str
    :type hidden_size: int
    :type num_layers: int
    :type input_size: int
    :type dropout: Float
    :type clip_size: float
    :type tcn_kernel_size: int
    :type attention_heads: int
    :type learning_rate: float
    :type weight_decay: float
    :type scaler: sklearn.preprocessing.StandardScaler or None
    """
    def __init__(self, config, scaler=None):
        super(TEMPUS, self).__init__()
        self.device = config.get("device", "cpu")
        self.hidden_size = config.get("hidden_size", 128)  # Increased default size for better representation
        self.num_layers = config.get("num_layers", 2)
        self.input_size = config.get("input_size", 10)
        self.dropout = config.get("dropout", 0.2)
        self.clip_size = config.get("clip_size", 1.0)
        self.tcn_kernel_size = config.get("tcn_kernel_size", 5)  # Single kernel size for efficiency
        self.attention_heads = config.get("attention_heads", 8)  # Increased heads for better attention
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.use_amp = config.get("use_amp", True)  # Enable automatic mixed precision by default
        self.use_fp8 = config.get("use_fp8", False)  # Enable FP8 precision for Ada Lovelace hardware

        # FP8 recipe for transformer engine
        self.fp8_recipe = None
        if self.use_fp8 and self.device != 'cpu':
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.E4M3,  # FP8 format for Ada Lovelace
                amax_history_len=16,     # Length of amax history
                amax_compute_algo="max", # Algorithm for computing amax
            )

        if scaler is not None:
            self.scaler = scaler
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))

        # Single efficient bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Layer normalization for LSTM output
        self.lstm_norm = nn.LayerNorm(self.hidden_size * 2)

        # Projection layer for residual connections
        if self.use_fp8 and self.device != 'cpu':
            self.residual_proj = te.Linear(self.input_size, self.hidden_size * 2, fp8_recipe=self.fp8_recipe)
        else:
            self.residual_proj = nn.Linear(self.input_size, self.hidden_size * 2)

        # Efficient TCN with single kernel size and dilation
        self.tcn = nn.Sequential(
            nn.Conv1d(self.input_size, self.hidden_size * 2, 
                      kernel_size=self.tcn_kernel_size,
                      padding=(self.tcn_kernel_size - 1) // 2, 
                      stride=1),
            nn.BatchNorm1d(self.hidden_size * 2),
            nn.GELU()  # GELU activation for better gradient flow
        )

        # Multi-head attention for temporal relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True
        )

        # Positional encoding for attention
        self.pos_encoder = PositionalEncoding(self.hidden_size * 2, self.dropout)

        # Streamlined fully connected layers
        if self.use_fp8 and self.device != 'cpu':
            # FP8-enabled fully connected layers
            self.fc = nn.Sequential(
                te.Linear(self.hidden_size * 2, self.hidden_size, fp8_recipe=self.fp8_recipe),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                te.Linear(self.hidden_size, self.hidden_size // 2, fp8_recipe=self.fp8_recipe),
                nn.LayerNorm(self.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                te.Linear(self.hidden_size // 2, 1, fp8_recipe=self.fp8_recipe)
            )
        else:
            # Standard fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size // 2, 1)
            )

    def downsample_sequence(self, x, factor):
        """Downsample time sequence by average pooling"""
        batch_size, seq_len, features = x.size()
        if seq_len % factor != 0:
            # Pad sequence if needed
            pad_len = factor - (seq_len % factor)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len += pad_len

        # Reshape for pooling
        x = x.view(batch_size, seq_len // factor, factor, features)
        # Average pool
        x = torch.mean(x, dim=2)
        return x

    def forward(self, x):
        """
        Forward pass through the TEMPUS model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, seq_len, 1)
        """
        # Apply scaling if available
        if hasattr(self, 'scaler') and self.scaler is not None:
            x = (x - self.mean) / self.scale

        batch_size, seq_len, features = x.size()

        # Create time features for positional information
        time_features = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)

        # Process with TCN
        x_tcn = x.transpose(1, 2)  # TCN expects (batch, channels, seq_len)
        tcn_features = self.tcn(x_tcn)
        tcn_features = tcn_features.transpose(1, 2)  # Back to (batch, seq, features)

        # Process with LSTM
        lstm_out, _ = self.lstm(x)
        lstm_features = self.lstm_norm(lstm_out)

        # Add residual connection
        x_residual = self.residual_proj(x)
        features = lstm_features + x_residual

        # Add positional encoding for attention
        features = self.pos_encoder(features)

        # Apply multi-head attention
        attn_output, _ = self.attention(
            query=features,
            key=features,
            value=features
        )

        # Combine with TCN features through residual connection
        combined_features = attn_output + tcn_features

        # Pass through final fully connected layers
        outputs = self.fc(combined_features)

        return outputs

    def train_model(self, train_loader, val_loader, test_loader, num_epochs=100, patience=10):
        """
        Train the model with a regression task using automatic mixed precision for Ada Lovelace hardware.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            num_epochs: Maximum number of training epochs
            patience: Number of epochs to wait for improvement before early stopping

        Returns:
            dict: Training history with metrics
        """
        self.to(self.device)

        # Define loss function and optimizer with weight decay
        criterion = nn.MSELoss()
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning rate scheduler with cosine annealing for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart period after each restart
            eta_min=1e-6  # Minimum learning rate
        )

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler(enabled=self.use_amp and self.device != 'cpu')

        # Early stopping variables
        best_val_mape = float('inf')
        patience_counter = 0
        best_model_state = None

        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'val_rmse': [], 'test_rmse': [],
            'val_mape': [], 'test_mape': [],
            'learning_rates': []
        }

        epoch_progress = tqdm(range(num_epochs), desc="Training Epochs")
        for epoch in epoch_progress:
            # Training phase with mixed precision
            train_loss = self._train_epoch(train_loader, criterion, optimizer, scaler)

            # Validation phase
            val_loss, val_rmse, val_mape = self.evaluate(val_loader, criterion)

            # Test phase
            test_loss, test_rmse, test_mape = self.evaluate(test_loader, criterion)

            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['test_rmse'].append(test_rmse)
            self.history['val_mape'].append(val_mape)
            self.history['test_mape'].append(test_mape)
            self.history['learning_rates'].append(current_lr)

            # Update progress
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'RMSE': f'{test_rmse:.4f}',
                'MAPE': f'{test_mape:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

            # Model selection and early stopping
            if val_mape < best_val_mape:
                best_val_mape = val_mape
                best_model_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Load the best model state
        if best_model_state is not None:
            print("Loading the best model state...")
            self.load_state_dict(best_model_state)

        # Final evaluation
        final_test_loss, final_test_rmse, final_test_mape = self.evaluate(test_loader, criterion)
        print(f"\nFinal Test Results | Loss: {final_test_loss:.4f}, RMSE: {final_test_rmse:.4f}, MAPE: {final_test_mape:.2f}%")

        return self.history

    def _train_epoch(self, train_loader, criterion, optimizer, scaler=None):
        """
        Helper method for training a single epoch with optional mixed precision.

        Args:
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            scaler: GradScaler for mixed precision training

        Returns:
            float: Average training loss for the epoch
        """
        self.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            # Use mixed precision for forward pass if scaler is provided
            if scaler is not None and self.use_amp and self.device != 'cpu':
                if self.use_fp8 and hasattr(self, 'fp8_recipe') and self.fp8_recipe is not None:
                    # Use transformer engine's FP8 autocast
                    with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                        outputs = self(inputs)
                        if outputs.dim() > 1:
                            outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                        loss = criterion(outputs, targets)
                else:
                    # Use standard PyTorch autocast (FP16/BF16)
                    with autocast():
                        outputs = self(inputs)
                        if outputs.dim() > 1:
                            outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                        loss = criterion(outputs, targets)

                # Scale gradients and optimize with mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_size)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                outputs = self(inputs)
                if outputs.dim() > 1:
                    outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_size)
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # Calculate metrics
        avg_loss = total_loss / len(train_loader.dataset)

        return avg_loss

    def evaluate(self, data_loader, criterion, use_amp=None):
        """
        Evaluate the model with improved metrics and optional mixed precision.

        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function
            use_amp: Whether to use automatic mixed precision (defaults to self.use_amp)

        Returns:
            tuple: (average loss, RMSE, MAPE)
        """
        self.eval()
        total_loss = 0
        all_predictions, all_targets = [], []

        # Default to model's use_amp setting if not specified
        if use_amp is None:
            use_amp = getattr(self, 'use_amp', False)

        # Only use mixed precision if on GPU
        use_amp = use_amp and self.device != 'cpu'

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Use mixed precision for inference if enabled
                if use_amp:
                    if self.use_fp8 and hasattr(self, 'fp8_recipe') and self.fp8_recipe is not None:
                        # Use transformer engine's FP8 autocast for inference
                        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                            outputs = self(inputs)
                            if outputs.dim() > 1:
                                outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                            loss = criterion(outputs, targets)
                    else:
                        # Use standard PyTorch autocast (FP16/BF16)
                        with autocast():
                            outputs = self(inputs)
                            if outputs.dim() > 1:
                                outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                            loss = criterion(outputs, targets)
                else:
                    outputs = self(inputs)
                    if outputs.dim() > 1:
                        outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                    loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        avg_loss = total_loss / len(data_loader.dataset)
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        # Improved MAPE calculation with epsilon to avoid division by zero
        epsilon = 1e-6
        abs_percentage_errors = np.abs((all_targets - all_predictions) / np.maximum(np.abs(all_targets), epsilon))
        # Clip extreme values for more stable MAPE
        abs_percentage_errors = np.clip(abs_percentage_errors, 0, 10)
        mape = np.mean(abs_percentage_errors) * 100

        return avg_loss, rmse, mape

    def plot_training_history(self):
        if self.history is not None:
            # Create subplots for loss and accuracy
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss Over Epochs',
                                                                'MAPE Over Epochs'))

            # Plot losses
            fig.add_trace(go.Scatter(y=self.history['train_loss'], name='Train Loss', line=dict(color='blue')),
                          row=1, col=1)
            fig.add_trace(go.Scatter(y=self.history['test_loss'], name='Test Loss', line=dict(color='green')),
                          row=1, col=1)
            fig.add_trace(go.Scatter(y=self.history['val_loss'], name='Validation Loss', line=dict(color='orange')),
                          row=1, col=1)
            fig.update_xaxes(title_text="Epochs", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)

            # Plot MAPE
            fig.add_trace(go.Scatter(y=self.history['test_mape'], name='Test MAPE', line=dict(color='green')),
                          row=1, col=2)
            fig.add_trace(go.Scatter(y=self.history['val_mape'], name='Validation MAPE', line=dict(color='orange')),
                          row=1, col=2)
            fig.update_xaxes(title_text="Epochs", row=1, col=2)
            fig.update_yaxes(title_text="MAPE %", row=1, col=2)

            fig.update_layout(
                title='Model Training Metrics',
                height=700,
                template='ggplot2',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            return fig

        else:
            print("Training history not available. Please run train_model() first.")
            return

    def export_model_to_torchscript(self, save_path, data_loader, device):
        """
        Export the model to TorchScript format for deployment.

        Args:
            save_path: Path to save the exported model
            data_loader: DataLoader to get example inputs
            device: Device to use for export

        Returns:
            str: Path to the saved model or None if export failed
        """
        try:
            self.eval()
            self.to(device)
            self.device = device

            # Fetch a sample input tensor from DataLoader
            example_inputs, _ = next(iter(data_loader))
            example_inputs = example_inputs.to(device)

            # Export model to TorchScript using tracing with optimization
            with torch.no_grad():
                # Use torch.jit.optimize_for_inference for better performance
                scripted_model = torch.jit.trace(self, example_inputs)
                optimized_model = torch.jit.optimize_for_inference(scripted_model)

            # Save the optimized TorchScript model
            torch.jit.save(optimized_model, save_path)

            print(f"Model successfully exported and saved to {save_path}")
            return save_path

        except Exception as e:
            print(f"Error exporting model to TorchScript: {str(e)}")
            return None

    def export_model_to_onnx(self, save_path, data_loader, device, opset_version=13):
        """
        Export the model to ONNX format for deployment on Ada Lovelace hardware.

        ONNX format provides better optimization for tensor cores on Ada Lovelace GPUs
        through TensorRT integration.

        Args:
            save_path: Path to save the exported model
            data_loader: DataLoader to get example inputs
            device: Device to use for export
            opset_version: ONNX opset version to use

        Returns:
            str: Path to the saved model or None if export failed
        """
        try:
            self.eval()
            self.to(device)
            self.device = device

            # Fetch a sample input tensor from DataLoader
            example_inputs, _ = next(iter(data_loader))
            example_inputs = example_inputs.to(device)

            # Set dynamic axes for variable batch size and sequence length
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }

            # Export to ONNX with optimization flags
            torch.onnx.export(
                self,                                      # model being run
                example_inputs,                            # model input
                save_path,                                 # where to save the model
                export_params=True,                        # store the trained parameter weights inside the model file
                opset_version=opset_version,               # the ONNX version to export the model to
                do_constant_folding=True,                  # whether to execute constant folding for optimization
                input_names=['input'],                     # the model's input names
                output_names=['output'],                   # the model's output names
                dynamic_axes=dynamic_axes,                 # variable length axes
                verbose=False
            )

            print(f"Model successfully exported to ONNX and saved to {save_path}")

            # Verify the ONNX model
            import onnx
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verified successfully")

            return save_path

        except Exception as e:
            print(f"Error exporting model to ONNX: {str(e)}")
            return None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)])

def torchscript_predict(model_path, input_df, device, window_size, target_col='shifted_prices', prediction_mode=False):
    # Load the TorchScript model
    loaded_model = torch.jit.load(model_path,map_location=device)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    predictions = []
    dates = []
    tickers = []
    if not prediction_mode:
        actuals = []

    for i in range(window_size, len(input_df)):
        # Get date, actual value, and ticker for the current index
        date = input_df.index[i]
        ticker = input_df['Ticker'].iloc[i] if 'Ticker' in input_df.columns else None

        if not prediction_mode:
            actual = input_df[target_col].iloc[i] if target_col in input_df.columns else None

        # Get a sequence
        if prediction_mode:
            values = input_df.drop(columns=['Ticker']).values.astype(np.float32)
        elif not prediction_mode:
            values = input_df.drop(columns=['Ticker',target_col]).values.astype(np.float32)
        input_window = values[i - window_size:i]
        input_tensor = torch.tensor(input_window, dtype=torch.float32, device=device).unsqueeze(0)

        # Predict the next value based on the previous window
        pred = loaded_model(input_tensor)
        # Handle different output shapes
        if pred.dim() > 1:
            pred = pred[:, -1, 0] if pred.size(1) > 1 and pred.size(2) > 0 else pred.squeeze()
        pred_value = pred.detach().cpu().numpy().item()

        predictions.append(pred_value)
        dates.append(date)
        tickers.append(ticker)
        if not prediction_mode:
            actuals.append(actual)

    # Create DataFrame with predictions
    if not prediction_mode:
        preds_df = pd.DataFrame({
            'Ticker': tickers,
            'Actual': actuals,
            'Predicted': predictions
        },index=dates)
    else:
        preds_df = pd.DataFrame({
            'Ticker': tickers,
            'Predicted': predictions
        },index=dates)

    return preds_df

class DataModule:
    def __init__(
        self,
        data,
        window_size=20,
        batch_size=32,
        val_size=0.1,
        test_size=0.1,
        target_col='shifted_prices',
    ):
        # Keep raw data (including Ticker) for grouping
        self.data = data.copy()
        self.window_size = window_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.target_col = target_col
        self.setup()

    def setup(self):
        # Verify index is DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex for splitting")

        # Determine global split dates for walk-forward chronological splits
        all_dates = pd.Series(self.data.index.unique()).sort_values()
        n = len(all_dates)
        train_cut = int(n * (1 - self.val_size - self.test_size))
        val_cut = int(n * (1 - self.test_size))
        train_date = all_dates.iloc[train_cut]
        val_date = all_dates.iloc[val_cut]

        # Split per ticker using global date boundaries
        train_dfs, val_dfs, test_dfs = [], [], []
        for ticker, df in self.data.groupby('Ticker'):
            df = df.sort_index()
            train_dfs.append(df[df.index <= train_date])
            val_dfs.append(df[(df.index > train_date) & (df.index <= val_date)])
            test_dfs.append(df[df.index > val_date])
        # Concatenate splits
        self.df_train = pd.concat(train_dfs)
        self.df_val = pd.concat(val_dfs)
        self.df_test = pd.concat(test_dfs)

        # Determine feature columns (exclude target and ticker)
        feature_cols = [c for c in self.df_train.columns if c not in [self.target_col, 'Ticker']]
        self.num_features = len(feature_cols)

        # Fit and apply scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_train[feature_cols])
        for df in [self.df_train, self.df_val, self.df_test]:
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        # Build per-ticker SequenceDatasets to prevent sequence bleed
        self.train_dataset = ConcatDataset([SequenceDataset(df.drop(columns=['Ticker']), self.target_col, feature_cols, self.window_size) for df in train_dfs])
        self.val_dataset = ConcatDataset([SequenceDataset(df.drop(columns=['Ticker']), self.target_col, feature_cols, self.window_size) for df in val_dfs])
        self.test_dataset = ConcatDataset([SequenceDataset(df.drop(columns=['Ticker']),self.target_col, feature_cols, self.window_size) for df in test_dfs])

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class SequenceDataset(Dataset):
    def __init__(self,dataframe,target, features,window_size):
        self.features = features
        self.target = target
        self.window_size = window_size
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if idx >= self.window_size - 1:
            start = idx - self.window_size + 1
            x = self.X[start:idx + 1]
        else:
            pad_len = self.window_size - idx - 1
            pad = torch.zeros(pad_len, self.X.shape[1])
            x = torch.cat([pad, self.X[:idx+1]], dim=0)
        return x, self.y[idx]


class EchoStateNetwork(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9,
                 sparsity=0.1, noise=0.001, bidirectional=False):
        super(EchoStateNetwork, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.bidirectional = bidirectional

        # Input weights (fixed)
        self.register_buffer('W_in', self._initialize_input_weights())

        # Reservoir weights (fixed)
        self.register_buffer('W', self._initialize_reservoir_weights())

        # Output weights (trainable)
        self.W_out = nn.Linear(reservoir_size, output_size)

        if bidirectional:
            # Second set of weights for backward direction
            self.register_buffer('W_in_reverse', self._initialize_input_weights())
            self.register_buffer('W_reverse', self._initialize_reservoir_weights())
            self.W_out_reverse = nn.Linear(reservoir_size, output_size)
            # Combined output
            self.W_combined = nn.Linear(output_size * 2, output_size)

    def _initialize_input_weights(self):
        W_in = torch.zeros(self.reservoir_size, self.input_size)
        W_in = torch.nn.init.xavier_uniform_(W_in)
        return W_in

    def _initialize_reservoir_weights(self):
        # Create sparse matrix
        W = torch.zeros(self.reservoir_size, self.reservoir_size)
        num_connections = int(self.sparsity * self.reservoir_size * self.reservoir_size)
        indices = torch.randperm(self.reservoir_size * self.reservoir_size)[:num_connections]
        rows = indices // self.reservoir_size
        cols = indices % self.reservoir_size
        values = torch.randn(num_connections)
        W[rows, cols] = values

        # Scale to desired spectral radius
        eigenvalues = torch.linalg.eigvals(W)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        W = W * (self.spectral_radius / max_eigenvalue)
        return W

    def _reservoir_step(self, x, h_prev, W_in, W):
        """Execute one step of the reservoir"""
        # h_new = tanh(W_in @ x + W @ h_prev + noise)
        h_new = torch.tanh(torch.mm(x, W_in.t()) + torch.mm(h_prev, W.t()) +
                           self.noise * torch.randn(h_prev.shape, device=h_prev.device))
        return h_new

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()

        # Forward pass
        h = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        outputs_forward = []

        for t in range(seq_len):
            h = self._reservoir_step(x[:, t], h, self.W_in, self.W)
            outputs_forward.append(self.W_out(h))

        outputs_forward = torch.stack(outputs_forward, dim=1)  # (batch_size, seq_len, output_size)

        if not self.bidirectional:
            return outputs_forward

        # Backward pass for bidirectional ESN
        h_reverse = torch.zeros(batch_size, self.reservoir_size, device=x.device)
        outputs_reverse = []

        for t in range(seq_len - 1, -1, -1):
            h_reverse = self._reservoir_step(x[:, t], h_reverse, self.W_in_reverse, self.W_reverse)
            outputs_reverse.insert(0, self.W_out_reverse(h_reverse))

        outputs_reverse = torch.stack(outputs_reverse, dim=1)  # (batch_size, seq_len, output_size)

        # Combine forward and backward outputs
        combined = torch.cat((outputs_forward, outputs_reverse), dim=2)
        return self.W_combined(combined)

# Implementation of custom Temporal Attention
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_attn = nn.Sequential(
            nn.Linear(1, 16),  # Simple time feature processing
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.feature_attn = nn.Linear(hidden_dim, 1)
        #self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, lstm_output, time_features):
        # lstm_output: [batch, seq_len, hidden]
        # time_features: [batch, seq_len, 1] - normalized position in sequence

        # Compute base attention scores from features
        feature_scores = self.feature_attn(lstm_output)  # [batch, seq_len, 1]
        # Compute time-based attention
        time_weights = self.time_attn(time_features)  # [batch, seq_len, 1]
        # Combine feature and time attention
        combined_scores = feature_scores + time_weights
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(combined_scores, dim=1)
        # Apply attention to get context vector
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_output)  # [batch, 1, hidden]

        return context, attention_weights
