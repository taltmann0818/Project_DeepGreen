import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TEMPUS(nn.Module):
    """
    TEMPUS is a neural network model designed to process and analyze temporal data
    by combining multiple aspects of time-series modeling.
    It incorporates LSTMs with multiple temporal resolutions, Temporal Convolutional Networks (TCNs),
    and Transformer encoders for temporal attention.

    The model integrates several functionalities with layer normalization and
    residual connections for efficient feature extraction, fusion, and sequence
    processing. It is primarily used for regression tasks on temporal data.

    :ivar device: The device to execute the model on (e.g., 'cpu', 'cuda').
    :ivar hidden_size: Number of hidden units used in LSTM and other layers.
    :ivar num_layers: Number of layers in both LSTM modules.
    :ivar input_size: Number of input features per timestep.
    :ivar dropout: Dropout probability for regularization.
    :ivar clip_size: Gradient clipping threshold to prevent exploding gradients.
    :ivar tcn_kernel_sizes: List of kernel sizes for TCN layers.
    :ivar attention_heads: Number of attention heads used in the Transformer encoder.
    :ivar learning_rate: Learning rate for the optimizer.
    :ivar weight_decay: Weight decay for L2 regularization in the optimizer.
    :ivar scaler: Feature scaler for the input data (optional).
    :type device: Str
    :type hidden_size: int
    :type num_layers: int
    :type input_size:
    :type dropout: Float
    :type clip_size: float
    :type tcn_kernel_sizes: list[int]
    :type attention_heads: int
    :type learning_rate: float
    :type weight_decay: float
    :type scaler: sklearn.preprocessing.StandardScaler or None
    """
    def __init__(self, config, scaler=None):
        super(TEMPUS, self).__init__()
        self.device = config.get("device", "cpu")
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.input_size = config.get("input_size", 10)
        self.dropout = config.get("dropout", 0.2)
        self.clip_size = config.get("clip_size", 1.0)
        self.tcn_kernel_sizes = config.get("tcn_kernel_sizes", [3, 5, 7])
        self.attention_heads = config.get("attention_heads", 4)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.01)

        if scaler is not None:
            self.scaler = scaler
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))

        # Multiple Temporal Resolutions of LSTM with layer normalization
        self.lstm_short = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        self.lstm_medium = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Layer normalization for LSTM outputs
        self.lstm_short_norm = nn.LayerNorm(self.hidden_size * 2)
        self.lstm_medium_norm = nn.LayerNorm(self.hidden_size * 2)

        # Fusion layer for temporal resolutions with residual connection
        self.temporal_fusion = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.temporal_fusion_norm = nn.LayerNorm(self.hidden_size * 2)

        # Projection layer for residual connections when dimensions don't match
        self.residual_proj = nn.Linear(self.input_size, self.hidden_size * 2)

        # Temporal Convolutional Network (TCN) with layer normalization 
        self.tcn_modules = nn.ModuleList()
        for i, k_size in enumerate(self.tcn_kernel_sizes):
            dilation = 2 ** i  # Exponentially increasing dilation
            padding = ((k_size - 1) * dilation) // 2  # Adjusted padding for dilation
            self.tcn_modules.append(nn.Sequential(
                nn.Conv1d(self.input_size, self.hidden_size, kernel_size=k_size,
                          padding=padding, dilation=dilation, stride=1),
                nn.BatchNorm1d(self.hidden_size),
                nn.GELU(), # Switching from ReLU to GELU
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=k_size,
                          padding=padding, dilation=dilation, stride=1),
                nn.BatchNorm1d(self.hidden_size),
                nn.GELU() # Switching from ReLU to GELU
            ))
        self.tcn_fusion = nn.Linear(self.hidden_size * len(self.tcn_kernel_sizes), self.hidden_size * 2)
        self.tcn_fusion_norm = nn.LayerNorm(self.hidden_size * 2)

        # Combine TCN and LSTM features
        self.feature_fusion = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.feature_fusion_norm = nn.LayerNorm(self.hidden_size * 2)

        # Transformer encoder for temporal attention (replacing custom attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size * 2,
            nhead=self.attention_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(self.hidden_size * 2, self.dropout)

        # Fully connected layers with dropout and layer normalization
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc1_norm = nn.LayerNorm(self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2_norm = nn.LayerNorm(self.hidden_size // 2)
        self.regression_head = nn.Linear(self.hidden_size // 2, 1)
        self.dropout_layer = nn.Dropout(self.dropout)

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
        if self.scaler is not None:
            x = (x - self.mean) / self.scale

        batch_size, seq_len, features = x.size()
        #time_features = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1).to(x.device)

        # Process with TCN
        tcn_outputs = []
        x_tcn = x.transpose(1, 2)  # TCN expects (batch, channels, seq_len)
        for tcn_module in self.tcn_modules:
            tcn_out = tcn_module(x_tcn)
            tcn_outputs.append(tcn_out)

        # Concatenate TCN outputs
        tcn_combined = torch.cat(tcn_outputs, dim=1)
        tcn_combined = tcn_combined.transpose(1, 2)  # Back to (batch, seq, features)
        tcn_features = self.tcn_fusion(tcn_combined)
        tcn_features = self.tcn_fusion_norm(tcn_features)

        # Multiple Temporal Resolutions
        # Original sequence for short-term patterns
        lstm_short_out, _ = self.lstm_short(x)
        lstm_short_out = self.lstm_short_norm(lstm_short_out)

        # Downsampled sequence for medium-term patterns
        x_medium = self.downsample_sequence(x, 2)
        lstm_medium_out, _ = self.lstm_medium(x_medium)

        # Upsample medium resolution back to original sequence length
        lstm_medium_out = F.interpolate(
            lstm_medium_out.transpose(1, 2).to('cpu'),
            size=seq_len,
            mode='linear'
        ).transpose(1, 2).to(self.device)
        lstm_medium_out = self.lstm_medium_norm(lstm_medium_out)

        # Combine temporal resolutions
        lstm_combined = torch.cat([lstm_short_out, lstm_medium_out], dim=2)
        lstm_features = self.temporal_fusion(lstm_combined)
        lstm_features = self.temporal_fusion_norm(lstm_features)

        # Add residual connection with projection if needed
        x_residual = self.residual_proj(x)
        lstm_features = lstm_features + x_residual

        # Combine LSTM and TCN features
        combined_features = torch.cat([lstm_features, tcn_features], dim=2)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.feature_fusion_norm(fused_features)

        # Add positional encoding for transformer
        fused_features = self.pos_encoder(fused_features)

        # Apply transformer encoder (replacing custom attention)
        attended_features = self.transformer_encoder(fused_features)

        # Final output layers with layer normalization
        x = F.relu(self.fc1(attended_features))
        x = self.fc1_norm(x)
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_norm(x)
        x = self.dropout_layer(x)
        outputs = self.regression_head(x)

        return outputs

    def train_model(self, train_loader, val_loader, test_loader, num_epochs=100, patience=10):
        """
        Train the model with a regression task
        """
        self.to(self.device)

        # Define loss function and optimizer with weight decay
        criterion = nn.MSELoss()
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        grad_scaler = GradScaler()

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
            'train_mape': [], 'val_mape': [], 'test_mape': []
        }

        epoch_progress = tqdm(range(num_epochs), desc="Training Epochs")
        for epoch in epoch_progress:
            # Training phase
            self.train()
            train_loss, train_rmse, train_mape = self._train_epoch(train_loader, criterion, optimizer, grad_scaler)
            # Validation phase
            val_loss, val_rmse, val_mape = self.evaluate(val_loader, criterion, grad_scaler)
            # Test phase
            test_loss, test_rmse, test_mape = self.evaluate(test_loader, criterion, grad_scaler)
            # Update learning rate
            scheduler.step(val_loss)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['test_rmse'].append(test_rmse)
            self.history['train_mape'].append(train_mape)
            self.history['val_mape'].append(val_mape)
            self.history['test_mape'].append(test_mape)

            # Update progress
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Test Loss': f'{test_loss:.4f}',
                'RMSE': f'{test_rmse:.4f}',
                'MAPE': f'{test_mape:.2f}%'
            })

            # Model selection and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Load the best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

            # Final evaluation
        final_test_loss, final_test_rmse, final_test_mape = self.evaluate(test_loader, criterion)
        print(
            f"\nFinal Test Results | Loss: {final_test_loss:.4f}, RMSE: {final_test_rmse:.4f}, MAPE: {final_test_mape:.2f}%")

        return self.history

    def _train_epoch(self, train_loader, criterion, optimizer, grad_scaler=None):
        """Helper method for training a single epoch"""
        self.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            # Forward pass
            with autocast():
                outputs = self(inputs)
                # Squeeze the last dimension if it exists
                if outputs.dim() > 1:
                    outputs = outputs[:, -1, 0] if outputs.size(1) > 1 and outputs.size(2) > 0 else outputs.squeeze()
                loss = criterion(outputs, targets)

            # Backward pass and optimize
            grad_scaler.scale(loss).backward()

            # Gradient clipping
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_size)

            grad_scaler.step(optimizer)
            grad_scaler.update()

            total_loss += loss.item() * inputs.size(0)
            all_predictions.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        avg_loss = total_loss / len(train_loader.dataset)
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        # Improved MAPE calculation to handle near-zero values
        epsilon = 1e-6  # Larger epsilon for numerical stability
        abs_percentage_errors = np.abs((all_targets - all_predictions) / np.maximum(np.abs(all_targets), epsilon))
        # Clip extreme values
        abs_percentage_errors = np.clip(abs_percentage_errors, 0, 10)  # Cap at 1000%
        mape = np.mean(abs_percentage_errors) * 100

        return avg_loss, rmse, mape

    def evaluate(self, data_loader, criterion, grad_scaler=None):
        """Evaluate the model with improved metrics"""
        self.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            with autocast():
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self(inputs)
                    # Squeeze the last dimension if it exists
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

        # Improved MAPE calculation
        epsilon = 1e-6
        abs_percentage_errors = np.abs((all_targets - all_predictions) / np.maximum(np.abs(all_targets), epsilon))
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
            fig.add_trace(go.Scatter(y=self.history['train_mape'], name='Train MAPE', line=dict(color='blue')),
                          row=1, col=2)
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
        try:
            self.eval()
            # Fetch a sample input tensor from DataLoader
            example_inputs, _ = next(iter(data_loader))

            # Export model to TorchScript using tracing
            scripted_model = torch.jit.trace(self.to(device), example_inputs.to(device))

            # Save the TorchScript model
            torch.jit.save(scripted_model, save_path)

            print(f"Model successfully exported and saved to {save_path}")
            return save_path

        except Exception as e:
            print(f"Error exporting model to TorchScript: {str(e)}")
            return None

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

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

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class DataModule:
    def __init__(self, data, window_size=20, batch_size=32, val_size=0.1, test_size=0.1,
                 random_state=42, target_col='shifted_prices', padding_strategy='zero'):
        if 'Ticker' in data.columns:
            self.data = data.drop(columns=['Ticker'])
        elif data.columns[0] == 'Ticker':
            # Drop the first column by index
            self.data = data.iloc[:, 1:]
        else:
            self.data = data

        self.window_size = window_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.target_col = target_col
        self.padding_strategy = padding_strategy

        self.setup()

    def setup(self):
        # Get datetime index from the data
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex for year-based splitting")

        # Ensure data is sorted chronologically
        length = len(self.data)
        # Calculate the number of days in each split
        test_days = int(length * self.test_size)
        val_days = int(length * self.val_size)
        train_days = length - test_days - val_days

        # Split the data chronologically
        self.df_train = self.data[:train_days]
        self.df_val = self.data[train_days:train_days + val_days]
        self.df_test = self.data[train_days + val_days:]

        # Create datasets and data loaders
        feature_cols = [col for col in self.data.columns if col != self.target_col]
        self.num_features = len(feature_cols)

         # Handle infinite values before scaling
        self.df_train[feature_cols] = self.df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
        self.df_val[feature_cols] = self.df_val[feature_cols].replace([np.inf, -np.inf], np.nan)
        self.df_test[feature_cols] = self.df_test[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with appropriate method (median is generally robust)
        for col in feature_cols:
            median_val = self.df_train[col].median()
            self.df_train[col] = self.df_train[col].fillna(median_val)
            self.df_val[col] = self.df_val[col].fillna(median_val)
            self.df_test[col] = self.df_test[col].fillna(median_val)

        # Data Scaler
        self.scaler = StandardScaler()
        train_features = self.df_train[feature_cols]
        self.scaler.fit(train_features)

        # Create datasets with an improved padding strategy
        self.train_dataset = SequenceDataset(
            self.df_train, target=self.target_col, features=feature_cols,
            window_size=self.window_size, padding_strategy=self.padding_strategy
        )
        self.val_dataset = SequenceDataset(
            self.df_val, target=self.target_col, features=feature_cols,
            window_size=self.window_size, padding_strategy=self.padding_strategy
        )
        self.test_dataset = SequenceDataset(
            self.df_test, target=self.target_col, features=feature_cols,
            window_size=self.window_size, padding_strategy=self.padding_strategy
        )

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, window_size, padding_strategy='zero'):
        self.features = features
        self.target = target
        self.window_size = window_size
        self.padding_strategy = padding_strategy
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.window_size - 1:
            i_start = i - self.window_size + 1
            x = self.X[i_start:(i + 1), :]
        else:
            # Improved padding strategy
            if self.padding_strategy == 'zero':
                # Zero padding
                padding = torch.zeros(self.window_size - i - 1, self.X.shape[1])
            elif self.padding_strategy == 'mean':
                # Mean padding
                padding = torch.mean(self.X[:i+1], dim=0).unsqueeze(0).repeat(self.window_size - i - 1, 1)
            elif self.padding_strategy == 'repeat':
                # Repeat first value (original strategy)
                padding = self.X[0].repeat(self.window_size - i - 1, 1)
            else:
                # Default to zero padding
                padding = torch.zeros(self.window_size - i - 1, self.X.shape[1])

            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


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