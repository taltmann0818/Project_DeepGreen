import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TEMPUS(nn.Module):
    """
    The TEMPUS model for time-series data processing and prediction tasks.

    TEMPUS implements a hybrid deep learning architecture
    that combines LSTM at multiple temporal resolutions, Temporal Convolution Networks (TCN), feature fusion, Temporal Attention (TA), and fully connected layers for regression tasks. It is designed
    to handle sequential data effectively by capturing temporal
    relationships and high-level data representations.

    """
    #def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,tcn_kernel_sizes=[3, 5, 7], attention_heads=4, device="cpu"):
    def __init__(self, config, scaler=None):
        super(TEMPUS, self).__init__()
        self.device = config["device"] #device
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"] #num_layers
        self.input_size = config["input_size"]
        self.dropout = config["dropout"] #dropout
        self.clip_size = config["clip_size"] #tcn_kernel_sizes
        self.tcn_kernel_sizes = [3, 5, 7] #tcn_kernel_sizes

        if scaler is not None:
            self.scaler = scaler
            self.register_buffer("mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scaler.scale_, dtype=torch.float32))

        # Multiple Temporal Resolutions of LSTM
        self.lstm_short = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True, dropout=self.dropout if self.num_layers > 1 else 0, bidirectional=True)
        self.lstm_medium = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=self.dropout if self.num_layers > 1 else 0, bidirectional=True)
        self.temporal_fusion = nn.Linear(self.hidden_size * 4, self.hidden_size * 2) # Fusion layer for temporal resolutions

        # Temporal Convolutional Network (TCN)
        self.tcn_modules = nn.ModuleList()
        for k_size in self.tcn_kernel_sizes:
            padding = (k_size - 1) // 2  # Same padding
            self.tcn_modules.append(nn.Sequential(
                nn.Conv1d(self.input_size, self.hidden_size, kernel_size=k_size, padding=padding),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=k_size, padding=padding),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU()
            ))
        self.tcn_fusion = nn.Linear(self.hidden_size * len(self.tcn_kernel_sizes), self.hidden_size * 2)

        # Combine TCN and LSTM features
        self.feature_fusion = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)

        # Temporal attention
        #self.temporal_attention = TemporalAttention(
        #    d_model=self.hidden_size * 2,
        #    nhead=4,
        #    dropout=self.dropout
        #)
        self.temporal_attention = TemporalAttention(self.hidden_size * 2)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.regression_head = nn.Linear(self.hidden_size // 2, 1)
        self.dropout = nn.Dropout(self.dropout)

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
        time_features = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1).to(x.device)

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

        # Multiple Temporal Resolutions
        # Original sequence for short-term patterns
        lstm_short_out, _ = self.lstm_short(x)

        # Downsampled sequence for medium-term patterns (every 2 time steps)
        x_medium = self.downsample_sequence(x, 2)
        lstm_medium_out, _ = self.lstm_medium(x_medium)

        # Upsample medium resolution back to original sequence length
        lstm_medium_out = F.interpolate(
            lstm_medium_out.transpose(1, 2),
            size=seq_len,
            mode='linear'
        ).transpose(1, 2)

        # Combine temporal resolutions
        lstm_combined = torch.cat([lstm_short_out, lstm_medium_out], dim=2)
        lstm_features = self.temporal_fusion(lstm_combined)

        # 2. Add residual connection
        if features == lstm_features.size(2):  # If dimensions match
            lstm_features = lstm_features + x

        # Combine LSTM and TCN features
        combined_features = torch.cat([lstm_features, tcn_features], dim=2)
        fused_features = self.feature_fusion(combined_features)

        # Apply temporal attention
        #attended_features = self.temporal_attention(fused_features)
        attended_features, _ = self.temporal_attention(fused_features, time_features)

        # Global pooling (average) across time dimension to get single feature vector per sequence
        #pooled_features = torch.mean(attended_features, dim=1)

        # Final output layers
        #x = F.relu(self.fc1(pooled_features))
        x = F.relu(self.fc1(attended_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        outputs = self.regression_head(x)

        return outputs

    def train_model(self, train_loader, test_loader, criterion, optimizer, num_epochs=10):
        """
        Train the model with a regression task
        """
        self.to(self.device)

        best_test_mape = float('inf')
        best_model_state = None

        self.history = {
            'train_loss': [], 'test_loss': [],
            'rmse': [], 'mape': []
        }

        epoch_progress = tqdm(range(num_epochs), desc="Training Epochs")
        for epoch in epoch_progress:
            self.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs).to(self.device).squeeze()
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_size)
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # Evaluate on validation set
            test_loss, test_rmse, test_mape = self.evaluate(test_loader, criterion)
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Test Loss': f'{test_loss:.4f}',
                'RMSE': f'{test_rmse:.4f}',
                'MAPE': f'{test_mape:.2f}%'
            })
            #print(f"Epoch {epoch}/{num_epochs}; Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, RMSE: {test_rmse:.2f}%")

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['rmse'].append(test_rmse)
            self.history['mape'].append(test_mape)

            # Store model state with best val mape
            if test_mape < best_test_mape:
                best_test_mape = test_mape
                self.best_model_state = self.state_dict()

        # Load the best model state before returning
        if self.best_model_state is not None:
            self.load_state_dict(self.best_model_state)

        # Final evaluation with best model state
        print(f"\nBest MAPE: {best_test_mape:.2f}%")

        return self.history

    def evaluate(self, data_loader, criterion):
        self.to(self.device)
        self.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs).to(self.device).squeeze()
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate RMSE and MAPE
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        # Avoid division by zero by adding a small epsilon
        mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
        avg_loss = total_loss / len(data_loader.dataset)

        return avg_loss, rmse, mape

    def plot_training_history(self):
        if self.history is not None:
            # Create subplots for loss and accuracy
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss Over Epochs',
                                                                'MAPE Over Epochs'))

            # Plot losses
            fig.add_trace(go.Scatter(y=self.history['train_loss'], name='Train Loss', line=dict(color='blue')),
                          row=1, col=1)
            fig.add_trace(go.Scatter(y=self.history['test_loss'], name='Test Loss', line=dict(color='orange')),
                          row=1, col=1)
            fig.update_xaxes(title_text="Epochs", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)

            # Plot MAPE
            fig.add_trace(go.Scatter(y=self.history['mape'], name='Test MAPE', line=dict(color='orange')),
                          row=1, col=2)
            fig.update_xaxes(title_text="Epochs", row=1, col=2)
            fig.update_yaxes(title_text="MAPE %", row=1, col=2)

            fig.update_layout(
                title='Model Training Metrics',
                height=700,
                #template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            return fig

        else:
            print("Training history not available. Please run train_model() first.")
            return

    def export_model_to_torchscript(self, save_path, data_loader, device):
        try:
            # Set model to evaluation mode
            self.eval()
    
            # Fetch a sample input tensor from DataLoader
            example_inputs, _ = next(iter(data_loader))
            example_inputs = example_inputs.to(device)
    
            # Export model to TorchScript using tracing
            scripted_model = torch.jit.trace(self.to(device), example_inputs)
    
            # Save the TorchScript model
            torch.jit.save(scripted_model, save_path)
    
            print(f"Model successfully exported and saved to {save_path}")
            return save_path
    
        except Exception as e:
            print(f"Error exporting model to TorchScript: {str(e)}")
            return None

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

class DONTUSETemporalAttentionv2(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(DONTUSETemporalAttentionv2, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        attn_output, _ = self.self_attn(x, x, x)
        out = self.norm(x + self.dropout(attn_output))
        return out

class BaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.0):
        # Define device
        self.history = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        super(BaseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout_rate = dropout
        self.num_classes = num_classes

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=True
        )

        # Self-attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
                        or possibly with an extra singleton dimension.

        Returns:
            Tensor: Output logits for each class.
        """
        # If input is 4D with a singleton dimension, squeeze it.
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)

        # Initialize hidden and cell states for LSTM (accounting for bidirectionality)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)

        # Get LSTM outputs; out shape: (batch_size, sequence_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        attention_weights = self.attention(out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)

        # Fully connected layers with dropout applied between layers
        out = self.fc1(context_vector)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

def torchscript_predict(model_path, input_df, device, window_size, target_col):
    # Load the TorchScript model
    loaded_model = torch.jit.load(model_path,map_location=device)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    predictions = []
    dates = []
    actuals = []
    tickers = []
    uncertainties = []

    for i in range(window_size, len(input_df)):
        # Get date, actual value, and ticker for the current index
        date = input_df.index[i]
        actual = input_df[target_col].iloc[i] if target_col in input_df.columns else None
        ticker = input_df['Ticker'].iloc[i] if 'Ticker' in input_df.columns else None  
        
        # Get sequence
        values = input_df.drop(columns=['Ticker',target_col]).values.astype(np.float32)
        input_window = values[i - window_size:i]
        input_tensor = torch.tensor(input_window, dtype=torch.float32, device=device).unsqueeze(0)

        # Predict the next value based on the previous window
        pred = loaded_model(input_tensor)
        pred_value = pred.detach().numpy().flatten()[0]

        predictions.append(pred_value)
        dates.append(date)
        actuals.append(actual)
        tickers.append(ticker)

    # Create DataFrame with predictions
    preds_df = pd.DataFrame({
        'Ticker': tickers,
        'Actual': actuals,
        'Predicted': predictions
    },index=dates)

    return preds_df

class DataModule:
    def __init__(self, data, window_size=20, batch_size=32, eval_size=0.2, random_state=42):
        """
        data: pandas DataFrame or similar data structure with .iloc
        seq_length: length of the sequence window
        batch_size: batch size for DataLoaders
        test_size: proportion of the data to hold out in the first split
        eval_size: proportion of the holdout to be used as evaluation set (the rest becomes test)
        random_state: seed for reproducibility
        """
        # IF ticker columns exists in data then drop it
        if 'Ticker' in data.columns:
            self.data = data.drop(columns=['Ticker'])
        elif data.columns[0] == 'Ticker':
            # Drop the first column by index
            self.data = data.iloc[:, 1:]
        else:
            self.data = data

        self.window_size = window_size
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.random_state = random_state

        self.setup()

    def setup(self):
        # Instead of a random split, use a sequential split.
        train_end = int((1 - self.eval_size) * len(self.data))
        self.df_train = self.data.iloc[:train_end].copy()
        self.df_test = self.data.iloc[train_end:].copy()

        # Create datasets and data loaders
        target_col = 'shifted_prices'
        feature_cols = [col for col in self.data.columns if col != target_col]
        self.num_features = len(feature_cols)

        # Data Scaler
        self.scaler = StandardScaler()
        train_features = self.df_train[feature_cols]
        self.scaler.fit(train_features)

        self.train_dataset = SequenceDataset(self.df_train,target=target_col,features=feature_cols,window_size=self.window_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = SequenceDataset(self.df_test,target=target_col,features=feature_cols,window_size=self.window_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, window_size):
        self.features = features
        self.target = target
        self.window_size = window_size
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.window_size - 1:
            i_start = i - self.window_size + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.window_size - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]