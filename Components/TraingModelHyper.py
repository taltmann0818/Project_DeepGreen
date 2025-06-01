import os
import math

import optuna.samplers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import Stopper
from ray.tune import Tuner, TuneConfig, RunConfig

from Components.TickerData import TickerData

# -- Model Definition --
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

class TEMPUS(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Hyperparameters
        self.device = config.get('device', 'cpu')
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.input_size = config['input_size']
        self.dropout = config['dropout']
        self.clip_size = config.get('clip_size', 1.0)
        self.tcn_kernel_sizes = [3, 5, 7]
        self.attention_heads = config['attention_heads']
        self.weight_decay = config['weight_decay']
        # Optional scaler
        self.scaler = None
        if 'scaler' in config and config['scaler'] is not None:
            sc = config['scaler']
            self.register_buffer('mean', torch.tensor(sc.mean_, dtype=torch.float32))
            self.register_buffer('scale', torch.tensor(sc.scale_, dtype=torch.float32))
            self.scaler = True

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
                nn.GELU(),  # Switching from ReLU to GELU
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=k_size,
                          padding=padding, dilation=dilation, stride=1),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU()  # Switching from ReLU to GELU
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
        # time_features = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1).to(x.device)

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

class DataModule:
    def __init__(self, data, target_col, window_size, batch_size, val_size, test_size):
        # Verify index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex for splitting")

        # Determine global split dates for walk-forward chronological splits
        all_dates = pd.Series(data.index.unique()).sort_values()
        n = len(all_dates)
        train_cut = int(n * (1 - val_size - test_size))
        val_cut = int(n * (1 - test_size))
        train_date = all_dates.iloc[train_cut]
        val_date = all_dates.iloc[val_cut]

        # Split per ticker using global date boundaries
        train_dfs, val_dfs, test_dfs = [], [], []
        for ticker, df in data.groupby('Ticker'):
            df = df.sort_index()
            train_dfs.append(df[df.index <= train_date])
            val_dfs.append(df[(df.index > train_date) & (df.index <= val_date)])
            test_dfs.append(df[df.index > val_date])
        # Concatenate splits
        self.df_train = pd.concat(train_dfs)
        self.df_val = pd.concat(val_dfs)
        self.df_test = pd.concat(test_dfs)

        # Determine feature columns (exclude target and ticker)
        feature_cols = [c for c in self.df_train.columns if c not in [target_col, 'Ticker']]
        self.num_features = len(feature_cols)

        # Fit and apply scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_train[feature_cols])
        for df in [self.df_train, self.df_val, self.df_test]:
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        # Build per-ticker SequenceDatasets to prevent sequence bleed
        self.train_dataset = ConcatDataset([SequenceDataset(df.drop(columns=['Ticker']), target_col, feature_cols, window_size) for df in train_dfs])
        self.val_dataset = ConcatDataset([SequenceDataset(df.drop(columns=['Ticker']), target_col, feature_cols, window_size) for df in val_dfs])
        self.test_dataset = ConcatDataset([SequenceDataset(df.drop(columns=['Ticker']),target_col, feature_cols, window_size) for df in test_dfs])

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

# -- Trainable function --
def train_tempus(config):
    # Create data module first
    dm = DataModule(config['dataframe'], config['target'], config['window_size'], config['batch_size'], config['val_ratio'], config['test_ratio'])
    # Ensure model input_size matches actual feature count
    config['input_size'] = dm.num_features
    device = 'cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu'
    config['device'] = device
    model = TEMPUS(config).to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(config['max_epochs']):
        # Training loop
        total_train = 0
        model.train()
        for X, y in dm.train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            if out.dim() > 1:
                out = out[:, -1, 0] if out.size(1) > 1 else out.squeeze()
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_size'])
            optimizer.step()
            total_train += loss.item() * y.size(0)
        train_loss = total_train / len(dm.train_dataset)
        # Validation loop
        total_val = 0
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in dm.val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                if out.dim() > 1:
                    out = out[:, -1, 0] if out.size(1) > 1 else out.squeeze()
                total_val += criterion(out, y).item() * y.size(0)
                preds.append(out.squeeze().cpu().numpy())
                targets.append(y.cpu().numpy())
        val_loss = total_val / len(dm.val_dataset)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        epsilon = 1e-6
        abs_percentage_errors = np.abs((targets - preds) / np.maximum(np.abs(targets), epsilon))
        abs_percentage_errors = np.clip(abs_percentage_errors, 0, 10)
        mape = np.mean(abs_percentage_errors) * 100

        scheduler.step(val_loss)
        tune.report(metrics={"train_loss": train_loss, "val_loss": val_loss, "val_rmse": rmse, "val_mape": mape})

class ThresholdStopper(Stopper):
    def __init__(self, metric, threshold):
        self.metric = metric
        self.threshold = threshold

    def __call__(self, trial_id, result):
        # Stop trial if metric > threshold on first iteration
        return (
            result.get("training_iteration", 0) == 1
            and result.get(self.metric, float("inf")) > self.threshold
        )

    def stop_all(self):
        # Never stop the entire experiment
        return False

# -- Hyperparameter tuning entrypoint --
def run_tuning(df, target='shifted_prices', num_samples=50, use_gpu=True):
    config = {
        'hidden_size': tune.choice([32, 64, 128]),
        'num_layers': tune.choice([1, 2]),
        'dropout': tune.uniform(0.1, 0.5),
        'clip_size': tune.choice([0.5, 1.0, 2.0]),
        'tcn_kernel_sizes': [3, 5, 7],
        'attention_heads': tune.choice([4, 8, 16]),
        'lr': tune.loguniform(1e-4, 1e-2),
        'weight_decay': tune.loguniform(1e-5, 1e-2),
        'batch_size': tune.choice([16, 32]),
        'window_size': tune.choice([20, 30]),
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'max_epochs': 10,
        'target': target,
        'dataframe': df,
        'use_gpu': use_gpu
    }

    search_alg = OptunaSearch(metric="val_mape",mode="min")
    scheduler = ASHAScheduler(time_attr='training_iteration', metric='val_mape', mode='min', max_t=100, grace_period=10,reduction_factor=2)
    stopper = ThresholdStopper(metric="val_mape", threshold=40.0)

    ray.shutdown()
    ray.init(num_cpus=6, num_gpus=1)
    tuner = Tuner(
        train_tempus,
        tune_config=TuneConfig(scheduler=scheduler,
                               search_alg=search_alg,
                               num_samples=num_samples,
                               trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id[:4]}"),
        run_config=RunConfig(name="tempus_experiment",stop=stopper),
        param_space=config,
    )
    results = tuner.fit()
    best = results.get_best_result(metric='val_mape', mode='min')
    print('Best config:', best.config)
    print('Best val_mape:', best.metrics['val_mape'])
    return best

if __name__ == '__main__':
    training_data = pd.read_csv('training_data.csv')
    training_data = training_data.set_index(pd.DatetimeIndex(training_data.index)).drop(columns=['date'])
    best = run_tuning(training_data)