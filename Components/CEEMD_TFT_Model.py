import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import math
import copy
import PyEMD
from PyEMD import CEEMDAN
from torch.cuda.amp import GradScaler, autocast

class DirectionalMSELoss(nn.Module):
    """
    Custom loss function that combines MSE loss with a directional accuracy component.

    This loss function encourages the model to correctly predict the sign of returns
    by adding a penalty term when the predicted direction differs from the actual direction.

    Args:
        alpha (float): Weight for the directional component (0 <= alpha <= 1)
        target_accuracy (float): Target directional accuracy (0 <= target_accuracy <= 1)
    """
    def __init__(self, alpha=0.2, target_accuracy=0.55):
        super(DirectionalMSELoss, self).__init__()
        self.alpha = alpha
        self.target_accuracy = target_accuracy
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets):
        """
        Calculate the combined loss.

        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Combined loss value
        """
        # Calculate standard MSE loss
        mse_loss = self.mse(predictions, targets)

        # For multi-horizon forecasts, calculate directional accuracy across all horizons
        if predictions.dim() > 1:
            # Calculate signs of day-to-day changes in predictions and targets
            # For multi-day forecasts, we look at the direction between consecutive days
            pred_signs = torch.sign(predictions[:, 1:] - predictions[:, :-1])
            target_signs = torch.sign(targets[:, 1:] - targets[:, :-1])

            # Calculate directional accuracy (1 if signs match, 0 otherwise)
            correct_directions = (pred_signs == target_signs).float()

            # Calculate directional accuracy rate
            dir_accuracy = correct_directions.mean()

            # Directional loss: penalize when below target accuracy
            dir_loss = torch.max(torch.tensor(0.0, device=predictions.device), 
                                self.target_accuracy - dir_accuracy)

            # Combine losses
            combined_loss = (1 - self.alpha) * mse_loss.mean() + self.alpha * dir_loss

            return combined_loss
        else:
            # For single-value predictions, just return MSE loss
            return mse_loss.mean()

class CEEMD_Decomposer:
    """
    Class for performing CompleteEnsemble Empirical Mode Decomposition (CEEMD) on time series data.
    CEEMD decomposes a signal into multiple Intrinsic Mode Functions (IMFs) representing different frequency components.
    """
    def __init__(self, noise_std=0.05, trials=100, max_imfs=10):
        """
        Initialize the CEEMD decomposer.

        Args:
            noise_std (float): Standard deviation of the added noise
            trials (int): Number of trials/realizations for the ensemble
            max_imfs (int): Maximum number of IMFs to extract
        """
        self.noise_std = noise_std
        self.trials = trials
        self.max_imfs = max_imfs
        self.ceemdan = CEEMDAN(trials=trials)

    def decompose(self, series):
        """
        Decompose a time series into its IMFs using CEEMD.

        If the series is shorter than three samples CEEMDAN cannot work.  For
        such tiny inputs we simply return the (normalised) series itself as a
        single “IMF”, which is enough for the rest of the pipeline and avoids
        run-time failures.
        """
        if isinstance(series, pd.Series):
            series = series.values

        # Short series cannot be processed by CEEMDAN ─ return a safe fallback
        if len(series) < 3:
            return np.expand_dims(series.astype(float), axis=0)

        # Normalise to improve decomposition stability
        series = (series - np.mean(series)) / np.std(series)

        # Perform CEEMD decomposition
        imfs = self.ceemdan(series)

        # Limit the number of IMFs if needed
        if len(imfs) > self.max_imfs:
            imfs = imfs[: self.max_imfs]

        return imfs

    def batch_decompose(self, df, column="Close", group_col="Ticker"):
        """
        Apply CEEMD decomposition to grouped data in a DataFrame.
        """
        result_df = df.copy()

        # Prepare IMF columns
        for i in range(self.max_imfs):
            result_df[f"IMF_{i + 1}"] = np.nan

        # Process each group separately
        for name, group in df.groupby(group_col):
            series = group[column].values

            # Skip or fallback if not enough data
            if len(series) < 3:
                # Just fill zeros so that shapes stay consistent
                for i in range(self.max_imfs):
                    result_df.loc[group.index, f"IMF_{i + 1}"] = 0.0
                continue

            imfs = self.decompose(series)

            # Add IMFs to the output DataFrame, aligning by index
            for i, imf in enumerate(imfs):
                if i >= self.max_imfs:
                    break

                idx = group.index
                if len(imf) != len(idx):  # pad / trim if necessary
                    imf = np.pad(imf, (0, len(idx) - len(imf)),
                                 mode="constant", constant_values=np.nan)[: len(idx)]

                result_df.loc[idx, f"IMF_{i + 1}"] = pd.Series(imf, index=idx)

        # Fill any remaining NaNs produced by padding with zeros
        for i in range(self.max_imfs):
            result_df[f"IMF_{i + 1}"] = result_df[f"IMF_{i + 1}"].fillna(0.0)

        return result_df


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) as described by Lim et al. (2021) for a Temporal Fusion Transformer.
    GRN enables efficient information flow with skip connections and gating layers.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        # Linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()

        # Skip connection if input and output dimensions differ
        self.skip_connection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

        # Gating layer
        self.gate = nn.Linear(input_size, output_size)

        # Layer normalization
        self.norm = nn.LayerNorm(output_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # Main branch
        hidden = self.fc1(x)
        hidden = self.elu(hidden)
        hidden = self.dropout_layer(hidden)
        hidden = self.fc2(hidden)

        # Skip connection
        skip = self.skip_connection(x)

        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))

        # Combine with gating
        output = hidden * gate + skip * (1 - gate)

        # Apply normalization
        output = self.norm(output)

        return output

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) as described by Lim et al. (2021) for a Temporal Fusion Transformer.
    VSN enables judicious selection of the most salient features based on the input.
    """
    def __init__(self, input_sizes, hidden_size, output_size, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.output_size = output_size

        # GRNs for each variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                dropout=dropout
            ) for input_size in input_sizes
        ])

        # GRN for variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_size=sum(input_sizes),
            hidden_size=hidden_size,
            output_size=len(input_sizes),
            dropout=dropout
        )

    def forward(self, x_list):
        # Process each variable with its own GRN
        processed_vars = [grn(x) for grn, x in zip(self.variable_grns, x_list)]

        # Concatenate all inputs for the selection GRN
        x_concat = torch.cat(x_list, dim=-1)

        # Get variable selection weights
        weights = F.softmax(self.selection_grn(x_concat), dim=-1)

        # Apply weights to processed variables
        weighted_sum = torch.zeros_like(processed_vars[0])
        for i, processed_var in enumerate(processed_vars):
            weighted_sum += processed_var * weights[..., i].unsqueeze(-1)

        return weighted_sum, weights

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for multi-horizon time series forecasting.

    TFT combines high-performance multi-horizon forecasting with interpretable insights into 
    temporal dynamics. It uses variable selection, gated residual networks, and multi-head 
    attention for processing static metadata, time-varying past inputs, and time-varying 
    a priori known future inputs.
    """
    def __init__(self, config):
        super(TemporalFusionTransformer, self).__init__()

        # Model configuration
        self.device = config.get("device", "cpu")
        self.hidden_size = config.get("hidden_size", 128)
        self.lstm_layers = config.get("lstm_layers", 1)
        self.dropout = config.get("dropout", 0.1)
        self.num_heads = config.get("num_heads", 4)
        self.forecast_horizon = config.get("forecast_horizon", 3)  # 3 days forecast

        # Input dimensions
        self.static_dim = config.get("static_dim", 0)
        self.time_varying_categorical_dim = config.get("time_varying_categorical_dim", 0)
        self.time_varying_real_dim = config.get("time_varying_real_dim", 1)
        self.num_imfs = config.get("num_imfs", 5)  # Number of IMFs from CEEMD

        # Total input dimensions
        self.total_time_varying_dim = (
            self.time_varying_categorical_dim + 
            self.time_varying_real_dim + 
            self.num_imfs
        )

        # Variable selection networks
        if self.static_dim > 0:
            self.static_vsn = VariableSelectionNetwork(
                input_sizes=[1] * self.static_dim,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout=self.dropout
            )

        self.encoder_vsn = VariableSelectionNetwork(
            input_sizes=[1] * self.total_time_varying_dim,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        )

        self.decoder_vsn = VariableSelectionNetwork(
            input_sizes=[1] * self.total_time_varying_dim,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size
        )

        # Static context vectors
        if self.static_dim > 0:
            self.static_context_variable_selection = GatedResidualNetwork(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                output_size=self.total_time_varying_dim,
                dropout=self.dropout
            )

            self.static_context_enrichment = GatedResidualNetwork(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout=self.dropout
            )

            self.static_context_state_h = GatedResidualNetwork(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout=self.dropout
            )

            self.static_context_state_c = GatedResidualNetwork(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout=self.dropout
            )

        # LSTM layers for local processing
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True
        )

        # Gated skip connection
        self.post_lstm_gate_encoder = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        )

        # Layer normalization for attention
        self.attention_norm = nn.LayerNorm(self.hidden_size)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        # Position-wise feed-forward network
        self.pos_wise_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        )

        # Output layers for multi-horizon forecasting
        self.forecast_layer = nn.Linear(self.hidden_size, self.forecast_horizon)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)

    def forward(self, x):
        """
        Forward pass through the TFT model.

        Args:
            x (dict): Input dictionary containing:
                - 'static': Static features [batch_size, static_dim]
                - 'time_varying_categorical': Categorical features [batch_size, time_steps, cat_dim]
                - 'time_varying_real': Real-valued features [batch_size, time_steps, real_dim]
                - 'imfs': IMFs from CEEMD [batch_size, time_steps, num_imfs]

        Returns:
            torch.Tensor: Forecasts for the next 'forecast_horizon' steps [batch_size, forecast_horizon]
        """
        # Extract inputs
        static = x.get('static', None)
        time_varying_categorical = x.get('time_varying_categorical', None)
        time_varying_real = x.get('time_varying_real')
        #imfs = x.get('imfs')

        # Ensure inputs are tensors, not lists
        if isinstance(time_varying_categorical, list):
            time_varying_categorical = torch.stack(time_varying_categorical, dim=-1)
        if isinstance(time_varying_real, list):
            time_varying_real = torch.stack(time_varying_real, dim=-1)
        if static is not None and isinstance(static, list):
            static = torch.stack(static, dim=-1)

        # Combine time-varying inputs
        encoder_inputs = torch.cat([time_varying_categorical, time_varying_real], dim=-1)
        decoder_inputs = encoder_inputs[:, -self.forecast_horizon:, :]  # Use last forecast_horizon steps

        batch_size, time_steps, _ = encoder_inputs.shape
        decoder_time_steps = decoder_inputs.shape[1]

        # Static context variable selection
        static_context_variable_selection = None
        if static is not None:
            static_context_variable_selection, _ = self.static_vsn([static])

        # Static context enrichment
        static_context_enrichment = None
        if static_context_variable_selection is not None:
            static_context_enrichment = self.static_context_enrichment(static_context_variable_selection)

        # Encoder variable selection
        encoder_inputs_reshape = encoder_inputs.contiguous().view(batch_size * time_steps, -1)

        # Split the tensor into the original per-variable blocks expected by the VSN
        encoder_inputs_split = torch.split(
            encoder_inputs_reshape,
            self.encoder_vsn.input_sizes,  # list with the size of every variable block
            dim=-1
        )
        encoder_inputs_split = list(encoder_inputs_split)

        # Pass the list to the VSN
        selected_encoder, encoder_weights = self.encoder_vsn(encoder_inputs_split)

        # Reshape back
        selected_encoder = selected_encoder.view(batch_size, time_steps, -1)

        # Decoder variable selection
        decoder_inputs_reshape = decoder_inputs.contiguous().view(batch_size * decoder_time_steps, -1)

        # Split decoder inputs similarly
        decoder_inputs_split = torch.split(
            decoder_inputs_reshape,
            self.decoder_vsn.input_sizes,
            dim=-1
        )
        decoder_inputs_split = list(decoder_inputs_split)

        selected_decoder, decoder_weights = self.decoder_vsn(decoder_inputs_split)
        selected_decoder = selected_decoder.view(batch_size, decoder_time_steps, -1)

        # Static context state initialization for LSTM
        if static_context_enrichment is not None:
            c_enrichment = self.static_context_state_c(static_context_enrichment)
            h_enrichment = self.static_context_state_h(static_context_enrichment)
            c_enrichment = c_enrichment.view(self.lstm_layers, batch_size, self.hidden_size)
            h_enrichment = h_enrichment.view(self.lstm_layers, batch_size, self.hidden_size)
        else:
            c_enrichment = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=selected_encoder.device)
            h_enrichment = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=selected_encoder.device)

        # LSTM encoder
        lstm_out, (hidden_state, cell_state) = self.lstm_encoder(selected_encoder, (h_enrichment, c_enrichment))

        # Post-LSTM gating for encoder
        gated_lstm_out = self.post_lstm_gate_encoder(lstm_out)

        # Add positional encoding
        gated_lstm_out = self.pos_encoder(gated_lstm_out)

        # Self-attention
        # Use Flash Attention for better memory efficiency
        attn_out, attn_weights = self.multihead_attn(gated_lstm_out, gated_lstm_out, gated_lstm_out)

        # Layer normalization
        attn_out = self.attention_norm(attn_out + gated_lstm_out)  # Residual connection

        # Position-wise feed-forward
        ff_out = self.pos_wise_ff(attn_out)

        # Final forecast layer
        output = self.forecast_layer(ff_out[:, -1, :])  # Use last time step

        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
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
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data with CEEMD decomposition.
    """
    def __init__(self, data, window_size, forecast_horizon=3, target_col='Close', 
                 static_cols=None, categorical_cols=None, num_imfs=5):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.static_cols = static_cols or []
        self.categorical_cols = categorical_cols or []
        self.num_imfs = num_imfs

        # Identify real-valued columns (excluding target, static, categorical, and IMFs)
        self.real_cols = [col for col in data.columns if col not in 
                          [target_col] + self.static_cols + self.categorical_cols + 
                          [f'IMF_{i+1}' for i in range(num_imfs)]]

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # Extract window and forecast horizon
        window_data = self.data.iloc[idx:idx+self.window_size]
        forecast_data = self.data.iloc[idx+self.window_size:idx+self.window_size+self.forecast_horizon]

        # Prepare inputs
        inputs = {}

        # Static features (same for all time steps)
        if self.static_cols:
            static = torch.tensor(window_data[self.static_cols].iloc[0].values, dtype=torch.float32)
            inputs['static'] = static

        # Time-varying categorical features
        if self.categorical_cols:
            categorical = torch.tensor(window_data[self.categorical_cols].values, dtype=torch.float32)
            inputs['time_varying_categorical'] = categorical

        # Time-varying real features
        real = torch.tensor(window_data[self.real_cols].values, dtype=torch.float32)
        inputs['time_varying_real'] = real

        # IMFs from CEEMD
        #imfs = torch.tensor(window_data[[f'IMF_{i+1}' for i in range(self.num_imfs)]].values, dtype=torch.float32)
        #inputs['imfs'] = imfs

        # Target values for forecasting
        target = torch.tensor(forecast_data[self.target_col].values, dtype=torch.float32)

        return inputs, target

class CEEMD_TFT_Model:
    """
    Main class for the CEEMD-TFT model that combines CEEMD decomposition with 
    Temporal Fusion Transformer for multi-horizon forecasting.
    """
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cpu")
        self.window_size = config.get("window_size", 20)
        self.forecast_horizon = config.get("forecast_horizon", 3)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0001)
        self.num_imfs = config.get("num_imfs", 5)
        self.patience = config.get("patience", 10)
        self.num_epochs = config.get("num_epochs", 100)

        # Add AMP support for Ada Lovelace
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Enable TensorFloat-32 for faster training on Ada Lovelace
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Initialize CEEMD decomposer
        self.decomposer = CEEMD_Decomposer(
            noise_std=config.get("noise_std", 0.05),
            trials=config.get("trials", 100),
            max_imfs=self.num_imfs
        )

        # Initialize TFT model
        self.model = TemporalFusionTransformer(config)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Initialize loss function
        self.dir_alpha = config.get("dir_alpha", 0.2)  # Weight for directional component
        self.target_accuracy = config.get("target_accuracy", 0.80)  # Target directional accuracy
        self.criterion = DirectionalMSELoss(alpha=self.dir_alpha, target_accuracy=self.target_accuracy)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_dir_acc': [],  # Directional accuracy for training
            'val_dir_acc': [],    # Directional accuracy for validation
            'test_dir_acc': [],    # Directional accuracy for testing
            'test_rmse': [],  # Directional accuracy for testing
            'val_rmse': [],  # Directional accuracy for testing
            'test_mape': [],  # Directional accuracy for testing
            'val_mape': []  # Directional accuracy for testing
        }

    def preprocess_data(self, data, target_col='Close', static_cols=None, categorical_cols=None):
        """
        Preprocess data by applying CEEMD decomposition and scaling.

        Args:
            data (pandas.DataFrame): Input data
            target_col (str): Target column name
            static_cols (list): List of static column names
            categorical_cols (list): List of categorical column names

        Returns:
            tuple: Processed data, scaler
        """
        # Apply CEEMD decomposition
        #decomposed_data = self.decomposer.batch_decompose(data, column=target_col)
        data = data.drop(columns=['Ticker'])

        # Scale the data
        scaler = StandardScaler()

        # Identify columns to scale (excluding categorical and target)
        cols_to_scale = [col for col in data.columns if col not in
                         (categorical_cols or []) + [target_col]]

        # Fit and transform
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

        return data, scaler

    def create_datasets(self, data, val_size=0.15, test_size=0.15, target_col='Close', 
                        static_cols=None, categorical_cols=None):
        """
        Create train, validation, and test datasets.

        Args:
            data (pandas.DataFrame): Preprocessed data
            val_size (float): Validation set size ratio
            test_size (float): Test set size ratio
            target_col (str): Target column name
            static_cols (list): List of static column names
            categorical_cols (list): List of categorical column names

        Returns:
            tuple: Train, validation, and test DataLoaders
        """
        # Split data chronologically
        n = len(data)
        train_end = int(n * (1 - val_size - test_size))
        val_end = int(n * (1 - test_size))

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data, 
            self.window_size, 
            self.forecast_horizon, 
            target_col, 
            static_cols, 
            categorical_cols,
            self.num_imfs
        )

        val_dataset = TimeSeriesDataset(
            val_data, 
            self.window_size, 
            self.forecast_horizon, 
            target_col, 
            static_cols, 
            categorical_cols,
            self.num_imfs
        )

        test_dataset = TimeSeriesDataset(
            test_data, 
            self.window_size, 
            self.forecast_horizon, 
            target_col, 
            static_cols, 
            categorical_cols,
            self.num_imfs
        )

        # Optimized DataLoader settings for Ada Lovelace
        num_workers = min(8, torch.multiprocessing.cpu_count())
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader, test_loader):
        """
        Train the model.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader

        Returns:
            dict: Training history
        """
        # Early stopping variables
        best_val_mape = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            all_train_predictions = []
            all_train_targets = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                train_loss += loss.item() * targets.size(0)
                all_train_predictions.append(outputs.detach().cpu())
                all_train_targets.append(targets.cpu())

            # Calculate average training loss
            train_loss /= len(train_loader.dataset)

            # Calculate training directional accuracy
            train_predictions = torch.cat(all_train_predictions, dim=0)
            train_targets = torch.cat(all_train_targets, dim=0)
            train_dir_acc = self.calculate_directional_accuracy(train_predictions, train_targets)

            # Validation phase
            val_loss, val_dir_acc, val_rmse, val_mape = self.evaluate(val_loader)

            # Test phase
            test_loss, test_dir_acc, test_rmse, test_mape = self.evaluate(test_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_dir_acc'].append(train_dir_acc)
            self.history['val_dir_acc'].append(val_dir_acc)
            self.history['test_dir_acc'].append(test_dir_acc)
            self.history['val_rmse'].append(val_rmse)
            self.history['test_rmse'].append(test_rmse)
            self.history['val_mape'].append(val_mape)
            self.history['test_mape'].append(test_mape)

            # Print progress
            print(f'Epoch {epoch+1}/{self.num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Test Loss: {test_loss:.4f} | '
                  f'Dir Acc: {train_dir_acc:.2f}%/{val_dir_acc:.2f}%/{test_dir_acc:.2f}% | '
                  f'Val MAPE: {val_mape:.2f} | Test MAPE: {test_mape:.2f}')

            # Early stopping
            if val_mape < best_val_mape:
                best_val_mape = val_mape
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def calculate_directional_accuracy(self, predictions, targets):
        """
        Calculate directional accuracy between predictions and targets.

        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values

        Returns:
            float: Directional accuracy (percentage of correct direction predictions)
        """
        # For multi-horizon forecasts, calculate directional accuracy across all horizons
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Calculate signs of day-to-day changes in predictions and targets
            pred_signs = torch.sign(predictions[:, 1:] - predictions[:, :-1])
            target_signs = torch.sign(targets[:, 1:] - targets[:, :-1])

            # Calculate directional accuracy (1 if signs match, 0 otherwise)
            correct_directions = (pred_signs == target_signs).float()

            # Calculate directional accuracy rate
            dir_accuracy = correct_directions.mean().item()
        else:
            # For single-value predictions, return 0 (not applicable)
            dir_accuracy = 0.0

        return dir_accuracy * 100  # Convert to percentage

    def evaluate(self, data_loader):
        """
        Evaluate the model.

        Args:
            data_loader (DataLoader): Data loader for evaluation

        Returns:
            tuple: (Average loss, Directional accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                # Move data to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)

                # Store predictions and targets for directional accuracy calculation
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)

        # Calculate metrics
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        dir_accuracy = self.calculate_directional_accuracy(predictions, targets)

        rmse = np.sqrt(np.mean((predictions.numpy() - targets.numpy()) ** 2))

        # Improved MAPE calculation with epsilon to avoid division by zero
        epsilon = 1e-6
        abs_percentage_errors = np.clip(np.abs((targets.numpy() - predictions.numpy()) / np.maximum(np.abs(targets.numpy()), epsilon)), 0, 10)
        mape = np.mean(abs_percentage_errors) * 100

        return avg_loss, dir_accuracy, rmse, mape

    def predict(self, data_loader):
        """
        Make predictions with the model.

        Args:
            data_loader (DataLoader): Data loader for prediction

        Returns:
            tuple: Predictions and actual values
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                # Move data to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Store predictions and targets
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return predictions, targets

    def save_model(self, path):
        """
        Save the model.

        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path):
        """
        Load the model.

        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return self
