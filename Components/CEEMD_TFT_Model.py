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

        Args:
            series (numpy.ndarray or pandas.Series): The time series to decompose

        Returns:
            numpy.ndarray: Array of shape (n_imfs, n_samples) containing the IMFs
        """
        if isinstance(series, pd.Series):
            series = series.values

        # Normalize the series to improve decomposition stability
        series = (series - np.mean(series)) / np.std(series)

        # Perform CEEMD decomposition
        imfs = self.ceemdan(series)

        # Limit the number of IMFs if needed
        if len(imfs) > self.max_imfs:
            imfs = imfs[:self.max_imfs]

        return imfs

    def batch_decompose(self, df, column='Close', group_col='Ticker'):
        """
        Apply CEEMD decomposition to grouped data in a DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame containing the time series data
            column (str): Column name of the time series to decompose
            group_col (str): Column name to group by (e.g., 'Ticker')

        Returns:
            pandas.DataFrame: DataFrame with original data and added IMF columns
        """
        result_df = df.copy()

        # Process each group separately
        for name, group in df.groupby(group_col):
            series = group[column].values
            imfs = self.decompose(series)

            # Add IMFs as new columns
            for i, imf in enumerate(imfs):
                result_df.loc[group.index, f'IMF_{i+1}'] = imf

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
        imfs = x.get('imfs')

        batch_size, time_steps, _ = time_varying_real.shape

        # Process static inputs if available
        if static is not None and self.static_dim > 0:
            # Split static inputs for variable selection
            static_inputs = [static[:, i].unsqueeze(-1) for i in range(self.static_dim)]
            static_embedding, static_weights = self.static_vsn(static_inputs)

            # Create static context vectors
            static_context_variable_selection = self.static_context_variable_selection(static_embedding)
            static_context_enrichment = self.static_context_enrichment(static_embedding)
            static_context_state_h = self.static_context_state_h(static_embedding)
            static_context_state_c = self.static_context_state_c(static_embedding)
        else:
            static_context_variable_selection = None
            static_context_enrichment = None
            static_context_state_h = torch.zeros((batch_size, self.hidden_size), device=self.device)
            static_context_state_c = torch.zeros((batch_size, self.hidden_size), device=self.device)

        # Process encoder inputs (past inputs)
        encoder_inputs = []

        # Add time-varying categorical inputs if available
        if time_varying_categorical is not None and self.time_varying_categorical_dim > 0:
            for i in range(self.time_varying_categorical_dim):
                encoder_inputs.append(time_varying_categorical[:, :, i].unsqueeze(-1))

        # Add time-varying real inputs
        for i in range(self.time_varying_real_dim):
            encoder_inputs.append(time_varying_real[:, :, i].unsqueeze(-1))

        # Add IMFs from CEEMD
        for i in range(self.num_imfs):
            encoder_inputs.append(imfs[:, :, i].unsqueeze(-1))

        # Reshape for variable selection network
        encoder_inputs_reshape = []
        for i in range(len(encoder_inputs)):
            # Reshape to [batch_size * time_steps, 1]
            reshaped = encoder_inputs[i].reshape(batch_size * time_steps, 1)
            encoder_inputs_reshape.append(reshaped)

        # Apply variable selection with static context if available
        if static_context_variable_selection is not None:
            # Repeat static context for each time step
            static_context = static_context_variable_selection.unsqueeze(1).repeat(1, time_steps, 1)
            static_context = static_context.reshape(batch_size * time_steps, -1)

            # Add static context to variable selection
            selected_encoder, encoder_weights = self.encoder_vsn(encoder_inputs_reshape)
        else:
            selected_encoder, encoder_weights = self.encoder_vsn(encoder_inputs_reshape)

        # Reshape back to [batch_size, time_steps, hidden_size]
        selected_encoder = selected_encoder.reshape(batch_size, time_steps, self.hidden_size)

        # Add positional encoding
        selected_encoder = self.pos_encoder(selected_encoder)

        # LSTM encoder
        lstm_input = selected_encoder

        # Initialize LSTM state with static context if available
        if static_context_state_h is not None and static_context_state_c is not None:
            h0 = static_context_state_h.unsqueeze(0).repeat(self.lstm_layers, 1, 1)
            c0 = static_context_state_c.unsqueeze(0).repeat(self.lstm_layers, 1, 1)
            lstm_output, _ = self.lstm_encoder(lstm_input, (h0, c0))
        else:
            lstm_output, _ = self.lstm_encoder(lstm_input)

        # Apply gated skip connection
        temporal_features = self.post_lstm_gate_encoder(lstm_output)

        # Prepare for attention
        query = self.attention_norm(temporal_features)
        key = self.attention_norm(temporal_features)
        value = temporal_features

        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(query, key, value)

        # Apply position-wise feed-forward network
        outputs = self.pos_wise_ff(attn_output)

        # Get the last time step for forecasting
        last_step = outputs[:, -1]

        # Generate forecasts for multiple horizons
        forecasts = self.forecast_layer(last_step)

        return forecasts

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
        imfs = torch.tensor(window_data[[f'IMF_{i+1}' for i in range(self.num_imfs)]].values, dtype=torch.float32)
        inputs['imfs'] = imfs

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
            patience=5,
            verbose=True
        )

        # Initialize loss function
        self.dir_alpha = config.get("dir_alpha", 0.2)  # Weight for directional component
        self.target_accuracy = config.get("target_accuracy", 0.55)  # Target directional accuracy
        self.criterion = DirectionalMSELoss(alpha=self.dir_alpha, target_accuracy=self.target_accuracy)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_dir_acc': [],  # Directional accuracy for training
            'val_dir_acc': [],    # Directional accuracy for validation
            'test_dir_acc': []    # Directional accuracy for testing
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
        decomposed_data = self.decomposer.batch_decompose(data, column=target_col)

        # Scale the data
        scaler = StandardScaler()

        # Identify columns to scale (excluding categorical and target)
        cols_to_scale = [col for col in decomposed_data.columns if col not in 
                         (categorical_cols or []) + [target_col]]

        # Fit and transform
        decomposed_data[cols_to_scale] = scaler.fit_transform(decomposed_data[cols_to_scale])

        return decomposed_data, scaler

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

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
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
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            all_train_predictions = []
            all_train_targets = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Move data to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * targets.size(0)

                # Store predictions and targets for directional accuracy calculation
                all_train_predictions.append(outputs.detach().cpu())
                all_train_targets.append(targets.cpu())

            # Calculate average training loss
            train_loss /= len(train_loader.dataset)

            # Calculate training directional accuracy
            train_predictions = torch.cat(all_train_predictions, dim=0)
            train_targets = torch.cat(all_train_targets, dim=0)
            train_dir_acc = self.calculate_directional_accuracy(train_predictions, train_targets)

            # Validation phase
            val_loss, val_dir_acc = self.evaluate(val_loader)

            # Test phase
            test_loss, test_dir_acc = self.evaluate(test_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_dir_acc'].append(train_dir_acc)
            self.history['val_dir_acc'].append(val_dir_acc)
            self.history['test_dir_acc'].append(test_dir_acc)

            # Print progress
            print(f'Epoch {epoch+1}/{self.num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Test Loss: {test_loss:.4f} | '
                  f'Dir Acc: {train_dir_acc:.2f}%/{val_dir_acc:.2f}%/{test_dir_acc:.2f}%')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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

        # Calculate directional accuracy
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        dir_accuracy = self.calculate_directional_accuracy(predictions, targets)

        return avg_loss, dir_accuracy

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
