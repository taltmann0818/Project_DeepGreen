from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import pandas as pd

class DataModule:
    def __init__(self, data, seq_length=10, batch_size=32, eval_size=0.2, random_state=42,volatility_window=20):
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

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.random_state = random_state
        self.volatility_window = volatility_window

        self.setup()

    def calculate_volatility(self, close_prices):
        # Calculate returns
        returns = np.log(close_prices).diff().fillna(0)

        # Calculate rolling standard deviation
        volatility = returns.rolling(window=self.volatility_window).std().fillna(method='bfill')
        # Normalize volatility to [0, 1] range for easier training
        min_vol = volatility.min()
        max_vol = volatility.max()
        normalized_volatility = (volatility - min_vol) / (max_vol - min_vol)

        return normalized_volatility

    def setup(self):
        # Create sequences from the raw data
        self.data['Volatility'] = self.calculate_volatility(self.data['Close'])
        X, y, vol = self.create_sequences(self.data, self.seq_length)

        # Split into train, validation, and test sets
        total_samples = len(X)
        eval_samples = int(total_samples * self.eval_size)
        train_samples = total_samples - eval_samples

        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        indices = np.random.permutation(total_samples)

        train_indices = indices[:train_samples]
        eval_indices = indices[train_samples:train_samples + eval_samples]
        test_indices = indices[train_samples + eval_samples:]

        # Split data
        X_train, y_train, vol_train = X[train_indices], y[train_indices], vol[train_indices]
        X_eval, y_eval, vol_eval = X[eval_indices], y[eval_indices], vol[eval_indices]

        # Create datasets
        self.train_dataset = ModelDataset(X_train, y_train, vol_train)
        self.eval_dataset = ModelDataset(X_eval, y_eval, vol_eval)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )


    def create_sequences(self, data, seq_length):
        """
        Create overlapping sequences from the data.
        Assumes 'data' supports .iloc (e.g. a pandas DataFrame).
        """
        xs, ys, vols = [], [], []

        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)]
            y = data.iloc[i + seq_length]['Target']
            vol = data.iloc[i + seq_length]['Volatility']
            xs.append(x)
            ys.append(y)
            vols.append(vol)
        return np.array(xs), np.array(ys), np.array(vols)

class ModelDataset(Dataset):
    """Dataset that provides both direction and volatility targets"""

    def __init__(self, data, labels, volatility):
        """
        Args:
            data: Feature sequences
            labels: Direction labels
            volatility: Volatility targets
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.volatility = torch.FloatTensor(volatility)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        features = self.data[idx]
        direction_target = self.labels[idx]
        volatility_target = self.volatility[idx]

        # Return features and a dictionary of targets
        targets = {
            'direction': direction_target,
            'volatility': volatility_target
        }

        return features, targets

# Example usage:
# Assume `merged_df` is a pandas DataFrame containing your data.
# data_module = DataModule(merged_df, seq_length=10, batch_size=32)
# train_loader = data_module.train_loader
# eval_loader = data_module.eval_loader
# test_loader = data_module.test_loader
