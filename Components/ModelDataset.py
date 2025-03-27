from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import pandas as pd

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
        # Create sequences from the raw data
        X, y = self.create_sequences()

        # Instead of a random split, use a sequential split.
        train_size = int((1 - self.eval_size) * len(X)) # Calculate the index where the evaluation set should begin.
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Fit the scaler on just the training features to prevent leakage
        n_samples, seq_length, n_features = X.shape
        scaler = StandardScaler()
        X_train = X[:train_size].reshape(-1, n_features)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train).reshape(train_size, seq_length, n_features)
        X_test = scaler.transform(X[train_size:].reshape(-1, n_features)).reshape(len(X)-train_size, seq_length, n_features)
                    
        # Create datasets and data loaders
        self.train_dataset = ModelDataset(X_train, y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = ModelDataset(X_test, y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def create_sequences(self):
        """
        Split time-series into training sequence X and outcome value Y
        Args:
            data - dataset
            N - window size, e.g., 50 for 50 days of historical stock prices
            offset - position to start the split
        """
        X, y = [], []

        for i in range(self.window_size, len(self.data)):
            X.append(self.data.iloc[i - self.window_size : i])
            y.append(self.data.iloc[i]['shifted_prices'])
    
        return np.array(X), np.array(y)

class ModelDataset(Dataset):
    """Dataset that provides direction targets"""

    def __init__(self, data, shifted_prices):
        """
        Args:
            data: Feature sequences
            target: shifted equity prices
        """
        self.data = torch.FloatTensor(data)
        self.shifted_prices = torch.FloatTensor(shifted_prices)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        features = self.data[idx]
        targets = {
            'shifted_prices': self.shifted_prices[idx],
        }
        
        # Return features and a dictionary of targets
        return features, targets
