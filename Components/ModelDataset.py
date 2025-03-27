from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import pandas as pd

class DataModule:
    def __init__(self, data, seq_length=10, batch_size=32, test_size=0.2, eval_size=0.5, random_state=42):
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
        self.test_size = test_size
        self.eval_size = eval_size
        self.random_state = random_state
        
        self.setup()
    
    def setup(self):
        # Create sequences from the raw data
        self.X, self.y = self.create_sequences(self.data, self.seq_length)

        # Apply MinMaxScaler to self.X
        n_samples, seq_length, n_features = self.X.shape # reshape X from (n_samples, seq_length, n_features) to (-1, n_features)
        self.X = self.X.reshape(-1, n_features)
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(self.X).astype(np.float32)
        self.X = self.X.reshape(n_samples, seq_length, n_features) # Reshape back to original shape
        
        # Split the data into training and temporary sets
        target_column = self.data.columns[-1]

        train_x, temp_x, train_y, temp_y = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        # Split the temporary set into evaluation and test sets
        eval_x, test_x, eval_y, test_y = train_test_split(
            temp_x, temp_y, test_size=self.eval_size, random_state=self.random_state
        )
        
        # Create Dataset objects for each split
        self.train_dataset = self.ModelDataset(train_x, train_y)
        self.eval_dataset = self.ModelDataset(eval_x, eval_y)
        self.test_dataset = self.ModelDataset(test_x, test_y)
        
        # Create DataLoaders for each split
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def create_sequences(self, data, seq_length):
        """
        Create overlapping sequences from the data.
        Assumes 'data' supports .iloc (e.g. a pandas DataFrame).
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)]
            y = data.iloc[i + seq_length]['Target']
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    class ModelDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.tensor(np.array(data), dtype=torch.float32)
            self.labels = torch.tensor(np.array(labels), dtype=torch.long)

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index], self.labels[index]
            
# Example usage:
# Assume `merged_df` is a pandas DataFrame containing your data.
# data_module = DataModule(merged_df, seq_length=10, batch_size=32)
# train_loader = data_module.train_loader
# eval_loader = data_module.eval_loader
# test_loader = data_module.test_loader
