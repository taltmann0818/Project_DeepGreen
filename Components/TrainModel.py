import contextlib

import torch
import torch.nn as nn
import shap
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TEMPUS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0,
                 tcn_kernel_sizes=[3, 5, 7]):
        # Define device
        self.history = None
        #self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"

        super(TEMPUS, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout_rate = dropout

        # Multiple Temporal Resolutions
        self.lstm_short = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True, dropout=self.dropout_rate if self.num_layers > 1 else 0,bidirectional=True)

        self.lstm_medium = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=self.dropout_rate if self.num_layers > 1 else 0,bidirectional=True)
        self.temporal_fusion = nn.Linear(hidden_size * 4, hidden_size * 2) # Fusion layer for temporal resolutions

        # Temporal Convolutional Network (TCN)
        self.tcn_modules = nn.ModuleList()
        for k_size in tcn_kernel_sizes:
            padding = (k_size - 1) // 2  # Same padding
            self.tcn_modules.append(nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=k_size, padding=padding),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=k_size, padding=padding),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU()
            ))
        self.tcn_fusion = nn.Linear(hidden_size * len(tcn_kernel_sizes), hidden_size * 2)

        # Combine TCN and LSTM features
        self.feature_fusion = nn.Linear(hidden_size * 4, hidden_size * 2)

        # Enhanced Temporal Attention
        self.temporal_attention = TemporalAttention(hidden_size * 2)

        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)

        self.regression_head = nn.Linear(hidden_size // 2, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)

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
        batch_size, seq_len, features = x.size()

        # Create time features (day of sequence percentage)
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

        # 1. Multiple Temporal Resolutions
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

        # 8. Apply temporal attention with time features
        attended_features, attention_weights = self.temporal_attention(fused_features, time_features)

        # Fully connected layers with dropout
        x = F.gelu(self.fc1(attended_features))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        
        # Use the regression head directly without softmax.
        output = self.regression_head(x).squeeze(-1)
        
        return {'regression': output, 'attention': attention_weights}

    def train_model(self, train_loader, test_loader, criterion, optimizer, num_epochs=10, clip_value=1.0, lambda_att=0.1):
        """
        Train the model with regression task
        """
        self.to(self.device)

        best_test_rmse = float('inf')
        best_model_state = None

        history = {
            'train_loss': [], 'test_loss': [],
            'rmse': [], 'mape': []
        }

        epoch_progress = tqdm(range(num_epochs), desc="Training Epochs")
        for epoch in epoch_progress:
            self.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                target_values = targets['shifted_prices'].to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                predictions = outputs['regression']
                loss = criterion(predictions, target_values)

                # Optionally add attention regularization
                attention_entropy = 0
                if lambda_att > 0:
                    attn = outputs['attention']
                    attention_entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=1).mean()
                    combined_loss = loss + lambda_att * attention_entropy # Combine losses (adjust lambda_att as needed)
                else:
                    combined_loss = loss
                
                combined_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += combined_loss.item()
            
            train_loss = running_loss / len(train_loader)
            
            # Evaluate on validation set
            test_loss, test_rmse, test_mape = self.evaluate(test_loader, criterion)
            epoch_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Test Loss': f'{test_loss:.4f}',
                'RMSE': f'{test_rmse:.4f}',
                'MAPE': f'{test_mape:.2f}%'
            })
            print(f"Epoch {epoch}/{num_epochs}; Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, RMSE: {test_rmse:.2f}%")

            # Store history
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['rmse'].append(test_rmse)
            history['mape'].append(test_mape)

            # Store model state with best val rmse
            if test_mape > best_test_rmse:
                best_test_rmse = test_rmse
                best_model_state = self.state_dict()

        # Load the best model state before returning
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        # Final evaluation with best model state
        test_loss, test_rmse, test_mape = self.evaluate(test_loader, criterion)
        print(f"\nBest Loss: {test_loss:.4f}, Best RMSE: {test_rmse:.4f}, Best MAPE: {test_mape:.2f}%")

        self.history = history

        return history

    def evaluate(self, data_loader, criterion):
        self.to(self.device)
        self.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                target_values = targets['shifted_prices'].to(self.device).unsqueeze(1)  # shape: (batch, 1)
                
                outputs = self(inputs)
                predictions = outputs['regression']  # use the regression key
                
                loss = criterion(predictions, target_values)
                total_loss += loss.item() * inputs.size(0)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_values.cpu().numpy())
        
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
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss Over Epochs', 'MAPE Over Epochs'))

            # Plot losses
            fig.add_trace(go.Scatter(y=self.history['train_loss'], name='Train Loss', line=dict(color='blue')), row=1,
                          col=1)
            fig.add_trace(go.Scatter(y=self.history['test_loss'], name='Test Loss', line=dict(color='orange')), row=1,
                          col=1)

            # Plot MAPE
            fig.add_trace(go.Scatter(y=self.history['mape'], name='Test MAPE', line=dict(color='orange')), row=1,
                          col=2)

            fig.update_layout(
                title='Model Training Metrics',
                xaxis_title='Epochs',
                height=700,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="MAPE", row=1, col=2)

            return fig

        else:
            print("Training history not available. Please run train_model() first.")
            return

# Implementation of Temporal Attention
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_attn = nn.Sequential(
            nn.Linear(1, 16),  # Simple time feature processing
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.feature_attn = nn.Linear(hidden_dim, 1)

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



# %%
# Create a tunable version of the LSTM model
class TunableLSTMClassifier(nn.Module):
    def __init__(self, config):
        super(TunableLSTMClassifier, self).__init__()
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.dropout_rate = config["dropout_rate"]

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=False
        )

        # Self-attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Apply attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers with dropout
        out = self.fc1(context_vector)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

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

    def evaluate(self, data_loader, criterion):
        """
        Evaluates the model on a given data loader.
        
        Args:
            data_loader (DataLoader): DataLoader for evaluation or testing.
            criterion: Loss function.
            
        Returns:
            tuple: Average loss and accuracy.
        """

        self.to(self.device)

        self.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # Calculate metrics
        f1 = f1_score(all_targets, all_preds, average='weighted')
        accuracy = accuracy_score(all_targets, all_preds)
        loss = val_loss / len(data_loader)

        return loss, accuracy, f1

    def train_model(self, train_loader, eval_loader, test_loader, criterion, optimizer, epochs=15, patience=10):
        best_eval_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        history = {
            'train_loss': [], 'train_acc': [],
            'eval_loss': [], 'eval_acc': [],
            'test_loss': [], 'test_acc': [],
            'eval_f1': [], 'test_f1': []
        }

        self.to(self.device)

        epoch_progress = tqdm(range(epochs), desc="Training Epochs")

        for epoch in epoch_progress:
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
    
                # Update metrics
                running_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
            # Calculate epoch metrics
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            eval_loss, eval_acc, eval_f1 = self.evaluate(eval_loader, criterion)
            test_loss, test_acc, test_f1 = self.evaluate(test_loader, criterion)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['eval_loss'].append(eval_loss)
            history['eval_acc'].append(eval_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['eval_f1'].append(eval_f1)
            history['test_f1'].append(test_f1)

            # Store best eval loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = self.state_dict()

            epoch_progress.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc*100:.2f}%',
                'val_loss': f'{eval_loss:.4f}',
                'val_acc': f'{eval_acc*100:.2f}%'
            })

        # Load the best model state before returning
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        # Final evaluation on test set
        test_loss, test_accuracy, _ = self.evaluate(test_loader, criterion)
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")

        self.history = history

        return history

    def get_predictions(self, training_df, seq_length=10):
        predictions = []
        dates = []
        actuals = []
        tickers = []
        confidences = []

        for i in range(seq_length, len(training_df)):
            # Get sequence
            sequence = training_df.iloc[i - seq_length:i].drop(columns=['Ticker']).values.astype(np.float32)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            sequence_tensor = sequence_tensor.to(self.device)

            # Get date, actual value, and ticker for the current index
            date = training_df.index[i]
            actual = training_df['Target'].iloc[i]
            ticker = training_df['Ticker'].iloc[i]

            # Make prediction
            self.eval()
            with torch.no_grad():
                output = self(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)  # Convert outputs to probabilities
                confidence, pred = torch.max(probabilities, 1)  # Get confidence and predicted class

            predictions.append(pred.item())
            confidences.append(confidence.item())  # Store the confidence score
            dates.append(date)
            actuals.append(actual)
            tickers.append(ticker)

        # Create DataFrame with predictions
        preds_df = pd.DataFrame({
            'Date': dates,
            'Ticker': tickers,
            'Actual': actuals,
            'Predicted': predictions,
            'Confidence': confidences  # Add confidence scores to the DataFrame
        })
        preds_df['entry_signal'] = preds_df['Predicted'] == 2  # Buy signal
        preds_df['exit_signal'] = preds_df['Predicted'] == 1  # Sell signal

        return preds_df

    def plot_training_history(self):
        if self.history is not None:
            # Create subplots for loss and accuracy
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'F1 Score'))

            # Plot losses
            fig.add_trace(go.Scatter(y=self.history['train_loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.history['eval_loss'], name='Eval Loss', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.history['test_loss'], name='Test Loss', line=dict(color='green')), row=1, col=1)

            # Plot f1
            fig.add_trace(go.Scatter(y=self.history['eval_f1'], name='Eval F1 Score', line=dict(color='orange')), row=1,
                          col=2)
            fig.add_trace(go.Scatter(y=self.history['test_f1'], name='Test F1 Score', line=dict(color='green')), row=1,
                          col=2)

            fig.update_layout(
                title='Training Metrics',
                xaxis_title='Epochs',
                height=700,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )

            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="F1", row=1, col=2)

            return fig

        else:
            print("Training history not available. Please run train_model() first.")
            return
