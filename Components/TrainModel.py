import torch
import torch.nn as nn
import shap
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class LSTMClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        # Define device
        self.history = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        super(LSTMClassifierModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.dropout_rate = dropout

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention mechanism: maps LSTM output to a single attention score per time step
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

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

        # Compute attention weights and take a weighted sum over the time dimension
        attention_weights = torch.softmax(self.attention(out), dim=1)
        out = torch.sum(attention_weights * out, dim=1)

        # Fully connected layers with dropout applied between layers
        out = self.dropout(out)
        out = self.fc1(out)
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
