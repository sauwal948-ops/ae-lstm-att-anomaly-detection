import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Attention Mechanism ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        hidden = hidden.squeeze(0)
        
        # Repeat hidden state to match sequence length
        # [batch_size, hidden_size] -> [batch_size, seq_len, hidden_size]
        seq_len = encoder_outputs.size(1)
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate hidden state and encoder outputs
        # [batch_size, seq_len, hidden_size * 2]
        attn_input = torch.cat((hidden_repeated, encoder_outputs), dim=2)
        
        # Calculate attention scores
        # [batch_size, seq_len, hidden_size]
        attn_weights = torch.tanh(self.attn(attn_input))
        
        # [batch_size, seq_len]
        v_tensor = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attn_weights = torch.bmm(v_tensor, attn_weights.transpose(1, 2)).squeeze(1)
        
        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Context vector: [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        return context, attn_weights

# --- 2. Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # output: [batch_size, seq_len, hidden_size]
        # hidden: ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        output, (hidden, cell) = self.lstm(x)
        return output, hidden, cell

# --- 3. Decoder ---
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, input_size)
        self.attention = Attention(hidden_size)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        # input_step: [batch_size, 1, input_size]
        # hidden: [num_layers, batch_size, hidden_size]
        
        # Get context vector and attention weights
        # context: [batch_size, 1, hidden_size]
        # attn_weights: [batch_size, seq_len]
        context, attn_weights = self.attention(hidden[-1].unsqueeze(0), encoder_outputs)
        
        # Concatenate input and context
        # [batch_size, 1, input_size + hidden_size]
        lstm_input = torch.cat((input_step, context), dim=2)
        
        # output: [batch_size, 1, hidden_size]
        # hidden, cell: [num_layers, batch_size, hidden_size]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # output: [batch_size, input_size]
        output = self.out(output.squeeze(1))
        
        return output, hidden, cell, attn_weights

# --- 4. AE-LSTM-ATT Model ---
class AE_LSTM_ATT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, dropout=0.2):
        super(AE_LSTM_ATT, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, num_layers, dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(x)
        
        # Decoder
        outputs = torch.zeros(x.size(0), self.seq_len, self.input_size, device=x.device)
        attention_weights = torch.zeros(x.size(0), self.seq_len, self.seq_len, device=x.device)
        
        # The first input to the decoder is the last element of the encoder input sequence
        decoder_input = x[:, -1, :].unsqueeze(1) # [batch_size, 1, input_size]
        
        for t in range(self.seq_len):
            # output: [batch_size, input_size]
            # attn_weights_step: [batch_size, seq_len]
            output, hidden, cell, attn_weights_step = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output
            attention_weights[:, t, :] = attn_weights_step
            
            # Use the previous output as the next input (teacher forcing is not used here)
            decoder_input = output.unsqueeze(1)

        return outputs, attention_weights

# --- 5. Training and Evaluation Functions ---
def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    history = {'train_loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(data)
            
            # Loss calculation (reconstruction error)
            loss = criterion(outputs, data)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        history['train_loss'].append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return history

def evaluate_model(model, dataloader, device):
    model.eval()
    reconstruction_errors = []
    all_data = []
    all_reconstructions = []
    
    with torch.no_grad():
        for (data,) in dataloader:
            data = data.to(device).float()
            outputs, _ = model(data)
            
            # Calculate reconstruction error for each time step in the sequence
            error = torch.mean(torch.abs(outputs - data), dim=2) # [batch_size, seq_len]
            
            reconstruction_errors.append(error.cpu().numpy())
            all_data.append(data.cpu().numpy())
            all_reconstructions.append(outputs.cpu().numpy())

    reconstruction_errors = np.concatenate(reconstruction_errors, axis=0)
    all_data = np.concatenate(all_data, axis=0)
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    
    return reconstruction_errors, all_data, all_reconstructions

def get_attention_weights(model, data_point, device):
    model.eval()
    data_point = torch.from_numpy(data_point).unsqueeze(0).float().to(device)
    with torch.no_grad():
        _, attention_weights = model(data_point)
    # attention_weights: [1, seq_len, seq_len]
    return attention_weights.squeeze(0).cpu().numpy()

def calculate_metrics(errors, labels, contamination_factor=0.99):
    # Flatten errors and labels
    errors_flat = errors.flatten()
    labels_flat = labels.flatten()
    
    # 1. Determine Threshold (e.g., using a high percentile of training errors)
    # Since we are using an unsupervised model trained on nominal data, 
    # we use the errors to set a threshold.
    # For a proper ROC curve, we don't need a single threshold, but a range.
    
    # 2. ROC AUC Score
    # The anomaly score is the reconstruction error. Higher error means higher anomaly likelihood.
    # labels: 0 for nominal, 1 for anomaly
    roc_auc = roc_auc_score(labels_flat, errors_flat)
    
    # 3. Confusion Matrix (requires a single threshold)
    threshold = np.quantile(errors_flat, contamination_factor)
    predictions = (errors_flat > threshold).astype(int)
    
    # labels=[0, 1] ensures the matrix is 2x2 with order: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels_flat, predictions, labels=[0, 1])
    
    return roc_auc, threshold, cm

# --- Helper for Data Preparation (to be called from main.py) ---
def prepare_data_for_lstm(dataset, window_size):
    # dataset is a pandas DataFrame of the whole time series
    
    # The existing Dataset.py already handles sliding window creation, 
    # but it flattens the windows into a 2D array: [num_windows, window_size * num_features]
    # For LSTM, we need: [num_windows, window_size, num_features]
    
    num_features = len(Dataset.DEFAULT_COLUMNS)
    
    # Reshape the flattened array back into 3D:
    # [num_windows, window_size * num_features] -> [num_windows, window_size, num_features]
    data_3d = dataset.to_numpy().reshape(-1, window_size, num_features)
    
    return data_3d

# Add a placeholder for Dataset class to avoid circular dependency, 
# as the real Dataset class is in Dataset.py
class Dataset:
    DEFAULT_COLUMNS = [f'{key}_{i}' for key in ['position', 'velocity', 'effort'] for i in range(7)]

# --- Plotting Functions (to be called from main.py) ---
def plot_loss_curves(history, path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_reconstruction_error_distribution(errors, path='reconstruction_error_distribution.png'):
    plt.figure(figsize=(10, 6))
    sns.histplot(errors.flatten(), bins=50, kde=True)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Absolute Error (MAE)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def plot_attention_heatmap(attn_weights, path='attention_heatmap.png'):
    # attn_weights is [seq_len, seq_len]
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap='viridis', cbar_kws={'label': 'Attention Weight'})
    plt.title('Attention Heatmap (Decoder Step vs Encoder Time Step)')
    plt.xlabel('Encoder Time Step')
    plt.ylabel('Decoder Time Step')
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(cm, path='confusion_matrix.png', title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    
    # Calculate percentages
    total_nominal = cm[0, 0] + cm[0, 1]
    total_anomaly = cm[1, 0] + cm[1, 1]
    
    cm_perc = np.zeros_like(cm, dtype=float)
    if total_nominal > 0:
        cm_perc[0, :] = cm[0, :] / total_nominal * 100
    if total_anomaly > 0:
        cm_perc[1, :] = cm[1, :] / total_anomaly * 100
        
    group_percentages = [f"{value:.2f}%" for value in cm_perc.flatten()]
    
    labels = [f"{l1}\n{l2}\n{l3}" for l1, l2, l3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)
    
    plt.title(title)
    plt.xlabel('\nPredicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Nominal (0)', 'Anomaly (1)'])
    plt.yticks([0.5, 1.5], ['Nominal (0)', 'Anomaly (1)'], rotation=90)
    plt.savefig(path)
    plt.close()

