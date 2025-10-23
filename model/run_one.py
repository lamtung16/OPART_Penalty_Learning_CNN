import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
import lzma
from sklearn.model_selection import KFold
import copy

# Set random seed for reproducibility
random_seed = 123
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
row = params_df.iloc[param_row]
dataset   = row['dataset']
test_fold = row['test_fold']

# Early stopping parameters
patience = 200
max_epochs = 1000

# try to use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hinged Square Loss
class SquaredHingeLoss(nn.Module):
    def __init__(self, margin=1):
        super(SquaredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, y):
        low, high = y[:, 0:1], y[:, 1:2]
        loss_low = torch.relu(low - predicted + self.margin)
        loss_high = torch.relu(predicted - high + self.margin)
        loss = loss_low + loss_high
        return torch.mean(torch.square(loss))
    
# model
class DeepCNN(nn.Module):
    def __init__(self, n_conv_layers: int = 2, n_filters: int = 2, kernel_size: int = 2, n_dense_layers: int = 2, n_neurons: int = 10):
        super().__init__()

        # --- 1D CNN block ---
        conv_layers = []
        in_channels = 1
        for _ in range(n_conv_layers):
            conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
            ))
            conv_layers.append(nn.BatchNorm1d(n_filters))
            conv_layers.append(nn.Tanh())
            in_channels = n_filters
        self.conv = nn.Sequential(*conv_layers)

        # --- Global SUM pooling ---
        self.pool = lambda x: torch.sum(x, dim=-1, keepdim=True)

        # --- MLP block ---
        mlp_layers = []
        in_features = n_filters
        for _ in range(n_dense_layers):
            mlp_layers.append(nn.Linear(in_features, n_neurons))
            mlp_layers.append(nn.Tanh())
            in_features = n_neurons

        mlp_layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.mlp(x)
        return out

# Function to generate the predictions
def test_model(model, inputs):
    model.eval()                                                        # Set model to evaluation mode
    predictions = []
    with torch.no_grad():                                               # Disable gradient calculation
        for seq_input in inputs:
            seq_input = seq_input.unsqueeze(0).unsqueeze(0).to(device)  # Add batch dimension and move to device
            output_seq = model(seq_input)                               # Get model output
            predictions.append(output_seq.item())                       # Store the prediction
    return predictions

# Function to compute loss value
def get_loss_value(model, test_seqs, y_test, criterion):
    model.eval()
    total_test_loss = 0
    with torch.no_grad():                                               # Disable gradient calculation
        for i, seq_input in enumerate(test_seqs):
            target = y_test[i].unsqueeze(0).to(device)                  # Move target to device
            seq_input = seq_input.unsqueeze(0).unsqueeze(0).to(device)  # Prepare input and move to device
            output_seq = model(seq_input)                               # Get model output
            loss = criterion(output_seq, target)                        # Compute loss
            total_test_loss += loss.item()                              # Accumulate loss

    avg_test_loss = total_test_loss / len(test_seqs)                    # Calculate average loss
    return avg_test_loss

# Load sequence data from CSV
file_path = f'../data/{dataset}/profiles.csv.xz'
with lzma.open(file_path, 'rt') as file:
    signal_df = pd.read_csv(file)

# Group sequences by 'sequenceID'
seqs = tuple(signal_df.groupby('sequenceID'))

# Extract sequence IDs from seqs
sequence_ids = [group[0] for group in seqs]

# Load fold and target data
folds_df = pd.read_csv(f'../data/{dataset}/folds.csv').set_index('sequenceID').loc[sequence_ids].reset_index()
target_df = pd.read_csv(f'../data/{dataset}/target.csv').set_index('sequenceID').loc[sequence_ids].reset_index()

# Split data into train and test sets based on the fold
train_ids = folds_df[folds_df['fold'] != test_fold]['sequenceID']
test_ids = folds_df[folds_df['fold'] == test_fold]['sequenceID']

train_seqs = [torch.tensor(seq[1]['signal'].to_numpy(), dtype=torch.float32) for seq in seqs if seq[0] in list(train_ids)]
test_seqs = [torch.tensor(seq[1]['signal'].to_numpy(), dtype=torch.float32) for seq in seqs if seq[0] in list(test_ids)]

# Prepare target values for train and test
target_df_train = target_df[target_df['sequenceID'].isin(train_ids)]
target_df_test  = target_df[target_df['sequenceID'].isin(test_ids)]
y_train = torch.tensor(target_df_train.iloc[:, 1:].to_numpy(), dtype=torch.float32)
y_test  = torch.tensor(target_df_test.iloc[:, 1:].to_numpy(), dtype=torch.float32)

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
best_models = []

for train_idx, val_idx in kf.split(train_seqs):
    best_model = None
    best_val_loss = float('inf')

    for n_conv_layers in [2, 4]:
        for n_filters in [2, 4]:
            for kernel_size in [2, 4]:
                for n_dense_layers in [1, 2]:
                    for n_neurons in [10, 20]:
                        model = DeepCNN(n_conv_layers, n_filters, kernel_size, n_dense_layers, n_neurons).to(device)
                        optimizer = torch.optim.Adam(model.parameters())
                        criterion = SquaredHingeLoss()
                        patience_counter = 0
                        best_model_state = None
                        best_val_loss_model = float('inf')

                        # --- Training loop ---
                        for epoch in range(max_epochs):
                            model.train()
                            total_train_loss = 0.0

                            for i in train_idx:
                                seq_input = train_seqs[i].unsqueeze(0).unsqueeze(0).to(device)
                                target = y_train[i].unsqueeze(0).to(device)
                                
                                optimizer.zero_grad()
                                output = model(seq_input)
                                loss = criterion(output, target)
                                loss.backward()
                                optimizer.step()

                                total_train_loss += loss.item()

                            avg_train_loss = total_train_loss / len(train_idx)

                            # --- Validation ---
                            val_loss = get_loss_value(model, [train_seqs[i] for i in val_idx], y_train[val_idx], criterion)

                            # --- Early Stopping ---
                            if val_loss < best_val_loss_model:
                                best_val_loss_model = val_loss
                                best_model_state = copy.deepcopy(model.state_dict())
                                patience_counter = 0
                            else:
                                patience_counter += 1

                            if patience_counter >= patience:
                                break

                        # --- Save best model per config ---
                        if best_model_state:
                            model.load_state_dict(best_model_state)

                        if best_val_loss_model < best_val_loss:
                            best_val_loss = best_val_loss_model
                            best_model = copy.deepcopy(model)

    best_models.append(best_model)

# --- Test Predictions ---
model_outputs = []
for model in best_models:
    preds = test_model(model, test_seqs)
    model_outputs.append(preds)

# Average predictions across folds
target_mat_pred = np.mean(np.array(model_outputs), axis=0)

# --- Add sequenceID column ---
prediction = pd.DataFrame({
    'sequenceID': test_ids.values,
    'pred': target_mat_pred
})

# Save to CSV
prediction.to_csv(f"predictions/{dataset}.{test_fold}.csv", index=False)