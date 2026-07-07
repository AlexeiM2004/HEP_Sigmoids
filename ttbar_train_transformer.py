### ------------------------------ Code Brief ------------------------------ ###

# Selects device (GPU)
# Loads prepared and preprocessed data from "ttbar_data_prep_and_preprocess.py"
# Converts X and target into tensors
# Employers dataloaders for batching
# Defines model architecture
# Defines an early stopping mechanism
# Defines loss function, optimiser and scheduler
# Runs training loop
# Evaluates model
# Generates plots
# Saves predicted and true ttbar mass to file

### ------------------------------ Imports ------------------------------ ###

import matplotlib.pyplot as plt # Used to plot graphs 
import os
import numpy as np
import torch
import h5py
import time

from datetime import datetime

### ------------------------------ Print Current Timestamp ------------------------------ ###

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Job started at :", formatted_time)

### ------------------------------ Device Usage ------------------------------ ###

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device with number of GPUs: {torch.cuda.device_count()}")

### ------------------------------ Load Preprocessed Data ------------------------------ ###

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.X = torch.tensor(f["X"][:], dtype=torch.float32)
            self.Y = torch.tensor(f["Y"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ------------------------------ DataLoaders ------------------------------ #

from torch.utils.data import TensorDataset, DataLoader

batch_size = 4096

dataset_train = CustomDataset("../data/ttbar_train.h5")
dataset_val = CustomDataset("../data/ttbar_val.h5")
dataset_test = CustomDataset("../data/ttbar_test.h5")

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

### ------------------------------ Create Model Architecture (MLP for now) ------------------------------ ###

import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.025

import torch.nn as nn

class GroupedTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Project each group to d_model
        self.jet_proj = nn.Linear(64, d_model)
        self.muon_proj = nn.Linear(10, d_model)
        self.electron_proj = nn.Linear(10, d_model)
        self.met_proj = nn.Linear(2, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def pool(self, x):
        return x.mean(dim=1)
        
    def forward(self, x):
        # Split 86 features into groups
        jet_features = x[:, :64]
        muon_features = x[:, 64:74]
        electron_features = x[:, 74:84]
        met_features = x[:, 84:86]
        
        # Project each group to token and concatenate
        jet_token = self.jet_proj(jet_features).unsqueeze(1)
        muon_token = self.muon_proj(muon_features).unsqueeze(1)
        electron_token = self.electron_proj(electron_features).unsqueeze(1)
        met_token = self.met_proj(met_features).unsqueeze(1)
        
        tokens = torch.cat([jet_token, muon_token, electron_token, met_token], dim=1)
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Global pooling
        pooled = self.pool(tokens)
        
        return self.classifier(pooled)

model = GroupedTransformer(d_model=64,nhead=8,num_layers=8,dropout=0.1).to(device)

print(model)

### ------------------------------ Early stopping mechanism ------------------------------ ###

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience        # How many epochs to wait
        self.min_delta = min_delta      # Minimum improvement to count
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, avg_val_loss):
        if self.best_loss - avg_val_loss > self.min_delta:
            self.best_loss = avg_val_loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping()

### ------------------------------ Define loss func, optimiser, and scheduler ------------------------------ ###

loss = nn.HuberLoss()
learning_rate = 0.001
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Use the WAdam optimiser

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)

### ------------------------------ Run Training Loop ------------------------------ ###

# Run a training loop

print("Beginning Training Loop")
print("="*60)

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch
times = [] # Keep track of times

N_epochs = 200 # Number of epochs we iterate over

for epoch in range(N_epochs):

    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0

    for batch_x,batch_y in train_loader:
        # Ensure batch is on GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1) 
        # Perform a forward pass
        y_pred = model(batch_x)
        train_loss = loss(y_pred,batch_y)
        
        # Perform a backward pass + Optimisation
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()
        epoch_train_loss += train_loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    losses.append(avg_train_loss)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    model.eval() # Use this to keep track of the validation loss at each step
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # Ensure batch is on GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1) # Convert to (batch size, 1)
            y_pred = model(batch_x)
            val_loss = loss(y_pred, batch_y)
            epoch_val_loss += val_loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Call scheduler outside batch loop 
    scheduler.step(avg_val_loss)

    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... Training
        print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")

    if epoch >= 50: # Starts the early stop loss after the 50th epoch 
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            print(f'Final Epoch before early stop [{epoch+1}/{N_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            break

    torch.cuda.empty_cache()

### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler info
with h5py.File("../data/scaler_info.h5", "r") as f:
    scaler_Y_mean = f["Y_mean"][()]
    scaler_Y_scale = f["Y_scale"][()]

# Load test targets directly from H5
with h5py.File("../data/ttbar_test.h5", "r") as f:
    Y_test_scaled = f["Y"][:]

model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device) 
        outputs = model(inputs)
        list_of_predictions.append(outputs.cpu()) 


pred = torch.concatenate(list_of_predictions)
Y_pred = pred.detach().cpu().numpy().flatten()

from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score

MSE = mean_squared_error(Y_test_scaled, Y_pred)
RMS = root_mean_squared_error(Y_test_scaled,Y_pred)
MAE = mean_absolute_error(Y_test_scaled,Y_pred)
R2 = r2_score(Y_test_scaled,Y_pred)

# Inverse transform using saved scaler
Y_pred_geV = ((Y_pred * scaler_Y_scale) + scaler_Y_mean).flatten()
Y_test_geV = ((Y_test_scaled * scaler_Y_scale) + scaler_Y_mean).flatten()

MSE_GeV = mean_squared_error(Y_test_geV, Y_pred_geV)
RMS_GeV = root_mean_squared_error(Y_test_geV,Y_pred_geV)
MAE_GeV = mean_absolute_error(Y_test_geV,Y_pred_geV)
R2_GeV = r2_score(Y_test_geV,Y_pred_geV)

### ------------------------------ Feature Importances ------------------------------ ###

def make_prediction(model, data_loader):
    model.eval()
    list_of_predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            list_of_predictions.append(outputs.cpu().numpy())
    return np.vstack(list_of_predictions).flatten()

def calc_feature_importances(model, X_tensor, Y_tensor, Y_vals, batch_size):
    # Make dataloader for baseline
    baseline_dataset = TensorDataset(X_tensor, Y_tensor)
    baseline_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate baseline scores
    baseline_pred = make_prediction(model, baseline_loader)
    baseline_score = r2_score(Y_vals, baseline_pred)
    importances = []
    
    for feature_idx in range(X_tensor.shape[1]):
        # Shuffle the values of the current feature
        X_tensor_shuffled = X_tensor.clone()
        shuffle_idx = torch.randperm(X_tensor.shape[0])
        X_tensor_shuffled[:, feature_idx] = X_tensor_shuffled[:, feature_idx][shuffle_idx]
        
        # Make shuffled dataloader
        shuffled_dataset = TensorDataset(X_tensor_shuffled, Y_tensor)
        shuffled_dataloader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False)
        
        # Calculate shuffled score
        shuffled_pred = make_prediction(model, shuffled_dataloader)
        shuffled_score = r2_score(Y_vals, shuffled_pred)
        
        # Subtract scores for importance
        importances.append(baseline_score - shuffled_score)
    
    return np.array(importances)

# Load validation data for feature importance
with h5py.File("../data/ttbar_val.h5", "r") as f:
    X_val = torch.tensor(f["X"][:], dtype=torch.float32)
    Y_val = torch.tensor(f["Y"][:], dtype=torch.float32)

# Load feature names from preprocessing (you need to save these or define them)
feature_names = [f"jet_{i}" for i in range(64)] + [f"muon_{i}" for i in range(10)] + [f"electron_{i}" for i in range(10)] + ["met_met", "met_phi"]

# Calculate feature importances
feature_importances = calc_feature_importances(model, X_val, Y_val, Y_val.numpy(), batch_size)

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_names = np.array(feature_names)[sorted_idx]
sorted_importances = feature_importances[sorted_idx]
    
# ------------------------------ Plotting ------------------------------ #
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(24, 14))  # 2 rows, 2 columns

# Loss plot
axes[0,0].plot(losses, label='Train Loss')
axes[0,0].plot(val_losses, label='Validation Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training and Validation Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Text box with metrics
axes[0,0].text(
    0.98, 0.98,
    f"Epochs: {N_epochs}\nBatch size: {batch_size}\nLR: {learning_rate}\nMSE in GeV: {MSE_GeV:.4f}\nRMSE in GeV: {RMS_GeV:.4f}\nMAE in GeV: {MAE_GeV:.4f}\nR^2 in GeV: {R2_GeV:.4f}",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
    ha="right",
    va="top",
    transform=axes[0,0].transAxes
)

# True vs Predicted plot 

# Plot true vs predicted in GeV
axes[0,1].scatter(Y_test_geV, Y_pred_geV, alpha=0.3, s=1)
axes[0,1].plot([Y_test_geV.min(), Y_test_geV.max()], 
         [Y_test_geV.min(), Y_test_geV.max()], 'b--')
axes[0,1].set_xlabel("True ttbar mass (GeV)")
axes[0,1].set_ylabel("Predicted ttbar mass (GeV)")
axes[0,1].set_title(f"TTBar mass")

# Predicted masses histogram
axes[1,0].hist(Y_pred_geV, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1,0].set_xlabel("Predicted ttbar mass (GeV)")
axes[1,0].set_ylabel("Number of Events")
axes[1,0].set_title("Predicted Mass Distribution")
axes[1,0].grid(True, alpha=0.3)

# Resolution histogram
resolution = Y_pred_geV - Y_test_geV
axes[1,1].hist(resolution, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1,1].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Perfect')
axes[1,1].axvline(x=np.mean(resolution), color='blue', linestyle='-', linewidth=2, label=f'Mean = {np.mean(resolution):.2f} GeV')
axes[1,1].set_xlabel("Resolution (Predicted - True) [GeV]")
axes[1,1].set_ylabel("Number of Events")
axes[1,1].set_title("Resolution Distribution")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("TTBar_Mass.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    "ttbar_mass_predictions.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

import pandas as pd

importance_df = pd.DataFrame({
    "feature": sorted_names,
    "importance": sorted_importances
})
importance_df.to_csv("ttbar_mass_importances.csv", index=False)

print("Saved predictions to ../data/ttbar_mass_predictions.txt")
print("Saved feature importances to ../data/ttbar_mass_importances.csv")