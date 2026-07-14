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

dataset_train = CustomDataset("../train_inputs/larger_ttbar_train.h5")
dataset_val = CustomDataset("../train_inputs/larger_ttbar_val.h5")
dataset_test = CustomDataset("../train_inputs/larger_ttbar_test.h5")

with h5py.File("../train_inputs/feature_labels.h5", "r") as f:
    feature_names = f["Feature_labels"][:].astype(str)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

### ------------------------------ Create Model Architecture ------------------------------ ###

import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.02

import torch.nn as nn

class GroupedTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, dropout=0.1, latent_dim=16):
        super().__init__()

        self.jet_proj = nn.Linear(104, d_model)
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

        self.pooler = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        # Split 126 features into groups
        jet_features = x[:, :104]
        muon_features = x[:, 104:114]
        electron_features = x[:, 114:124]
        met_features = x[:, 124:126]
        
        # Project each group to token and concatenate
        jet_token = self.jet_proj(jet_features).unsqueeze(1)
        muon_token = self.muon_proj(muon_features).unsqueeze(1)
        electron_token = self.electron_proj(electron_features).unsqueeze(1)
        met_token = self.met_proj(met_features).unsqueeze(1)
        
        tokens = torch.cat([jet_token, muon_token, electron_token, met_token], dim=1)
        
        # Transformer
        transform_out = self.transformer(tokens)

        # Expand query
        q = self.query.expand(x.size(0), 1, -1)

        # Attention pooling
        pooled_out, _ = self.pooler(query=q, key=transform_out, value=transform_out)
        
        return self.classifier(pooled_out.squeeze(1))

model = GroupedTransformer(
    d_model=64,
    nhead=4,
    num_layers=8,
    dropout=0.1,
    latent_dim=16
    ).to(device)

print(model)

def distribution_considering_loss(pred, target, bins, hist_min, hist_max, sigma=0.20, eps=1e-8):
    # Handle both single-target [B] and multi-target [B, D] regression.
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: pred {pred.shape}, target {target.shape}")

    target_dim = pred.shape[1]
    if isinstance(hist_min, (float, int)):
        hist_min = pred.new_tensor([hist_min] * target_dim)
    if isinstance(hist_max, (float, int)):
        hist_max = pred.new_tensor([hist_max] * target_dim)

    if hist_min.numel() != target_dim or hist_max.numel() != target_dim:
        raise ValueError(
            f"Histogram bounds must have {target_dim} values, got min={hist_min.numel()}, max={hist_max.numel()}"
        )

    kl_sum = 0.0
    for dim_idx in range(target_dim):
        pred_dim = pred[:, dim_idx]
        target_dim_values = target[:, dim_idx]

        centers = torch.linspace(
            hist_min[dim_idx], hist_max[dim_idx], bins,
            device=pred.device, dtype=pred.dtype
        )

        pred_kernel = torch.exp(-0.5 * ((pred_dim.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)
        target_kernel = torch.exp(-0.5 * ((target_dim_values.unsqueeze(1) - centers.unsqueeze(0)) / sigma) ** 2)

        pred_hist = pred_kernel.mean(dim=0) + eps
        target_hist = target_kernel.mean(dim=0) + eps

        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()

        kl_sum = kl_sum + torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)))

    return kl_sum / target_dim

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
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=0.08) # Use the WAdam optimiser

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

### ------------------------------ Run Training Loop ------------------------------ ###

# Run a training loop

print("Beginning Training Loop")
print("="*60)

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch
times = [] # Keep track of times

N_epochs = 800 # Number of epochs we iterate over

# KL settings: keep KL weak at start so regression can lock onto the target first.
kl_weight_max = 0.05
kl_ramp_epochs = 15

with h5py.File("../train_inputs/larger_ttbar_train.h5", "r") as f:
    Y_train= f["Y"][:]

hist_min = torch.tensor([np.quantile(Y_train, 0.001) - 0.25])
hist_max = torch.tensor([np.quantile(Y_train, 0.999) + 0.25])

hist_bins = 100

for epoch in range(N_epochs):

    start_time = time.time()

    current_kl_weight = kl_weight_max * min(1.0, epoch / kl_ramp_epochs)

    train_total_sum = 0.0
    train_mse_sum = 0.0
    train_kl_sum = 0.0

    model.train()

    for batch_x,batch_y in train_loader:
        # Ensure batch is on GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1) 
        # Perform a forward pass
        y_pred = model(batch_x)
        mse_loss = loss(y_pred,batch_y)
        
        kl_loss = distribution_considering_loss(
            y_pred, batch_y, bins=hist_bins, hist_min=hist_min, hist_max=hist_max)

        total_loss = mse_loss + current_kl_weight * kl_loss

        # Perform a backward pass + Optimisation
        optimiser.zero_grad()
        total_loss.backward()

        optimiser.step()

        train_total_sum += total_loss.item() * batch_x.size(0)
        train_mse_sum += mse_loss.item() * batch_x.size(0)
        train_kl_sum += kl_loss.item() * batch_x.size(0)

    avg_train_loss = train_total_sum / len(train_loader.dataset)

    avg_train_loss = train_total_sum / len(train_loader.dataset)
    train_mse = train_mse_sum / len(train_loader.dataset)
    train_kl = train_kl_sum / len(train_loader.dataset)

    losses.append(avg_train_loss)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    val_total_sum = 0.0
    val_mse_sum = 0.0
    val_kl_sum = 0.0

    model.eval() # Use this to keep track of the validation loss at each step
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # Ensure batch is on GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1) # Convert to (batch size, 1)

            y_pred = model(batch_x)

            mse_val_loss = loss(y_pred, batch_y)
            kl_val_loss = distribution_considering_loss(
                y_pred, batch_y, bins=hist_bins, hist_min=hist_min, hist_max=hist_max)

            total_val_loss = mse_loss + current_kl_weight * kl_loss
            
            val_total_sum += total_loss.item() * batch_x.size(0)
            val_mse_sum += mse_loss.item() * batch_x.size(0)
            val_kl_sum += kl_loss.item() * batch_x.size(0)
    
    avg_val_loss = val_total_sum / len(val_loader.dataset)
    val_mse = val_mse_sum / len(val_loader.dataset)
    val_kl = val_kl_sum / len(val_loader.dataset)

    val_losses.append(avg_val_loss)

    # Call scheduler outside batch loop 
    scheduler.step(avg_val_loss)

    tot_secs = np.sum(times)
    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... Training
        print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | MSE Loss :  {val_mse:.4f} | KL Loss : {val_kl:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s | Total Time: {tot_secs//60:.0f} minutes {tot_secs%60:.2f}s")

    final_epoch = epoch+1
    
    if epoch >= 50: # Starts the early stop loss after the 50th epoch 
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            print(f'Final Epoch before early stop [{epoch+1}/{N_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            break

    torch.cuda.empty_cache()

### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler info
with h5py.File("../train_inputs/larger_scaler_info.h5", "r") as f:
    scaler_Y_mean = f["Y_mean"][()]
    scaler_Y_scale = f["Y_scale"][()]

# Load test targets directly from H5
with h5py.File("../train_inputs/larger_ttbar_test.h5", "r") as f:
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
from scipy.special import rel_entr

MSE = mean_squared_error(Y_test_scaled, Y_pred)
RMS = root_mean_squared_error(Y_test_scaled,Y_pred)
MAE = mean_absolute_error(Y_test_scaled,Y_pred)
R2 = r2_score(Y_test_scaled,Y_pred)

# 1. Histogram the data to turn event regression into a PDF
# Use identical binning for both!
bins = np.linspace(0, max(np.max(Y_test_scaled),np.max(Y_pred)), num=100) # Adjust range to your P_T spectrum
p_counts, _ = np.histogram(Y_test_scaled, bins=bins)
q_counts, _ = np.histogram(Y_pred, bins=bins)

# 2. Normalize so they sum to 1 (making them valid probabilities)
P = p_counts / np.sum(p_counts)
Q = q_counts / np.sum(q_counts)

# 3. Add a tiny epsilon to prevent log(0) or division by zero errors
epsilon = 1e-10
P = np.clip(P, epsilon, 1)
Q = np.clip(Q, epsilon, 1)

# 4. Calculate proper KL Divergence
KLD = np.sum(rel_entr(P, Q))

# Inverse transform using saved scaler
Y_pred_geV = ((Y_pred * scaler_Y_scale) + scaler_Y_mean).flatten()
Y_test_geV = ((Y_test_scaled * scaler_Y_scale) + scaler_Y_mean).flatten()

MSE_GeV = mean_squared_error(Y_test_geV, Y_pred_geV)
RMS_GeV = root_mean_squared_error(Y_test_geV,Y_pred_geV)
MAE_GeV = mean_absolute_error(Y_test_geV,Y_pred_geV)
R2_GeV = r2_score(Y_test_geV,Y_pred_geV)

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
    f"Epochs: {final_epoch}/{N_epochs}\nBatch size: {batch_size}\nLR: {learning_rate}\nMSE in GeV: {MSE_GeV:.4f}\nRMSE in GeV: {RMS_GeV:.4f}\nMAE in GeV: {MAE_GeV:.4f}\nR^2 in GeV: {R2_GeV:.4f}",
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
n_bin = 50
_,bin_edges = np.histogram(Y_pred_geV,bins=n_bin)

axes[1,0].hist(Y_pred_geV, bins=bin_edges, color='blue', histtype='step', label='Predicted')
axes[1,0].hist(Y_test_geV, bins=bin_edges, color='red', histtype='step', label='True')
axes[1,0].set_xlabel("Predicted ttbar mass (GeV)")
axes[1,0].set_ylabel("Number of Events")
axes[1,0].set_title("Mass Distribution")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Text box with KL Divergence
axes[1,0].text(
    0.98, 0.98,
    f"KL Divergence: {KLD:.4f}",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
    ha="right",
    va="top",
    transform=axes[1,0].transAxes
)

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
plt.savefig("../plots/ttbar_Mass_div.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    "../train_outputs/ttbar_mass_predictions_div.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

print("Saved predictions to ../data/ttbar_mass_predictions_div.txt")

print("---------------Metrics---------------")
print(f"Epochs: {final_epoch}/{N_epochs}")
print(f"Batch size: {batch_size}")
print(f"LR: {learning_rate}")
print(f"MSE in GeV: {MSE_GeV:.4f}")
print(f"RMSE in GeV: {RMS_GeV:.4f}")
print(f"MAE in GeV: {MAE_GeV:.4f}")
print(f"R^2 in GeV: {R2_GeV:.4f}")
print(f"KL Divergence: {KLD:.4f}")