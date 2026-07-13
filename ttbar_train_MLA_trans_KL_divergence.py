### ------------------------------ Code Brief ------------------------------ ###

# Selects device (GPU)
# Loads prepared and preprocessed data from 4 separate files (train,test,val,scaler)
# Converts X and target into tensors using a custom dataset
# Employs dataloaders for batching, with num workers = 4
# Defines MLA transformer model architecture with attention pooling
# Defines an early stopping mechanism
# Defines loss function (Huber loss), optimiser (Wadam) and scheduler (reduceLRonplateu)
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

dataset_train = CustomDataset("larger_importance_trimmed_ttbar_train.h5")
dataset_val = CustomDataset("larger_importance_trimmed_ttbar_val.h5")
dataset_test = CustomDataset("larger_importance_trimmed_ttbar_test.h5")

train_loader = DataLoader(
    dataset_train, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=10,
    pin_memory=True)

val_loader = DataLoader(
    dataset_val, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=10,
    pin_memory=True
)
test_loader = DataLoader(
    dataset_test, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=10,
    pin_memory=True
)

### ------------------------------ Transformer Model Architecture with Multi-head Latent Attention Mechansim ------------------------------ ###

import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.02

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)  # Learn token importance
        
    def forward(self, x):
        # x: (batch, tokens, d_model)
        weights = torch.softmax(self.attention(x), dim=1)  # (batch, tokens, 1)
        pooled = (x * weights).sum(dim=1)  # (batch, d_model)
        return pooled

class Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, dropout=0.1, latent_dim=16):
        super().__init__()
        
        # Jet compressors and decompressors (each jet has 8 features)
        self.jet_0_compressor = nn.Linear(8, latent_dim)
        self.jet_0_decompressor = nn.Linear(latent_dim, d_model)
        
        self.jet_1_compressor = nn.Linear(8, latent_dim)
        self.jet_1_decompressor = nn.Linear(latent_dim, d_model)
        
        self.jet_2_compressor = nn.Linear(8, latent_dim)
        self.jet_2_decompressor = nn.Linear(latent_dim, d_model)
        
        self.jet_3_compressor = nn.Linear(8, latent_dim)
        self.jet_3_decompressor = nn.Linear(latent_dim, d_model)
        
        # Muon features (10 features per muon pair)
        self.muon_compressor = nn.Linear(10, latent_dim)
        self.muon_decompressor = nn.Linear(latent_dim, d_model)
        
        # Electron features (10 features per electron pair)
        self.electron_compressor = nn.Linear(10, latent_dim)
        self.electron_decompressor = nn.Linear(latent_dim, d_model)
        
        # MET features (2 features)
        self.met_compressor = nn.Linear(2, latent_dim)
        self.met_decompressor = nn.Linear(latent_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = AttentionPooling(d_model)

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
        
    def forward(self, x):
        # Split features into groups
        leading_order_jet_features = x[:, 0:8]
        second_order_jet_features = x[:, 8:16]
        third_order_features = x[:, 16:24]
        fourth_order_features = x[:, 24:32]
        muon_features = x[:, 32:42]
        electron_features = x[:, 42:52]
        met_features = x[:, 52:54]
        
        # ========== COMPRESS → DECOMPRESS (MLA STYLE) ==========
        # 1. Compress: 8 → latent_dim (e.g., 16)
        # 2. Decompress: latent_dim → d_model (e.g., 64)
        # This forces a "bottleneck" that learns efficient representations
        
        jet_0_latent = self.jet_0_compressor(leading_order_jet_features)
        jet_0_token = self.jet_0_decompressor(jet_0_latent).unsqueeze(1)
        
        jet_1_latent = self.jet_1_compressor(second_order_jet_features)
        jet_1_token = self.jet_1_decompressor(jet_1_latent).unsqueeze(1)
        
        jet_2_latent = self.jet_2_compressor(third_order_features)
        jet_2_token = self.jet_2_decompressor(jet_2_latent).unsqueeze(1)
        
        jet_3_latent = self.jet_3_compressor(fourth_order_features)
        jet_3_token = self.jet_3_decompressor(jet_3_latent).unsqueeze(1)
        
        muon_latent = self.muon_compressor(muon_features)
        muon_token = self.muon_decompressor(muon_latent).unsqueeze(1)
        
        electron_latent = self.electron_compressor(electron_features)
        electron_token = self.electron_decompressor(electron_latent).unsqueeze(1)
        
        met_latent = self.met_compressor(met_features)
        met_token = self.met_decompressor(met_latent).unsqueeze(1)
        
        # Concatenate all tokens
        tokens = torch.cat([
            jet_0_token,
            jet_1_token,
            jet_2_token,
            jet_3_token,
            muon_token,
            electron_token,
            met_token
        ], dim=1)
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Attention pooling
        pooled = self.pool(tokens)
        
        return self.classifier(pooled)

# Create model with latent compression
model = Transformer(
    d_model=64,
    nhead=4,
    num_layers=4,
    dropout=0.1,
    latent_dim=16  # Compression
).to(device)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

### ------------------------------ Early stopping mechanism ------------------------------ ###

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
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

### ------------------------------ KL Divergence loss function ------------------------------ ###

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

        hist_min = hist_min.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        hist_max = hist_max.to(device=pred.device, dtype=pred.dtype).reshape(-1)

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

### ------------------------------ Define loss func, optimiser, and scheduler ------------------------------ ###

loss = nn.HuberLoss()
learning_rate = 0.001
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate) # Use the WAdam optimiser

warmup_epochs = 10
N_epochs = 200

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimiser, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=N_epochs - warmup_epochs, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimiser, 
    schedulers=[warmup_scheduler, cosine_scheduler], 
    milestones=[warmup_epochs]
)

### ------------------------------ Run Training Loop ------------------------------ ###

print("\n")
print("="*60)
print("Beginning Training Loop")
print("="*60)

losses = []
val_losses = []
times = []

# KL settings
kl_weight_max = 0.1  # Maximum KL weight
kl_ramp_epochs = 15   # Epochs to ramp up KL

for epoch in range(N_epochs):
    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    epoch_train_kl = 0.0

    # Ramp up KL weight
    current_kl_weight = kl_weight_max * min(1.0, (epoch + 1) / kl_ramp_epochs)

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).unsqueeze(1)
        
        y_pred = model(batch_x)
        
        # Huber loss (instead of MSE)
        huber_loss = loss(y_pred, batch_y)  # loss is already nn.HuberLoss()
        
        # KL divergence loss
        hist_min = batch_y.min().item() - 0.25
        hist_max = batch_y.max().item() + 0.25
        kl_loss = distribution_considering_loss(
            y_pred.flatten(),
            batch_y.flatten(),
            bins=100,
            hist_min=hist_min,
            hist_max=hist_max,
            sigma=0.20
        )
        
        # Combined loss: Huber + KL
        total_loss = huber_loss + current_kl_weight * kl_loss
        
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()
        
        epoch_train_loss += total_loss.item()
        epoch_train_kl += kl_loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_kl = epoch_train_kl / len(train_loader)
    losses.append(avg_train_loss)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_kl = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            y_pred = model(batch_x)
            
            huber_loss = loss(y_pred, batch_y)
            
            hist_min = batch_y.min().item() - 0.25
            hist_max = batch_y.max().item() + 0.25
            kl_loss = distribution_considering_loss(
                y_pred.flatten(),
                batch_y.flatten(),
                bins=100,
                hist_min=hist_min,
                hist_max=hist_max,
                sigma=0.20
            )
            
            total_loss = huber_loss + current_kl_weight * kl_loss
            epoch_val_loss += total_loss.item()
            epoch_val_kl += kl_loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_kl = epoch_val_kl / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step()

    print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} (KL: {avg_train_kl:.4f}) | Val Loss: {avg_val_loss:.4f} (KL: {avg_val_kl:.4f}) | KLw: {current_kl_weight:.3f} | Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")

    if epoch >= 150:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.cuda.empty_cache()
    
### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler info
with h5py.File("larger_importance_trimmed_scaler_info.h5", "r") as f:
    scaler_Y_mean = f["Y_mean"][()]
    scaler_Y_scale = f["Y_scale"][()]

# Load test targets directly from H5
with h5py.File("larger_importance_trimmed_ttbar_test.h5", "r") as f:
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

### ------------------------------ Calculate Performance Metrics ------------------------------ ###

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

def kl_divergence(pred, target, bins=100):
    # Identify common range
    min_val = min(pred.min(), target.min())
    max_val = max(pred.max(), target.max())
    
    # Create histograms using bins and common range
    pred_hist, _ = np.histogram(pred, bins=bins, range=(min_val, max_val))
    target_hist, _ = np.histogram(target, bins=bins, range=(min_val, max_val))
    
    # Convert to probabilities
    pred_probs = pred_hist / (pred_hist.sum() + 1e-10) # Addition of 1e-10 stops any divisions by 0
    target_probs = target_hist / (target_hist.sum() + 1e-10)
    
    # Avoid log(0)
    pred_probs = np.clip(pred_probs, 1e-10, 1.0)
    target_probs = np.clip(target_probs, 1e-10, 1.0)
    
    # KL divergence: target || pred
    kl = np.sum(target_probs * np.log(target_probs / pred_probs))
    
    return kl

def bootstrap_kl_divergence(pred, target, n_bootstrap=1000, bins=100):
    
    kl_original = kl_divergence(pred, target, bins)
    
    # Combine samples for resampling
    combined = np.concatenate([pred, target])
    n_pred = len(pred)
    n_target = len(target)
    
    # Bootstrap resampling
    kl_bootstrap = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        pred_resample = np.random.choice(combined, size=n_pred, replace=True)
        target_resample = np.random.choice(combined, size=n_target, replace=True)
        
        # Calculate KL on resampled data
        kl_bootstrap.append(kl_divergence(pred_resample, target_resample, bins))
    
    # Calculate statistics, chopping off lower 2.5% and upper 2.5%
    kl_std = np.std(kl_bootstrap)
    ci_lower = np.percentile(kl_bootstrap, 2.5)
    ci_upper = np.percentile(kl_bootstrap, 97.5)
    
    return kl_original, kl_std, (ci_lower, ci_upper), kl_bootstrap

kl_original, kl_std, kl_ci, kl_bootstrap = bootstrap_kl_divergence(
    Y_pred_geV, Y_test_geV, n_bootstrap=1000, bins=100
)

print(f"Epochs: {N_epochs}\nBatch size: {batch_size}\nLR: {learning_rate}\nMSE in GeV: {MSE_GeV:.4f}\nRMSE in GeV: {RMS_GeV:.4f}\nMAE in GeV: {MAE_GeV:.4f}\nR^2 in GeV: {R2_GeV:.4f}\nKL-Divergence : {kl_original:.4f} +- {kl_std:.4f}")

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
_, bin_edges = np.histogram(Y_pred_geV,bins=100)

axes[1,0].hist(Y_pred_geV, bins=bin_edges, color='blue', histtype='step')
axes[1,0].hist(Y_test_geV, bins=bin_edges, color='red', histtype='step')
axes[1,0].set_xlabel("Predicted ttbar mass (blue) vs true ttbar mass (red) in (GeV)")
axes[1,0].set_ylabel("Number of Events")
axes[1,0].set_title("Predicted Mass Distribution")
axes[1,0].text(
    0.98, 0.98,
    f"KL Divergence: {kl_original:.4f} ± {kl_std:.4f}\n"
    f"95% CI: [{kl_ci[0]:.4f}, {kl_ci[1]:.4f}]",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
    ha="right",
    va="top",
    transform=axes[1,0].transAxes
)
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
plt.savefig("ttbar_analysis_larger_4head_4layer_8workers_kl_div.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    "ttbar_analysis_larger_4head_4layer_8workers_kl_div.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

print("Saved predictions to ttbar_analysis_larger_4head_4layer_8workers_kl_div.txt")
print("Saved analysis plot to ttbar_analysis_larger_4head_4layer_8workers_kl_div.png")