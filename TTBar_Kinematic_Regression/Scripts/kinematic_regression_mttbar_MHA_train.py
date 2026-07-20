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

### ------------------------------ Imports ------------------------------ ###

import matplotlib.pyplot as plt # Used to plot graphs 
import os
import numpy as np
import torch
import h5py
import time
import vector
import awkward as ak

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
            self.M = torch.tensor(f["M"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.M[idx]

# ------------------------------ DataLoaders ------------------------------ #

from torch.utils.data import TensorDataset, DataLoader

batch_size = 4096

dataset_train = CustomDataset("kinematic_features_train.h5")
dataset_val = CustomDataset("kinematic_features_val.h5")
dataset_test = CustomDataset("kinematic_features_test.h5")

train_loader = DataLoader(
    dataset_train, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True)

val_loader = DataLoader(
    dataset_val, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    dataset_test, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

### ------------------------------ Transformer Model Architecture with MHA ------------------------------ ###
import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.02

import torch.nn as nn

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
    def __init__(self, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Project each group to d_model
        self.leading_order_jet_proj = nn.Linear(8, d_model)
        self.second_order_jet_proj = nn.Linear(8, d_model)
        self.third_order_jet_proj = nn.Linear(8, d_model)
        self.fourth_order_jet_proj = nn.Linear(8, d_model)
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
            nn.Linear(32, 8)
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
        
        # Project each group to token and concatenate
        leading_order_jet_token = self.leading_order_jet_proj(leading_order_jet_features).unsqueeze(1)
        second_order_jet_token = self.second_order_jet_proj(second_order_jet_features).unsqueeze(1)
        third_order_jet_token = self.third_order_jet_proj(third_order_features).unsqueeze(1)
        fourth_order_jet_token = self.fourth_order_jet_proj(fourth_order_features).unsqueeze(1)

        muon_token = self.muon_proj(muon_features).unsqueeze(1)
        electron_token = self.electron_proj(electron_features).unsqueeze(1)
        met_token = self.met_proj(met_features).unsqueeze(1)
        
        tokens = torch.cat([leading_order_jet_token,
                            second_order_jet_token,
                            third_order_jet_token,
                            fourth_order_jet_token,
                            muon_token, 
                            electron_token, 
                            met_token], dim=1)
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Attention pooling
        pooled = self.pool(tokens)
        
        return self.classifier(pooled)

model = Transformer(d_model=64,nhead=4,num_layers=8,dropout=0.1).to(device)

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
        target_dim = pred.shape[1]
        if isinstance(hist_min, (float, int)):
            hist_min = pred.new_tensor([hist_min] * target_dim)
        if isinstance(hist_max, (float, int)):
            hist_max = pred.new_tensor([hist_max] * target_dim)

        hist_min = hist_min.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        hist_max = hist_max.to(device=pred.device, dtype=pred.dtype).reshape(-1)

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

N_epochs = 20

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimiser, 
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6   
)

### ------------------------------ Run Training Loop ------------------------------ ###

# Load in mass scaling dat

with h5py.File("kinematic_features_scaler_info.h5", "r") as f:
    scaler_Y_mean = torch.tensor(f["Y_mean"][:], device=device, dtype=torch.float32)
    scaler_Y_scale = torch.tensor(f["Y_scale"][:], device=device, dtype=torch.float32)
    scaler_M_mean = torch.tensor(f["M_mean"][:], device=device, dtype=torch.float32)
    scaler_M_scale = torch.tensor(f["M_scale"][:], device=device, dtype=torch.float32) 


print("\n")
print("="*60)
print("Beginning Training Loop")
print("="*60)

# Track train losses
losses = [] # Kinematic train loss
kl_losses = []
mass_losses = []

# Track validation losses
val_losses = [] # Kinematic validation loss
kl_val_losses = []
mass_val_losses = []

# Track time
times = []

# KL settings (hyperparams)
kl_weight_max = 0.1  # Maximum KL weight
kl_ramp_epochs = 15   # Epochs to ramp up KL

# Mass loss settings (hyperparams)
mass_weight = 0.1

for epoch in range(N_epochs):
    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    epoch_train_kl = 0.0
    epoch_train_mass = 0.0

    # Ramp up KL weight
    current_kl_weight = kl_weight_max * min(1.0, (epoch + 1) / kl_ramp_epochs)

    for batch_x, batch_y, batch_m in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_m = batch_m.to(device)
        
        y_pred = model(batch_x)
        
        # Huber loss 
        huber_loss = loss(y_pred, batch_y)
        
        # KL divergence loss
        hist_min = batch_y.min().item() - 0.25
        hist_max = batch_y.max().item() + 0.25
        kl_loss = distribution_considering_loss(
            y_pred,
            batch_y,
            bins=100,
            hist_min=hist_min,
            hist_max=hist_max,
            sigma=0.20
        )

        # Unscale y pred and m 
        y_pred_unscaled = y_pred * scaler_Y_scale + scaler_Y_mean
        batch_m_unscaled = batch_m * scaler_M_scale + scaler_M_mean

        # Mass loss
        top_px, top_py, top_pz = y_pred_unscaled[:, 0], y_pred_unscaled[:, 1], y_pred_unscaled[:, 2]
        top_E = y_pred_unscaled[:, 6]

        antitop_px, antitop_py, antitop_pz = y_pred_unscaled[:, 3], y_pred_unscaled[:, 4], y_pred_unscaled[:, 5]
        antitop_E = y_pred_unscaled[:, 7]

        # Top mass from 4-vector
        top_m_pred = torch.sqrt(torch.clamp(top_E**2 - (top_px**2 + top_py**2 + top_pz**2), min=1e-6))
        antitop_m_pred = torch.sqrt(torch.clamp(antitop_E**2 - (antitop_px**2 + antitop_py**2 + antitop_pz**2), min=1e-6))

        # Mass loss
        mass_loss = loss(top_m_pred, batch_m_unscaled[:, 0]) + loss(antitop_m_pred, batch_m_unscaled[:, 1])

        # Combined loss: Huber + KL + Mass loss
        total_loss = huber_loss + current_kl_weight * kl_loss + mass_loss*mass_weight
        
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()
        
        epoch_train_loss += total_loss.item()
        epoch_train_kl += kl_loss.item()
        epoch_train_mass += mass_loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_kl = epoch_train_kl / len(train_loader)
    avg_train_mass = epoch_train_mass / len(train_loader)

    losses.append(avg_train_loss)
    kl_losses.append(avg_train_kl)
    mass_losses.append(avg_train_mass)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_kl = 0.0
    epoch_val_mass = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_m in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_m = batch_m.to(device)
            y_pred = model(batch_x)
            
            huber_loss = loss(y_pred, batch_y)
            
            hist_min = batch_y.min().item() - 0.25
            hist_max = batch_y.max().item() + 0.25
            kl_loss = distribution_considering_loss(
                y_pred,
                batch_y,
                bins=100,
                hist_min=hist_min,
                hist_max=hist_max,
                sigma=0.20
            )

            # Unscale y pred and m 
            y_pred_unscaled = y_pred * scaler_Y_scale + scaler_Y_mean
            batch_m_unscaled = batch_m * scaler_M_scale + scaler_M_mean

            # Mass loss
            top_px, top_py, top_pz = y_pred_unscaled[:, 0], y_pred_unscaled[:, 1], y_pred_unscaled[:, 2]
            top_E = y_pred_unscaled[:, 6]

            antitop_px, antitop_py, antitop_pz = y_pred_unscaled[:, 3], y_pred_unscaled[:, 4], y_pred_unscaled[:, 5]
            antitop_E = y_pred_unscaled[:, 7]

            # Top mass from 4-vector
            top_m_pred = torch.sqrt(torch.clamp(top_E**2 - (top_px**2 + top_py**2 + top_pz**2), min=1e-6))
            antitop_m_pred = torch.sqrt(torch.clamp(antitop_E**2 - (antitop_px**2 + antitop_py**2 + antitop_pz**2), min=1e-6))

            # Mass loss
            mass_loss = loss(top_m_pred, batch_m_unscaled[:, 0]) + loss(antitop_m_pred, batch_m_unscaled[:, 1])

            # Combined loss: Huber + KL + Mass loss
            total_loss = huber_loss + current_kl_weight * kl_loss + mass_loss*mass_weight      
            
            epoch_val_loss += total_loss.item()
            epoch_val_kl += kl_loss.item()
            epoch_val_mass += mass_loss.item()            
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_kl = epoch_val_kl / len(val_loader)
    avg_val_mass = epoch_val_mass / len(val_loader)

    val_losses.append(avg_val_loss)
    kl_val_losses.append(avg_val_kl)
    mass_val_losses.append(avg_val_mass)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} (KL: {avg_train_kl:.4f}) | Val Loss: {avg_val_loss:.4f} (KL: {avg_val_kl:.4f}) | Train - Val Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Mass Train Loss : {avg_train_mass:.4f} | Mass Val Loss : {avg_val_mass:.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")

    if epoch >= 150:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.cuda.empty_cache()
    
### ------------------------------ Evaluate Model ------------------------------ ###

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load scaler information
with h5py.File("kinematic_features_scaler_info.h5", "r") as f:
    scaler_Y_mean = f["Y_mean"][:]
    scaler_Y_scale = f["Y_scale"][:]

# Load test targets
with h5py.File("kinematic_features_test.h5", "r") as f:
    Y_test_scaled = f["Y"][:]

# Evaluate the model on test data
model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets, masses in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        list_of_predictions.append(outputs.cpu())

Y_pred = torch.cat(list_of_predictions).numpy()

# Compute and display metrics for each target feature

# Store target feature names in an array
target_names = ['top_px', 'top_py', 'top_pz', 
                'antitop_px', 'antitop_py', 'antitop_pz', 'top_E', 'antitop_E']

print("\n")
print("="*60)
print("TARGET FEATURE METRICS")
print("="*60)
r2_per_dim = []
for i in range(8):
    mse = mean_squared_error(Y_test_scaled[:, i], Y_pred[:, i])
    r2 = r2_score(Y_test_scaled[:, i], Y_pred[:, i])
    r2_per_dim.append(r2)
    mae = mean_absolute_error(Y_test_scaled[:,i], Y_pred[:,i])
    print(f"{target_names[i]}: MSE={mse:.4f}, R²={r2:.4f}, MAE = {mae:.4f}\n")
print("="*60)

# Inverse transform
Y_pred_geV = Y_pred * scaler_Y_scale + scaler_Y_mean
Y_test_geV = Y_test_scaled * scaler_Y_scale + scaler_Y_mean
    
### ------------------------------ Loss Curve and R^2 Plots ------------------------------ #

import matplotlib.pyplot as plt

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 12))

# Loss curve plot
axes1[0].plot(losses, label='Kinematic Train Loss')
axes1[0].plot(val_losses, label='Kinematic Validation Loss')
axes1[0].plot(kl_losses, label='KL Train Loss')
axes1[0].plot(kl_val_losses, label='KL Validation Loss')
axes1[0].plot(mass_losses, label='Mass Train Loss')
axes1[0].plot(mass_val_losses, label='Mass Validation Loss')
axes1[0].set_xlabel('Epoch')
axes1[0].set_ylabel('Loss')
axes1[0].set_title('Training and Validation Loss')
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)

# R squared values per dimension in bar chart representation plot
bars = axes1[1].bar(range(8), r2_per_dim, color='blue', edgecolor='black')
axes1[1].set_xticks(range(8))
axes1[1].set_xticklabels(target_names, rotation=45, ha='right')
axes1[1].set_ylabel('R-Squared')
axes1[1].set_title('R-Squared Bar Chart')
axes1[1].set_ylim([0,1])
axes1[1].grid(True, alpha=0.3)
# Add value labels on bars individually
for bar, val in zip(bars, r2_per_dim):
    axes1[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("summary_plots.png")

### ------------------------------ Target Feature Distribution and Resolution ------------------------------ #

fig2, axes2 = plt.subplots(2, 8, figsize=(24, 8))

for i in range(8):
    # Distribution (True vs Pred)
    axes2[0, i].hist(Y_test_scaled[:, i], bins=100, density=True, histtype='step',
                     label='True', color='blue', linewidth=1.5)
    axes2[0, i].hist(Y_pred[:, i], bins=100, density=True, histtype='step',
                     label='Pred', color='red', linewidth=1.5)
    axes2[0, i].set_title(f'{target_names[i]}')
    axes2[0, i].legend()
    axes2[0, i].grid(True, alpha=0.3)
    
    # Resolution (Pred - True)
    residuals = Y_pred[:, i] - Y_test_scaled[:, i]
    axes2[1, i].hist(residuals, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes2[1, i].axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect')
    axes2[1, i].axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=2,
                       label=f'μ={np.mean(residuals):.3f}')
    axes2[1, i].grid(True, alpha=0.3)
    axes2[1, i].legend()

# Set x-labels for top row
for i in range(8):
    axes2[0, i].set_xlabel('Distribution (True vs Pred)')

# Set x-labels for bottom row
for i in range(8):
    axes2[1, i].set_xlabel('Resolution (Pred - True)')

plt.tight_layout()
plt.savefig("per_target_feature_distribution_resolution.png")

### ------------------------------ Target Feature Scatter Plots ------------------------------ #

fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))

for i, ax in enumerate(axes3.flatten()):
    # Scatter plot: True vs Pred
    ax.scatter(Y_test_scaled[:, i], Y_pred[:, i], alpha=0.1, s=1, color='blue')
    
    # Perfect prediction line
    min_val = min(Y_test_scaled[:, i].min(), Y_pred[:, i].min())
    max_val = max(Y_test_scaled[:, i].max(), Y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    # R² for this dimension
    r2 = r2_score(Y_test_scaled[:, i], Y_pred[:, i])
    
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target_names[i]}\nR² = {r2:.4f}')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig("per_dimension_scatter_plots.png")
plt.show()

### ------------------------------ Invariant Mass Calculation Using Awk Vectors ------------------------------ #

import awkward as ak

Y_pred_unscaled = Y_pred * scaler_Y_scale + scaler_Y_mean
Y_test_unscaled = Y_test_scaled * scaler_Y_scale + scaler_Y_mean

# Convert NumPy to Awkward arrays
Y_pred_awk = ak.from_numpy(Y_pred_unscaled)
Y_test_awk = ak.from_numpy(Y_test_unscaled)

# Predicted vectors
top_pred = vector.Array(
    ak.zip({
        'px': Y_pred_awk[:, 0],
        'py': Y_pred_awk[:, 1],
        'pz': Y_pred_awk[:, 2],
        'E': Y_pred_awk[:,6]
    })
)

antitop_pred = vector.Array(
    ak.zip({
        'px': Y_pred_awk[:, 3],
        'py': Y_pred_awk[:, 4],
        'pz': Y_pred_awk[:, 5],
        'E': Y_pred_awk[:,7]
    })
)

# True vectors
top_true = vector.Array(
    ak.zip({
        'px': Y_test_awk[:, 0],
        'py': Y_test_awk[:, 1],
        'pz': Y_test_awk[:, 2],
        'E': Y_test_awk[:,6]
    })
)

antitop_true = vector.Array(
    ak.zip({
        'px': Y_test_awk[:, 3],
        'py': Y_test_awk[:, 4],
        'pz': Y_test_awk[:, 5],
        'E': Y_test_awk[:,7]
    })
)

# Add them
ttbar_pred = top_pred + antitop_pred
ttbar_true = top_true + antitop_true

# Get invariant mass
M_pred = ttbar_pred.mass
M_true = ttbar_true.mass

### ------------------------------ Invariant Mass Plots ------------------------------ #

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))

# Histogram comparison
axes4[0].hist(M_true, bins=100, density=True, histtype='step', 
              label='True M_ttbar', color='blue', linewidth=1.5)
axes4[0].hist(M_pred, bins=100, density=True, histtype='step', 
              label='Predicted M_ttbar', color='red', linewidth=1.5)
axes4[0].set_xlabel('M_ttbar (GeV)')
axes4[0].set_ylabel('Number of Events')
axes4[0].set_title('Invariant Mass Distribution')
axes4[0].legend()
axes4[0].grid(True, alpha=0.3)

# True vs Predicted scatter
axes4[1].scatter(M_true, M_pred, alpha=0.1, s=1, color='blue')
min_m = min(ak.min(M_true), ak.min(M_pred))
max_m = max(ak.max(M_true), ak.max(M_pred))
axes4[1].plot([min_m, max_m], [min_m, max_m], 'r--', linewidth=2, label='Perfect')
axes4[1].set_xlabel('True M_ttbar (GeV)')
axes4[1].set_ylabel('Predicted M_ttbar (GeV)')
axes4[1].set_title('True vs Predicted Invariant Mass')
axes4[1].legend()
axes4[1].grid(True, alpha=0.3)

# Resolution histogram
mass_resolution = M_pred - M_true
axes4[2].hist(mass_resolution, bins=50, density=True, color='red',  histtype='step')
axes4[2].axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect')
axes4[2].axvline(np.mean(mass_resolution), color='blue', linestyle='-', linewidth=2,
                 label=f'Mean = {np.mean(mass_resolution):.2f} GeV')
axes4[2].axvline(np.median(mass_resolution), color='green', linestyle='-', linewidth=2,
                 label=f'Median = {np.median(mass_resolution):.2f} GeV')
axes4[2].set_xlabel('Resolution (Pred - True) [GeV]')
axes4[2].set_ylabel('Number of Events')
axes4[2].set_title('Invariant Mass Resolution')
axes4[2].legend()
axes4[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("invariant_mass_plots.png")
plt.show()

### ------------------------------ Print Metrics for Invariant Mass ------------------------------ #

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

MSE_mass = mean_squared_error(M_true, M_pred)
RMSE_mass = np.sqrt(MSE_mass)
MAE_mass = mean_absolute_error(M_true, M_pred)
R2_mass = r2_score(M_true, M_pred)

print("\n" + "="*60)
print("INVARIANT MASS METRICS")
print("="*60)
print(f"MSE:  {MSE_mass:.4f}")
print(f"RMSE: {RMSE_mass:.4f} GeV")
print(f"MAE:  {MAE_mass:.4f} GeV")
print(f"R²:   {R2_mass:.4f}")
print("="*60)

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

results = np.column_stack([M_true, M_pred, M_pred - M_true])

np.savetxt(
    "ttbar_invariant_mass_results.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)