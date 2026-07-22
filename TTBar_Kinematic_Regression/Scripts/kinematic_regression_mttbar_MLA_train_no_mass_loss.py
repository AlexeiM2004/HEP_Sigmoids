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

import matplotlib.pyplot as plt 
import os
import numpy as np
import torch
import h5py
import time
import vector
import awkward as ak
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass, field

### ------------------------------ Print Current Timestamp ------------------------------ ###

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Job started at :", formatted_time)

### ------------------------------ Device Usage ------------------------------ ###

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device with number of GPUs: {torch.cuda.device_count()}")

### ------------------------------ Control Panels ------------------------------ ###

@dataclass
class Data_Configuration:
    train_file : str = "kinematic_features_train.h5"
    val_file : str = "kinematic_features_val.h5"
    test_file : str = "kinematic_features_test.h5"
    scaler_file : str = "kinematic_features_scaler.h5"
    batch_size : int = 4096
    num_workers : int = 4
    pin_memory : bool = True

@dataclass
class Model_Configuration:
    d_model : int = 64
    nhead : int = 4
    num_layers : int = 4
    dropout : float = 0.1
    latent_dim: int = 16 

@dataclass
class Training_Configuration:
    # Early stopping mechanism
    patience : int = 10
    min_delta : float = 0.0
    min_early_stop : int = 150

    # Training hyperparameters
    num_epochs : int = 150
    learning_rate : float = 0.005
    weight_decay : float = 0.01

    # KL divergence settings
    kl_weight_max: float = 0.05
    kl_ramp_epochs: int = 15
    kl_bins: int = 100
    kl_sigma: float = 0.20
    kl_eps: float = 1e-8

    # Scheduler settings
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6

@dataclass
class Data_Saving:
    loss_r2_summary_plots : str = "MLA_train_no_mass_loss_loss_r2_summary_plots.png"
    target_feature_plots : str = "MLA_train_no_mass_loss_target_feature_plots.png"
    target_dimension_scatter_plots : str = "MLA_train_no_mass_loss_target_dimension_scatter_plots.png"
    invariant_mass_plots : str = "MLA_train_no_mass_loss_invariant_mass_plots.png"
    invariant_mass_data : str = "MLA_train_no_mass_loss_invariant_mass_data.txt"

@dataclass
class Main_Configuration:
    data_config: Data_Configuration = field(default_factory=Data_Configuration)
    model_config: Model_Configuration = field(default_factory=Model_Configuration)
    train_config: Training_Configuration = field(default_factory=Training_Configuration)
    data_saving: Data_Saving = field(default_factory=Data_Saving)

control_panel = Main_Configuration()

def display_config(control_panel):
    print("\n" + "="*60)
    print("CONTROL PANEL")
    print("="*60)
    
    sections = {
        'Data': control_panel.data_config,
        'Model': control_panel.model_config,
        'Training': control_panel.train_config
    }
    
    for section_name, section in sections.items():
        print(f"\n{section_name.upper()} CONFIGURATION")
        for key, value in section.__dict__.items():
            print(f"  {key}: {value}")

display_config(control_panel)

### ------------------------------ Load Preprocessed Data ------------------------------ ###

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.X = torch.tensor(f["X"][:], dtype=torch.float32)
            self.Y = torch.tensor(f["Y"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ------------------------------ Data Loaders ------------------------------ #

def create_loader(split):
    file_map = {
        'train': control_panel.data_config.train_file,
        'val': control_panel.data_config.val_file,
        'test': control_panel.data_config.test_file
    }
    
    dataset = CustomDataset(file_map[split])
    
    return DataLoader(
        dataset,
        batch_size=control_panel.data_config.batch_size,
        shuffle=(split == 'train'),
        num_workers=control_panel.data_config.num_workers,
        pin_memory=control_panel.data_config.pin_memory
    )

train_loader, val_loader, test_loader = [create_loader(s) for s in ['train', 'val', 'test']]


### ------------------------------ Model Architecture ------------------------------ ###

# Define attention pooling mechanism 

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)  # Learn token importance from attention mechanism
        
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled

# Define MLA transformer

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, latent_dim):
        super().__init__()
        
        # Jet feature compressors and decompressors
        self.jet_0_compressor = nn.Linear(8, latent_dim)
        self.jet_0_decompressor = nn.Linear(latent_dim, d_model)
        
        self.jet_1_compressor = nn.Linear(8, latent_dim)
        self.jet_1_decompressor = nn.Linear(latent_dim, d_model)
        
        self.jet_2_compressor = nn.Linear(8, latent_dim)
        self.jet_2_decompressor = nn.Linear(latent_dim, d_model)
        
        self.jet_3_compressor = nn.Linear(8, latent_dim)
        self.jet_3_decompressor = nn.Linear(latent_dim, d_model)
        
        # Muon feature compressor and decompressor
        self.muon_compressor = nn.Linear(10, latent_dim)
        self.muon_decompressor = nn.Linear(latent_dim, d_model)
        
        # Electron feature compressor and decompressor
        self.electron_compressor = nn.Linear(10, latent_dim)
        self.electron_decompressor = nn.Linear(latent_dim, d_model)
        
        # MET feature compressors and decompressors
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

        # Encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling
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

# Create model instance

model = Transformer(
    d_model=control_panel.model_config.d_model,
    nhead=control_panel.model_config.nhead,
    num_layers=control_panel.model_config.num_layers,
    dropout=control_panel.model_config.dropout,
    latent_dim=control_panel.model_config.latent_dim
).to(device)

### ------------------------------ Early stopping mechanism ------------------------------ ###

class EarlyStopping:
    def __init__(self):
        self.patience = control_panel.train_config.patience # Number of epochs to wait
        self.min_delta = control_panel.train_config.min_delta # Minimum change
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, avg_val_loss):
        if self.best_loss - avg_val_loss > self.min_delta:
            self.best_loss = avg_val_loss
            self.counter = 0 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping()

### ------------------------------ KL Divergence loss function ------------------------------ ###

def distribution_considering_loss(pred, target, bins, hist_min, hist_max):
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

            pred_kernel = torch.exp(-0.5 * ((pred_dim.unsqueeze(1) - centers.unsqueeze(0)) / control_panel.train_config.kl_sigma) ** 2)
            target_kernel = torch.exp(-0.5 * ((target_dim_values.unsqueeze(1) - centers.unsqueeze(0)) / control_panel.train_config.kl_sigma) ** 2)

            pred_hist = pred_kernel.mean(dim=0) + control_panel.train_config.kl_eps
            target_hist = target_kernel.mean(dim=0) + control_panel.train_config.kl_eps

            pred_hist = pred_hist / pred_hist.sum()
            target_hist = target_hist / target_hist.sum()

            kl_sum = kl_sum + torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)))

        return kl_sum / target_dim

### ------------------------------ Define loss func, optimiser, and scheduler ------------------------------ ###

# Loss function
loss = nn.HuberLoss()

# Optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=control_panel.train_config.learning_rate, weight_decay=control_panel.train_config.weight_decay)

# Scheduler
scheduler = ReduceLROnPlateau(
    optimiser, 
    mode='min',
    factor=control_panel.train_config.scheduler_factor,
    patience=control_panel.train_config.scheduler_patience,
    min_lr=control_panel.train_config.scheduler_min_lr
)

### ------------------------------ Run Training Loop ------------------------------ ###

# Load in mass scaling data
with h5py.File("kinematic_features_scaler_info.h5", "r") as f:
    scaler_Y_mean = torch.tensor(f["Y_mean"][:], device=device, dtype=torch.float32)
    scaler_Y_scale = torch.tensor(f["Y_scale"][:], device=device, dtype=torch.float32)

# Track train losses
losses = [] # Kinematic train loss
kl_losses = []

# Track validation losses
val_losses = [] # Kinematic validation loss
kl_val_losses = []

# Track time
times = []

print("\n")
print("="*60)
print("Beginning Training Loop")
print("="*60)

# Run training loop
for epoch in range(control_panel.train_config.num_epochs):
    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    epoch_train_kl = 0.0

    # Ramp up KL weight
    current_kl_weight = control_panel.train_config.kl_weight_max * min(1.0, (epoch + 1) / control_panel.train_config.kl_ramp_epochs)

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        y_pred = model(batch_x)
        
        # Huber loss 
        huber_loss = loss(y_pred, batch_y)
        
        # KL divergence loss
        hist_min = batch_y.min().item() - 0.25
        hist_max = batch_y.max().item() + 0.25
        kl_loss = distribution_considering_loss(
            y_pred,
            batch_y,
            bins=control_panel.train_config.kl_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )

        # Combined loss: Huber + KL + Mass loss
        total_loss = huber_loss + current_kl_weight * kl_loss
        
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()
        
        epoch_train_loss += total_loss.item()
        epoch_train_kl += kl_loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_kl = epoch_train_kl / len(train_loader)

    losses.append(avg_train_loss)
    kl_losses.append(avg_train_kl)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    # Validation
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_kl = 0.0
    epoch_val_mass = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            huber_loss = loss(y_pred, batch_y)
            
            hist_min = batch_y.min().item() - 0.25
            hist_max = batch_y.max().item() + 0.25
            kl_loss = distribution_considering_loss(
                y_pred,
                batch_y,
                bins=control_panel.train_config.kl_bins,
                hist_min=hist_min,
                hist_max=hist_max,
            )

            # Combined loss: Huber + KL + Mass loss
            total_loss = huber_loss + current_kl_weight * kl_loss
            
            epoch_val_loss += total_loss.item()
            epoch_val_kl += kl_loss.item()      
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    avg_val_kl = epoch_val_kl / len(val_loader)

    val_losses.append(avg_val_loss)
    kl_val_losses.append(avg_val_kl)

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{control_panel.train_config.num_epochs} | Train Loss: {avg_train_loss:.4f} (KL: {avg_train_kl:.4f}) | Val Loss: {avg_val_loss:.4f} (KL: {avg_val_kl:.4f}) | Train - Val Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")

    if epoch >= control_panel.train_config.min_early_stop:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.cuda.empty_cache()
    
### ------------------------------ Evaluate Model ------------------------------ ###

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
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        list_of_predictions.append(outputs.cpu())

Y_pred = torch.cat(list_of_predictions).numpy()

# Unscale from 0-1 into GeV
Y_pred_unscaled = Y_pred * scaler_Y_scale + scaler_Y_mean
Y_test_unscaled = Y_test_scaled * scaler_Y_scale + scaler_Y_mean

### ------------------------------ Calculate KL Divergence ------------------------------ ###


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

### ------------------------------ Display Metrics ------------------------------ ###

# Store target feature names in an array
target_names = ['top_px', 'top_py', 'top_pz', 
                'antitop_px', 'antitop_py', 'antitop_pz', 'top_E', 'antitop_E']

print("\n")
print("="*60)
print("TARGET FEATURE METRICS")
print("="*60)
r2_per_dim = []
for i in range(8):
    mse = mean_squared_error(Y_test_unscaled[:, i], Y_pred_unscaled[:, i])
    r2 = r2_score(Y_test_unscaled[:, i], Y_pred_unscaled[:, i])
    r2_per_dim.append(r2)
    mae = mean_absolute_error(Y_test_unscaled[:,i], Y_pred_unscaled[:,i])
    print(f"{target_names[i]}: MSE={mse:.4f}, R²={r2:.4f}, MAE = {mae:.4f}\n")
print("="*60)
for i, name in enumerate(target_names):
    kl_feat, std_feat, ci_feat, _ = bootstrap_kl_divergence(Y_pred_unscaled[:, i], Y_test_unscaled[:, i], n_bootstrap=1000, bins=100)
    print(f"{name}: KL = {kl_feat:.4f} ± {std_feat:.4f}")
print("="*60)

### ------------------------------ Plot Loss Curve and R^2 Plots ------------------------------ #

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 12))

# Loss curve plot
axes1[0].plot(losses, label='Kinematic Train Loss')
axes1[0].plot(val_losses, label='Kinematic Validation Loss')
axes1[0].plot(kl_losses, label='KL Train Loss')
axes1[0].plot(kl_val_losses, label='KL Validation Loss')
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
plt.savefig(control_panel.data_saving.loss_r2_summary_plots)

### ------------------------------ Plot Target Feature Distribution and Resolution ------------------------------ #

fig2, axes2 = plt.subplots(2, 8, figsize=(24, 8))

for i in range(8):
    # Distribution (True vs Pred)
    axes2[0, i].hist(Y_test_unscaled[:, i], bins=100, density=True, histtype='step',
                     label='True', color='blue', linewidth=1.5)
    axes2[0, i].hist(Y_pred_unscaled[:, i], bins=100, density=True, histtype='step',
                     label='Pred', color='red', linewidth=1.5)
    axes2[0, i].set_title(f'{target_names[i]}')
    axes2[0, i].legend()
    axes2[0, i].grid(True, alpha=0.3)
    
    # Resolution (Pred - True)
    residuals = Y_pred_unscaled[:, i] - Y_test_unscaled[:, i]
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
plt.savefig(control_panel.data_saving.target_feature_plots)

### ------------------------------ Plot Target Feature Scatter Plots ------------------------------ #

fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))

for i, ax in enumerate(axes3.flatten()):
    # Scatter plot: True vs Pred
    ax.scatter(Y_test_unscaled[:, i], Y_pred_unscaled[:, i], alpha=0.1, s=1, color='blue')
    
    # Perfect prediction line
    min_val = min(Y_test_unscaled[:, i].min(), Y_pred_unscaled[:, i].min())
    max_val = max(Y_test_unscaled[:, i].max(), Y_pred_unscaled[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    # R² for this dimension
    r2 = r2_score(Y_test_unscaled[:, i], Y_pred_unscaled[:, i])
    
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target_names[i]}\nR² = {r2:.4f}')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(control_panel.data_saving.target_dimension_scatter_plots)
plt.show()

### ------------------------------ Invariant Mass Calculation Using Awk Vectors ------------------------------ #

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

### ------------------------------ Plot Invariant Mass ------------------------------ #

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
plt.savefig(control_panel.data_saving.invariant_mass_plots)
plt.show()

### ------------------------------ Print Metrics for Invariant Mass ------------------------------ #

MSE_mass = mean_squared_error(M_true, M_pred)
RMSE_mass = np.sqrt(MSE_mass)
MAE_mass = mean_absolute_error(M_true, M_pred)
R2_mass = r2_score(M_true, M_pred)

M_pred_np = ak.to_numpy(M_pred)
M_true_np = ak.to_numpy(M_true)

kl_mass, std_mass, ci_mass, _ = bootstrap_kl_divergence(
    M_pred_np, M_true_np, n_bootstrap=1000, bins=100
)

print("\n" + "="*60)
print("INVARIANT MASS METRICS")
print("="*60)
print(f"MSE:  {MSE_mass:.4f}")
print(f"RMSE: {RMSE_mass:.4f} GeV")
print(f"MAE:  {MAE_mass:.4f} GeV")
print(f"R²:   {R2_mass:.4f}")
print(f"ttbar_mass: KL = {kl_mass:.4f} ± {std_mass:.4f}")
print("="*60)

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

results = np.column_stack([M_true, M_pred, M_pred - M_true])

np.savetxt(
    control_panel.data_saving.invariant_mass_data, 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)