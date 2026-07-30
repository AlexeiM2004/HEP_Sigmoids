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
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
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
    train_file : str = "kinematic_withM_spin_observable_features_train.h5"
    val_file : str = "kinematic_withM_spin_observable_features_val.h5"
    test_file : str = "kinematic_withM_spin_observable_features_test.h5"
    scaler_file : str = "kinematic_withM_spin_observable_features_scaler_info.h5"
    batch_size : int = 4096
    num_workers : int = 4
    pin_memory : bool = True

@dataclass
class Model_Configuration:
    d_model : int = 64
    nhead : int = 4
    num_layers : int = 4
    dropout : float = 0.1 
    num_classifier_layers: int = 3
    classifier_start_neurons: int = 128
    classifier_dropout: float = 0.1

@dataclass
class Training_Configuration:
    # Early stopping mechanism
    patience : int = 10
    min_delta : float = 0.0
    min_early_stop : int = 150

    # Training hyperparameters
    num_epochs : int = 10
    learning_rate : float = 0.0005
    weight_decay : float = 0.01

    # KL divergence settings
    kl_weight_max: float = 0.2
    kl_ramp_epochs: int = 15
    kl_bins: int = 100
    kl_sigma: float = 0.40
    kl_eps: float = 1e-8
    
    # Mass loss settings
    mass_loss_weight: float = 0.01

    # Scheduler settings
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6

@dataclass
class Data_Saving:
    loss_curve_r2_summary_plots : str = "MHA_train_no_mass_loss_loss_curve_r2_summary_plots.png"
    distribution_summary_plots : str = "MHA_train_no_mass_loss_distribution_summary_plots.png"
    resolution_summary_plots : str = "MHA_train_no_mass_loss_resolution_summary_plots.png"
    scatter_summary_plots : str = "MHA_train_no_mass_loss_scatter_summary_plots.png"
    spin_observables_plots : str = "MHA_train_no_mass_loss_spin_observables_plots.png"

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
            self.M = torch.tensor(f["M"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.M[idx]

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

# Identify number of target dimensions
with h5py.File(control_panel.data_config.train_file, "r") as f:
    Y_shape = f["Y"].shape
    output_dim = Y_shape[1] 

# Build MLP

def build_mlp(input_dim, output_dim, hidden_layers, start_neurons, dropout):
    layers = []
    prev_dim = input_dim
    for i in range(hidden_layers):
        hidden_dim = max(start_neurons // (2**i), 4)
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

# Define attention pooling mechanism 

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)  # Learn token importance
        
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, num_classifier_layers, classifier_start_neurons, classifier_dropout, output_dim):
        super().__init__()
        
        # Project each group to d_model
        self.leading_order_jet_proj = nn.Linear(8, d_model)
        self.second_order_jet_proj = nn.Linear(8, d_model)
        self.third_order_jet_proj = nn.Linear(8, d_model)
        self.fourth_order_jet_proj = nn.Linear(8, d_model)
        self.electron_proj = nn.Linear(10, d_model)
        self.muon_proj = nn.Linear(10, d_model)
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
        self.classifier = build_mlp(
            input_dim=d_model,
            output_dim=output_dim,
            hidden_layers=num_classifier_layers,
            start_neurons=classifier_start_neurons,
            dropout=classifier_dropout
        )
        
    def forward(self, x):
        # Split features into groups
        leading_order_jet_features = x[:, 0:8]
        second_order_jet_features = x[:, 8:16]
        third_order_features = x[:, 16:24]
        fourth_order_features = x[:, 24:32]
        electron_features = x[:, 32:42]
        muon_features = x[:, 42:52]
        met_features = x[:, 52:54]
        
        # Project each group to token and concatenate
        leading_order_jet_token = self.leading_order_jet_proj(leading_order_jet_features).unsqueeze(1)
        second_order_jet_token = self.second_order_jet_proj(second_order_jet_features).unsqueeze(1)
        third_order_jet_token = self.third_order_jet_proj(third_order_features).unsqueeze(1)
        fourth_order_jet_token = self.fourth_order_jet_proj(fourth_order_features).unsqueeze(1)

        muon_token = self.muon_proj(muon_features).unsqueeze(1)
        met_token = self.met_proj(met_features).unsqueeze(1)
        electron_token = self.electron_proj(electron_features).unsqueeze(1)

        
        tokens = torch.cat([leading_order_jet_token,
                            second_order_jet_token,
                            third_order_jet_token,
                            fourth_order_jet_token,
                            electron_token,
                            muon_token,
                            met_token], dim=1)
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Attention pooling
        pooled = self.pool(tokens)
        
        return self.classifier(pooled)

model = Transformer(
    d_model=control_panel.model_config.d_model,
    nhead=control_panel.model_config.nhead,
    num_layers=control_panel.model_config.num_layers,
    dropout=control_panel.model_config.dropout,
    num_classifier_layers=control_panel.model_config.num_classifier_layers,
    classifier_start_neurons=control_panel.model_config.classifier_start_neurons,
    classifier_dropout=control_panel.model_config.classifier_dropout,
    output_dim=output_dim
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
        # Reshape target matrix
        target_dim = pred.shape[1]
        if isinstance(hist_min, (float, int)):
            hist_min = pred.new_tensor([hist_min] * target_dim)
        if isinstance(hist_max, (float, int)):
            hist_max = pred.new_tensor([hist_max] * target_dim)

        # Define max and min histogram
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

            # Define Gaussian kernels
            pred_kernel = torch.exp(-0.5 * ((pred_dim.unsqueeze(1) - centers.unsqueeze(0)) / control_panel.train_config.kl_sigma) ** 2)
            target_kernel = torch.exp(-0.5 * ((target_dim_values.unsqueeze(1) - centers.unsqueeze(0)) / control_panel.train_config.kl_sigma) ** 2)


            pred_hist = pred_kernel.mean(dim=0) + control_panel.train_config.kl_eps
            target_hist = target_kernel.mean(dim=0) + control_panel.train_config.kl_eps

            pred_hist = pred_hist / pred_hist.sum()
            target_hist = target_hist / target_hist.sum()

            kl_sum = kl_sum + torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)))

        return kl_sum / target_dim

### ------------------------------ Define loss func, optimiser, and scheduler ------------------------------ ###

# Loss functions
loss = nn.HuberLoss()
mass_loss_func = nn.HuberLoss()

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
with h5py.File(control_panel.data_config.scaler_file, "r") as f:
    scaler_Y_mean = torch.tensor(f["Y_mean"][:], device=device, dtype=torch.float32)
    scaler_Y_scale = torch.tensor(f["Y_scale"][:], device=device, dtype=torch.float32)
    
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
    epoch_train_mass = 0.0

    # Ramp up KL weight
    current_kl_weight = control_panel.train_config.kl_weight_max * min(1.0, (epoch + 1) / control_panel.train_config.kl_ramp_epochs)

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
            bins=control_panel.train_config.kl_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )

        # Unscale y_pred to reconstruct predicted invariant masses
        y_pred_unscaled = y_pred * scaler_Y_scale + scaler_Y_mean

        top_px_pred, top_py_pred, top_pz_pred = y_pred_unscaled[:, 0], y_pred_unscaled[:, 1], y_pred_unscaled[:, 2]
        top_E_pred = y_pred_unscaled[:, 3]

        antitop_px_pred, antitop_py_pred, antitop_pz_pred = y_pred_unscaled[:, 4], y_pred_unscaled[:, 5], y_pred_unscaled[:, 6]
        antitop_E_pred = y_pred_unscaled[:, 7]

        # Predicted top and antitop invariant masses
        top_m_pred = torch.sqrt(torch.clamp(top_E_pred**2 - (top_px_pred**2 + top_py_pred**2 + top_pz_pred**2), min=1e-6))
        antitop_m_pred = torch.sqrt(torch.clamp(antitop_E_pred**2 - (antitop_px_pred**2 + antitop_py_pred**2 + antitop_pz_pred**2), min=1e-6))

        # Predicted ttbar system invariant mass
        ttbar_px_pred = top_px_pred + antitop_px_pred
        ttbar_py_pred = top_py_pred + antitop_py_pred
        ttbar_pz_pred = top_pz_pred + antitop_pz_pred
        ttbar_E_pred  = top_E_pred + antitop_E_pred

        m_ttbar_pred = torch.sqrt(torch.clamp(ttbar_E_pred**2 - (ttbar_px_pred**2 + ttbar_py_pred**2 + ttbar_pz_pred**2), min=1e-6))

        # Calculate individual mass loss (top: batch_m[:, 0], antitop: batch_m[:, 1])
        individual_mass_loss = mass_loss_func(top_m_pred, batch_m[:, 0]) + mass_loss_func(antitop_m_pred, batch_m[:, 1])

        # Calculate system mass loss directly using batch_m[:, 2] (ttbar mass)
        system_mass_loss = mass_loss_func(m_ttbar_pred, batch_m[:, 2])

        # Sum of both constraints
        mass_loss = (individual_mass_loss + system_mass_loss) / 200

        # Combined loss: Huber + KL + Mass loss
        total_loss = huber_loss + current_kl_weight * kl_loss + mass_loss * control_panel.train_config.mass_loss_weight
        
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
                bins=control_panel.train_config.kl_bins,
                hist_min=hist_min,
                hist_max=hist_max,
            )

            # Unscale y_pred to reconstruct predicted invariant masses
            y_pred_unscaled = y_pred * scaler_Y_scale + scaler_Y_mean

            top_px_pred, top_py_pred, top_pz_pred = y_pred_unscaled[:, 0], y_pred_unscaled[:, 1], y_pred_unscaled[:, 2]
            top_E_pred = y_pred_unscaled[:, 3]

            antitop_px_pred, antitop_py_pred, antitop_pz_pred = y_pred_unscaled[:, 4], y_pred_unscaled[:, 5], y_pred_unscaled[:, 6]
            antitop_E_pred = y_pred_unscaled[:, 7]

            # Predicted top and antitop invariant masses
            top_m_pred = torch.sqrt(torch.clamp(top_E_pred**2 - (top_px_pred**2 + top_py_pred**2 + top_pz_pred**2), min=1e-6))
            antitop_m_pred = torch.sqrt(torch.clamp(antitop_E_pred**2 - (antitop_px_pred**2 + antitop_py_pred**2 + antitop_pz_pred**2), min=1e-6))

            # Predicted ttbar system invariant mass
            ttbar_px_pred = top_px_pred + antitop_px_pred
            ttbar_py_pred = top_py_pred + antitop_py_pred
            ttbar_pz_pred = top_pz_pred + antitop_pz_pred
            ttbar_E_pred  = top_E_pred + antitop_E_pred

            m_ttbar_pred = torch.sqrt(torch.clamp(ttbar_E_pred**2 - (ttbar_px_pred**2 + ttbar_py_pred**2 + ttbar_pz_pred**2), min=1e-6))

            # Calculate individual mass loss (top: batch_m[:, 0], antitop: batch_m[:, 1])
            individual_mass_loss = mass_loss_func(top_m_pred, batch_m[:, 0]) + mass_loss_func(antitop_m_pred, batch_m[:, 1])

            # Calculate system mass loss directly using batch_m[:, 2] (ttbar mass)
            system_mass_loss = mass_loss_func(m_ttbar_pred, batch_m[:, 2])

            # Sum of both constraints
            mass_loss = (individual_mass_loss + system_mass_loss) / 200

            # Combined loss: Huber + KL + Mass loss
            total_loss = huber_loss + current_kl_weight * kl_loss + mass_loss * control_panel.train_config.mass_loss_weight
            
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

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{control_panel.train_config.num_epochs} | Train Loss: {avg_train_loss:.4f} (KL: {avg_train_kl:.4f}) | Val Loss: {avg_val_loss:.4f} (KL: {avg_val_kl:.4f}) | Train - Val Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Mass Train Loss : {avg_train_mass:.4f} | Mass Val Loss : {avg_val_mass:.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")

    if epoch >= control_panel.train_config.min_early_stop:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler information
with h5py.File(control_panel.data_config.scaler_file, "r") as f:
    scaler_Y_mean = f["Y_mean"][:]
    scaler_Y_scale = f["Y_scale"][:]

# Load test targets
with h5py.File(control_panel.data_config.test_file, "r") as f:
    Y_test_scaled = f["Y"][:]

model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        list_of_predictions.append(outputs.cpu())

Y_pred = torch.cat(list_of_predictions).numpy()

# Unscale from 0-1 into GeV
Y_pred_unscaled = Y_pred * scaler_Y_scale + scaler_Y_mean
Y_test_unscaled = Y_test_scaled * scaler_Y_scale + scaler_Y_mean

### ------------------------------------------------------------------------------------------------------------------------------- ###
#                                                     SPLIT TRAIN FILE HERE                                                           #
### ------------------------------------------------------------------------------------------------------------------------------- ###

### ------------------------------ Calculate Metrics ------------------------------ ###

def compute_distribution_metrics(y_pred, y_true, bins=100):
    # Flatten to 1D numpy arrays
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    
    # MAE, MSE, RMSE, R^2
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # SciPy 1D Wasserstein Distance
    wd = wasserstein_distance(y_pred, y_true)
    
    # SciPy KL Divergence
    min_val = min(np.min(y_pred), np.min(y_true))
    max_val = max(np.max(y_pred), np.max(y_true))
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Compute histogram counts
    p_counts, _ = np.histogram(y_pred, bins=bin_edges)
    q_counts, _ = np.histogram(y_true, bins=bin_edges)
    
    # Convert counts to probabilities with epsilon smoothing to prevent log(0)
    eps = 1e-10
    p_prob = (p_counts + eps) / np.sum(p_counts + eps)
    q_prob = (q_counts + eps) / np.sum(q_counts + eps)
    
    # SciPy KL Divergence D_KL(P || Q)
    kl_div = entropy(p_prob, q_prob)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'Wasserstein': wd,
        'KL': kl_div
    }

def mahalanobis_distance(y_pred, y_true):
    # Mean and covariance of the true distribution
    mu_true = np.mean(y_true, axis=0)
    cov_true = np.cov(y_true, rowvar=False)
    # Add small regularization to avoid singular matrix
    cov_true += 1e-6 * np.eye(cov_true.shape[0])
    inv_cov = inv(cov_true)
    
    # Compute distance for each event
    distances = np.array([mahalanobis(p, mu_true, inv_cov) for p in y_pred])
    return distances

def mig_matrix(m_reco,m_truth, bins):
    # import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1. Define Binning
    # Example: 0 to 3000 GeV with 50 bins
    # bins = np.linspace(0, 500, 10) 

    # 2. Compute Histogram
    # histogram2d(x, y) results in H[i,j] where i is x-bin and j is y-bin
    # To have Reco on X and Truth on Y:
    H, xedges, yedges = np.histogram2d(m_reco, m_truth, bins=bins)

    # 3. Normalize by Truth (the Y-axis bins)
    # In the H matrix from histogram2d, summing over axis 0 (Reco) 
    # gives the total counts for each Truth bin (axis 1).
    truth_bin_counts = np.sum(H, axis=0)
    print(f"Truth bin counts: {truth_bin_counts}")

    # Avoid division by zero for empty truth bins
    truth_bin_counts[truth_bin_counts == 0] = 1

    # Normalize: divide each 'reco' column by the total 'truth' counts for that truth bin
    for i in range(H.shape[1]):
        H[:, i] /= truth_bin_counts[i]
        
    migration_matrix = H
    
    # Print the trace diagonal divided by the total sum
    diagonal_sum = np.trace(migration_matrix)
    total_sum = np.sum(migration_matrix)
    print(f"{diagonal_sum/total_sum:.2f}")


    # 4. Plotting
    plt.figure(figsize=(4, 3))
    # Use pcolormesh. Note: We transpose (.T) because pcolormesh expects 
    # the array shape to match (len(y), len(x))
    plt.pcolormesh(xedges, yedges, migration_matrix.T, cmap='viridis', shading='auto')

    plt.colorbar(label='P(Reco | Truth)')
    # plt.xlabel('Top $p_{T}$ [GeV]')
    # plt.ylabel('Antitop $p_{T}$ [GeV]')
    plt.title('Migration Matrix (Normalized by Truth Row)')

    # Perfect reconstruction line
    plt.plot([bins[0], bins[-1]], [bins[0], bins[-1]], color='white', linestyle='--', alpha=0.6)
    
    # Print the bin counts in each bin 
    plt.tight_layout()
    return plt

### ------------------------------ Display Metrics ------------------------------ ###

# Store target feature names in an array
target_names = ['top_px', 'top_py', 'top_pz', 'top_E',
                 'antitop_px', 'antitop_py', 'antitop_pz', 'antitop_E']

# Calculate MAE, RMSE, MSE, R2, Wasserstein distance, and KL divergence

print("\n")
print("="*60)
print("TARGET FEATURE METRICS")
print("="*60)

r2_per_dim = []
for i, feature in enumerate(target_names):
    res = compute_distribution_metrics(
        y_pred=Y_pred_unscaled[:, i], 
        y_true=Y_test_unscaled[:, i], 
        bins=100
    )
    r2_per_dim.append(res['R2'])
    print(f"Feature : {feature} | MAE : {res['MAE']:.4f} | RMSE : {res['RMSE']:.4f} | MSE : {res['MSE']:.4f} | R^2 : {res['R2']:.4f} | Wasserstein : {res['Wasserstein']:.4f} | KL Div : {res['KL']:.4f}")

print(f"Mahalanobis Distance for T and TBar : {np.mean(mahalanobis_distance(Y_pred_unscaled[:, 0:8], Y_test_unscaled[:, 0:8])):.3f}")
print(f"Mahalanobis Distance for T only : {np.mean(mahalanobis_distance(Y_pred_unscaled[:, 0:4], Y_test_unscaled[:, 0:4])):.3f}")
print(f"Mahalanobis Distance for TBar only : {np.mean(mahalanobis_distance(Y_pred_unscaled[:, 4:8], Y_test_unscaled[:, 4:8])):.3f}")

### ------------------------------ Plot Loss Curve and R^2 Plots ------------------------------ #

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))

# Loss curve
axes1[0].plot(losses, label='Train Loss')
axes1[0].plot(val_losses, label='Validation Loss')
axes1[0].set_xlabel('Epoch')
axes1[0].set_ylabel('Loss')
axes1[0].set_title('Training and Validation Loss')
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)

# R² bar chart (16 bars)
bars = axes1[1].bar(range(8), r2_per_dim, color='blue', edgecolor='black')
axes1[1].set_xticks(range(8))
axes1[1].set_xticklabels(target_names, rotation=90, ha='center')
axes1[1].set_ylabel('R-Squared')
axes1[1].set_title('R-Squared per Feature')
axes1[1].set_ylim([-0.1, 1.0])
axes1[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_per_dim):
    axes1[1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(control_panel.data_saving.loss_curve_r2_summary_plots)

### ------------------------------ Plot Target Feature Distribution ------------------------------ #

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 12))

for i in range(8):
    row = i // 4
    col = i % 4
    ax = axes2[row, col]
    
    ax.hist(Y_test_unscaled[:, i], bins=100, density=True, histtype='step',
            label='True', color='blue', linewidth=1.5)
    ax.hist(Y_pred_unscaled[:, i], bins=100, density=True, histtype='step',
            label='Pred', color='red', linewidth=1.5)
    ax.set_title(target_names[i], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(control_panel.data_saving.distribution_summary_plots)

### ------------------------------ Plot Target Feature Resolution Plots ------------------------------ #

fig3, axes3 = plt.subplots(2, 4, figsize=(16, 12))

for i in range(8):
    row = i // 4
    col = i % 4
    ax = axes3[row, col]
    
    residuals = Y_pred_unscaled[:, i] - Y_test_unscaled[:, i]
    ax.hist(residuals, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect')
    ax.axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=2,
               label=f'μ={np.mean(residuals):.2f}')
    ax.set_title(target_names[i], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(control_panel.data_saving.resolution_summary_plots)

### ------------------------------ Plot Target Feature Scatter Plots ------------------------------ #

fig4, axes4 = plt.subplots(2, 4, figsize=(16, 12))

for i in range(8):
    row = i // 4
    col = i % 4
    ax = axes4[row, col]
    
    ax.scatter(Y_test_unscaled[:, i], Y_pred_unscaled[:, i],
               alpha=0.1, s=1, color='blue')
    min_val = min(Y_test_unscaled[:, i].min(), Y_pred_unscaled[:, i].min())
    max_val = max(Y_test_unscaled[:, i].max(), Y_pred_unscaled[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True', fontsize=8)
    ax.set_ylabel('Pred', fontsize=8)
    r2 = r2_per_dim[i]
    ax.set_title(f'{target_names[i]}\nR² = {r2:.3f}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(control_panel.data_saving.scatter_summary_plots)

### ------------------------------ Spin Observable Calculation Functions ------------------------------ #


def boost(top, anti_top, leptonP, leptonM, top_mass_true, antitop_mass_true):

    # Build ttbar system
    ttbar = top + anti_top 

    # Boost tops lab --> ttbar_CoM
    top_in_CoM      = top.boostCM_of(ttbar)
    antitop_in_CoM  = anti_top.boostCM_of(ttbar) 

    # Apply on shell correction
    #px, py, pz = top_in_CoM.px, top_in_CoM.py, top_in_CoM.pz
    #p_norm = np.sqrt(px**2 + py**2 + pz**2)
    #E_corrected = np.sqrt(p_norm**2 + top_mass_true**2)
    #top_in_CoM_corrected = vector.zip({
    #    'px': px,
    #    'py': py,
    #    'pz': pz,
    #    'E': E_corrected})

    #apx, apy, apz = antitop_in_CoM.px, antitop_in_CoM.py, antitop_in_CoM.pz
    #ap_norm = np.sqrt(apx**2 + apy**2 + apz**2)
    #anti_E_corrected = np.sqrt(ap_norm**2 + antitop_mass_true**2)
    #antitop_in_CoM_corrected = vector.zip({
    #    'px': apx,
    #    'py': apy,
    #    'pz': apz,
    #    'E': anti_E_corrected})
    
    # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
    leptonP_in_CoM     = leptonP.boostCM_of(ttbar)
    leptonM_in_CoM     = leptonM.boostCM_of(ttbar)
    leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
    leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

    # Compute the unit 3-vector directions
    Kdirection        = top_in_CoM.to_beta3().unit()
    leptonP_direction = leptonP_in_top.to_beta3().unit()
    leptonM_direction = leptonM_in_antitop.to_beta3().unit()

    return Kdirection, leptonP_direction, leptonM_direction, ttbar, top_in_CoM, antitop_in_CoM, leptonP_in_CoM, leptonM_in_CoM, leptonP_in_top, leptonM_in_antitop


def helicity_basis_observables(Kdirection, leptonP_direction, leptonM_direction, ttbar, top_in_CoM_corrected, antitop_in_CoM_corrected, leptonP_in_CoM, leptonM_in_CoM, leptonP_in_top, leptonM_in_antitop):

    # Kinematics
    z     = vector.obj(x=0,y=0,z=1)
    cos_T = z.dot(Kdirection)
    sin_T = (1 - cos_T**2)**0.5
    mask = 1*(cos_T > 0) -1*(cos_T < 0)

    # Helicity basis observables
    cos_K_plus  = Kdirection.dot(leptonP_direction)
    cos_K_minus = -Kdirection.dot(leptonM_direction) 

    Ndirection = z.cross(Kdirection)/sin_T
    cos_N_plus  = mask*Ndirection.dot(leptonP_direction)
    cos_N_minus = -mask*Ndirection.dot(leptonM_direction) 

    Rdirection  =  1/sin_T * (z - cos_T*Kdirection)
    cos_R_plus  = mask*Rdirection.dot(leptonP_direction)
    cos_R_minus = -mask*Rdirection.dot(leptonM_direction) 

    cos_phi     = leptonP_direction.dot(leptonM_direction)

    helicity_observables = ak.zip({
    "cos_K_plus"  :  cos_K_plus,
    "cos_K_minus" :  cos_K_minus,
    "cos_N_plus"  :  cos_N_plus,
    "cos_N_minus" :  cos_N_minus,
    "cos_R_plus"  :  cos_R_plus,
    "cos_R_minus" :  cos_R_minus,
    "cos_phi"     :  cos_phi,
    "cos_theta_star" : cos_T 
    }, depth_limit=1, with_name="Event")

    return helicity_observables

### ------------------------------ Load RECO (pred) Leptons ------------------------------ ###

with h5py.File(control_panel.data_config.test_file, "r") as f:
    R_test = f["R"][:]

leptonP_reco = vector.Array(
    ak.zip({
        'px': R_test[:, 0],
        'py': R_test[:, 1],
        'pz': R_test[:, 2],
        'E': R_test[:, 3]
    })
)

leptonM_reco = vector.Array(
    ak.zip({
        'px': R_test[:, 4],
        'py': R_test[:, 5],
        'pz': R_test[:, 6],
        'E': R_test[:, 7]
    })
)

### ------------------------------ Load Parton (true) Leptons ------------------------------ ###

with h5py.File(control_panel.data_config.test_file, "r") as f:
    L_test = f["L"][:]   # shape (n_events, 8)

leptonP_truth = vector.Array(
    ak.zip({
        'px': L_test[:, 0],
        'py': L_test[:, 1],
        'pz': L_test[:, 2],
        'E': L_test[:, 3]
    })
)
leptonM_truth = vector.Array(
    ak.zip({
        'px': L_test[:, 4],
        'py': L_test[:, 5],
        'pz': L_test[:, 6],
        'E': L_test[:, 7]
    })
)

### ------------------------------ Spin Observable Calculation ------------------------------ #

# Convert NumPy to Awkward arrays
Y_pred_awk = ak.from_numpy(Y_pred_unscaled)
Y_test_awk = ak.from_numpy(Y_test_unscaled)

# Predicted vectors
top_pred = vector.Array(
    ak.zip({
        'px': Y_pred_awk[:, 0],
        'py': Y_pred_awk[:, 1],
        'pz': Y_pred_awk[:, 2],
        'E': Y_pred_awk[:,3]
    })
)

antitop_pred = vector.Array(
    ak.zip({
        'px': Y_pred_awk[:, 4],
        'py': Y_pred_awk[:, 5],
        'pz': Y_pred_awk[:, 6],
        'E': Y_pred_awk[:,7]
    })
)

# True vectors
top_true = vector.Array(
    ak.zip({
        'px': Y_test_awk[:, 0],
        'py': Y_test_awk[:, 1],
        'pz': Y_test_awk[:, 2],
        'E': Y_test_awk[:,3]
    })
)

antitop_true = vector.Array(
    ak.zip({
        'px': Y_test_awk[:, 4],
        'py': Y_test_awk[:, 5],
        'pz': Y_test_awk[:, 6],
        'E': Y_test_awk[:,7]
    })
)

### ------------------------------ Calculate Spin Observable Metrics ------------------------------ #

# Calculate Spin Observables
obs_fields = ['cos_K_plus','cos_K_minus','cos_N_plus','cos_N_minus','cos_R_plus','cos_R_minus','cos_phi','cos_theta_star']

# Load test masses
with h5py.File(control_panel.data_config.test_file, "r") as f:
    M_test = f["M"][:]

dic_of_spin_observables_pred = helicity_basis_observables(*boost(top_pred, antitop_pred, leptonP_reco, leptonM_reco, M_test[:, 0], M_test[:, 1]))
dic_of_spin_observables_true = helicity_basis_observables(*boost(top_true, antitop_true, leptonP_truth, leptonM_truth, M_test[:, 0], M_test[:, 1]))

pred_spin_observables = np.column_stack([ak.to_numpy(dic_of_spin_observables_pred[field]) for field in obs_fields])
true_spin_observables = np.column_stack([ak.to_numpy(dic_of_spin_observables_true[field]) for field in obs_fields])

# Compute metrics
print("\n" + "="*60)
print("SPIN OBSERVABLE METRICS")
print("="*60)

r2_spin = []
for i, feature in enumerate(obs_fields):
    res = compute_distribution_metrics(
        y_pred=pred_spin_observables[:, i], 
        y_true=true_spin_observables[:, i], 
        bins=100
    )
    r2_spin.append(res['R2'])
    print(f"Feature : {feature} | MAE : {res['MAE']:.4f} | RMSE : {res['RMSE']:.4f} | MSE : {res['MSE']:.4f} | R^2 : {res['R2']:.4f} | Wasserstein : {res['Wasserstein']:.4f} | KL Div : {res['KL']:.4f}")


### ------------------------------ Plot Spin Observable Metrics ------------------------------ #

fig, axes = plt.subplots(8, 3, figsize=(15, 25))  # 8 rows, 3 columns

for i, name in enumerate(obs_fields):
    # Distribution (true vs pred)
    ax0 = axes[i, 0]
    ax0.hist(true_spin_observables[:, i], bins=100, density=True,
             histtype='step', label='True', color='blue', linewidth=1.5)
    ax0.hist(pred_spin_observables[:, i], bins=100, density=True,
             histtype='step', label='Pred', color='red', linewidth=1.5)
    ax0.set_title(f'{name} Distribution', fontsize=10)
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Scatter (true vs pred)
    ax1 = axes[i, 1]
    ax1.scatter(true_spin_observables[:, i], pred_spin_observables[:, i],
                alpha=0.1, s=1, color='blue')
    min_val = min(true_spin_observables[:, i].min(), pred_spin_observables[:, i].min())
    max_val = max(true_spin_observables[:, i].max(), pred_spin_observables[:, i].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('True', fontsize=8)
    ax1.set_ylabel('Pred', fontsize=8)
    ax1.set_title(f'{name}\nR² = {r2_spin[i]:.3f}', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Resolution (pred - true)
    ax2 = axes[i, 2]
    residuals = pred_spin_observables[:, i] - true_spin_observables[:, i]
    ax2.hist(residuals, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect')
    ax2.axvline(np.mean(residuals), color='blue', linestyle='-', linewidth=2,
                label=f'μ={np.mean(residuals):.3f}')
    ax2.set_title(f'{name} Resolution', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(control_panel.data_saving.spin_observables_plots)
plt.show()
