### ------------------------------ Imports ------------------------------ ###

import matplotlib.pyplot as plt 
import numpy as np
import torch
import h5py
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


### ------------------------------ Print Current Timestamp ------------------------------ ###

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Job started at :", formatted_time)

### ------------------------------ Device Usage ------------------------------ ###

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device with number of GPUs: {torch.cuda.device_count()}")

### ------------------------------ Load Yaml ------------------------------ ###

import yaml

# Load YAML configs
with open("config_data.yaml", "r") as f:
    config_data = yaml.safe_load(f)

with open("config_model.yaml", "r") as f:
    config_model = yaml.safe_load(f)

# ------------------------------ Print Configs Like Control Panel ------------------------------ #

def print_yaml_config(config, title):
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

print_yaml_config(config_data, "DATA CONFIGURATION")
print_yaml_config(config_model, "MODEL CONFIGURATION")

# Data Configuration

train_file = config_data["data"]["train_file"]
val_file = config_data["data"]["val_file"]
test_file = config_data["data"]["test_file"]
scaler_file = config_data["data"]["scaler_file"]

batch_size = config_data["data"]["batch_size"]
num_workers = config_data["data"]["num_workers"]
pin_memory = config_data["data"]["pin_memory"]
use_mass_loss = config_data["data"]["use_mass_loss"]

# Model Configuration

d_model = config_model["model"]["d_model"]
nhead = config_model["model"]["nhead"]
num_layers = config_model["model"]["num_layers"]
dropout = config_model["model"]["dropout"]

num_classifier_layers = config_model["model"]["num_classifier_layers"]
classifier_start_neurons = config_model["model"]["classifier_start_neurons"]
classifier_dropout = config_model["model"]["classifier_dropout"]
feature_groups = config_model["model"]["feature_groups"]

# Training Configuration

patience = config_model["training"]["patience"]
min_delta = config_model["training"]["min_delta"]
min_early_stop = config_model["training"]["min_early_stop"]

num_epochs = config_model["training"]["num_epochs"]
learning_rate = config_model["training"]["learning_rate"]
weight_decay = config_model["training"]["weight_decay"]

kl_weight_max = config_model["training"]["kl_weight_max"]
kl_ramp_epochs = config_model["training"]["kl_ramp_epochs"]
kl_bins = config_model["training"]["kl_bins"]
kl_sigma = config_model["training"]["kl_sigma"]
kl_eps = config_model["training"]["kl_eps"]

mass_loss_weight = config_model["training"]["mass_loss_weight"]

scheduler_factor = config_model["training"]["scheduler_factor"]
scheduler_patience = config_model["training"]["scheduler_patience"]
scheduler_min_lr = config_model["training"]["scheduler_min_lr"]

# Saving Configuration

loss_curve = config_data["saving"]["loss_curve"]
evaluation_results = config_data["saving"]["evaluation_results"]

### ------------------------------ Load Preprocessed Data ------------------------------ ###

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.X = torch.tensor(f["X"][:], dtype=torch.float32)
            self.Y = torch.tensor(f["Y"][:], dtype=torch.float32)
            if use_mass_loss == True:
                self.M = torch.tensor(f["M"][:], dtype=torch.float32)
            else:
                self.M = torch.zeros((len(self.X), 3), dtype=torch.float32) # Empty array of 0's same length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.M[idx]

# ------------------------------ Data Loaders ------------------------------ #

def create_loader(split):
    file_map = {
        'train': train_file,
        'val': val_file,
        'test': test_file
    }
    
    dataset = CustomDataset(file_map[split])
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

train_loader, val_loader, test_loader = [create_loader(s) for s in ['train', 'val', 'test']]

### ------------------------------ Model Architecture ------------------------------ ###

# Identify number of target dimensions
with h5py.File(train_file, "r") as f:
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
    def __init__(self, d_model, nhead, num_layers, dropout, num_classifier_layers, classifier_start_neurons, classifier_dropout, output_dim, feature_groups):
        super().__init__()

        self.feature_groups = feature_groups

        self.projections = nn.ModuleDict()
        for name, (start,end) in feature_groups.items():
            in_dim = end - start
            self.projections[name] = nn.Linear(in_dim, d_model)

        
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
        tokens = []
        for name, (start, end) in self.feature_groups.items():
            token = self.projections[name](x[:, start:end]).unsqueeze(1)
            tokens.append(token)

        # Concatenate all tokens
        tokens = torch.cat(tokens, dim=1)
        
        # Pass into Transformer
        tokens = self.transformer(tokens)
        
        # Attention pooling
        pooled = self.pool(tokens)
        
        return self.classifier(pooled)

model = Transformer(
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    dropout=dropout,
    num_classifier_layers=num_classifier_layers,
    classifier_start_neurons=classifier_start_neurons,
    classifier_dropout=classifier_dropout,
    output_dim=output_dim,
    feature_groups=feature_groups
).to(device)

### ------------------------------ Early stopping mechanism ------------------------------ ###

class EarlyStopping:
    def __init__(self):
        self.patience = patience # Number of epochs to wait
        self.min_delta = min_delta # Minimum change
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
            pred_kernel = torch.exp(-0.5 * ((pred_dim.unsqueeze(1) - centers.unsqueeze(0)) / kl_sigma) ** 2)
            target_kernel = torch.exp(-0.5 * ((target_dim_values.unsqueeze(1) - centers.unsqueeze(0)) / kl_sigma) ** 2)


            pred_hist = pred_kernel.mean(dim=0) + kl_eps
            target_hist = target_kernel.mean(dim=0) + kl_eps

            pred_hist = pred_hist / pred_hist.sum()
            target_hist = target_hist / target_hist.sum()

            kl_sum = kl_sum + torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)))

        return kl_sum / target_dim

### ------------------------------ Define loss func, optimiser, and scheduler ------------------------------ ###

# Loss functions
loss = nn.HuberLoss()
mass_loss_func = nn.HuberLoss()

# Optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Scheduler
scheduler = ReduceLROnPlateau(
    optimiser, 
    mode='min',
    factor=scheduler_factor,
    patience=scheduler_patience,
    min_lr=scheduler_min_lr
)

### ------------------------------ Run Training Loop ------------------------------ ###

# Load in mass scaling data
with h5py.File(scaler_file, "r") as f:
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
for epoch in range(num_epochs):
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
        
        y_pred = model(batch_x)
        
        # Huber loss 
        huber_loss = loss(y_pred, batch_y)
        
        # KL divergence loss
        hist_min = batch_y.min().item() - 0.25
        hist_max = batch_y.max().item() + 0.25
        kl_loss = distribution_considering_loss(
            y_pred,
            batch_y,
            bins=kl_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )

        if use_mass_loss == True: 
            batch_m = batch_m.to(device)
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
        else:
            mass_loss = torch.tensor(0.0, device=device)

        # Combined loss: Huber + KL + Mass loss
        total_loss = huber_loss + current_kl_weight * kl_loss + mass_loss * mass_loss_weight
        
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
                bins=kl_bins,
                hist_min=hist_min,
                hist_max=hist_max,
            )

            if use_mass_loss == True:

                batch_m = batch_m.to(device)
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
            else:
                mass_loss = torch.tensor(0.0, device=device)

            # Combined loss: Huber + KL + Mass loss
            total_loss = huber_loss + current_kl_weight * kl_loss + mass_loss * mass_loss_weight
            
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

    if (epoch + 1) % 1 == 0 and use_mass_loss == True:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} (KL: {avg_train_kl:.4f}) | Val Loss: {avg_val_loss:.4f} (KL: {avg_val_kl:.4f}) | Train - Val Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Mass Train Loss : {avg_train_mass:.4f} | Mass Val Loss : {avg_val_mass:.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} (KL: {avg_train_kl:.4f}) | Val Loss: {avg_val_loss:.4f} (KL: {avg_val_kl:.4f}) | Train - Val Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")


    if epoch >= min_early_stop:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler information
with h5py.File(scaler_file, "r") as f:
    scaler_Y_mean = f["Y_mean"][:]
    scaler_Y_scale = f["Y_scale"][:]

# Load test targets
with h5py.File(test_file, "r") as f:
    Y_test_scaled = f["Y"][:]

model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        list_of_predictions.append(outputs.cpu())

Y_pred = torch.cat(list_of_predictions).numpy()

# Unscale to GeV

Y_pred_unscaled = Y_pred * scaler_Y_scale + scaler_Y_mean
Y_test_unscaled = Y_test_scaled * scaler_Y_scale + scaler_Y_mean

# ------------------------------ Plot Loss Curve ------------------------------ #

fig1, ax = plt.subplots(1, 1, figsize=(16, 6))

# Loss curve
ax.plot(losses, label='Train Loss')
ax.plot(val_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(loss_curve)

print("Saved loss curve to", loss_curve)

# ------------------------------ Save Results for Analysis ------------------------------ #

with h5py.File(evaluation_results, "w") as f:
    f.create_dataset("Y_pred", data=Y_pred_unscaled)
    f.create_dataset("Y_test", data=Y_test_unscaled)
    f.create_dataset("Y_mean", data=scaler_Y_mean)
    f.create_dataset("Y_scale", data=scaler_Y_scale)

print("Saved evaluation results to", evaluation_results)


