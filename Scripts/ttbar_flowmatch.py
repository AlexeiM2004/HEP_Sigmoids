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

N_inputs = dataset_train[0][0].shape[0]

with h5py.File("../train_inputs/feature_labels.h5", "r") as f:
    feature_names = f["Feature_labels"][:].astype(str)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

### ------------------------------ Create Model Architecture ------------------------------ ###

import torch.nn.functional as F

dropout_rate = 0.02

import torch.nn as nn

class ContextEmbeddor(nn.Module):
    def __init__(self,Ninputs,Nembed,Nhidden=128):
        super(ContextEmbeddor,self).__init__()
        self.lin1 = nn.Linear(Ninputs,Nhidden)
        self.lin2 = nn.Linear(Nhidden,Nembed)
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.lin1(x))
        return self.lin2(x1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalVelocityNet(nn.Module):
    def __init__(self, Ninput, Ncontext, TimeEmbedder, Nhidden=128):
        super().__init__()
        self.TimeEmbedder = TimeEmbedder

        self.net = nn.Sequential(
            nn.Linear(Ninput + self.TimeEmbedder.dim + Ncontext, Nhidden),
            nn.GELU(),
            nn.Linear(Nhidden,Nhidden),
            nn.GELU(),
            nn.Linear(Nhidden,1)
        )
    
    def forward(self, x ,t ,c):
        t = self.TimeEmbedder(t)
        inp = torch.cat([x,t,c], dim=1)
        return self.net(inp)

def conditional_flow_matching_loss(VelocityNet, ContEmbedder, X_train_batch, Y_train_batch, sigma_min=1e-4):
    # Target Sampling
    y1 = Y_train_batch.unsqueeze(1)

    # Sample Batch
    sample_batch_size = X_train_batch.shape[0]
    y0 = torch.randn_like(y1)

    # Embed context
    c = ContEmbedder(X_train_batch)

    # Sample time t
    t = torch.rand(sample_batch_size,1,device=device)

    # Interpolate between x0 and x1 (enforced straight line interpolation)
    xt = (1 - (1 - sigma_min) * t) * y0 + t * y1

    v_pred = VelocityNet(xt,t,c)
    v_target = y1 - y0

    return ((v_pred - v_target) ** 2).mean()

embed_dim = 64

Embedder = ContextEmbeddor(Ninputs=N_inputs,Nembed=embed_dim).to(device)
Sinusoidembed = SinusoidalPositionEmbeddings(dim=embed_dim).to(device)

VelNet = ConditionalVelocityNet(
    Ninput=1, 
    Ncontext=embed_dim, 
    TimeEmbedder=Sinusoidembed,
    Nhidden=256).to(device)

print(Embedder)
print(VelNet)

### ------------------------------ Sampling Functions ------------------------------ ###

def sample_flow(model, embedder, X_test, n_steps=100, device=device):
    with torch.no_grad():
        B = X_test.shape[0]

        # Condition
        c = embedder(X_test.to(device))

        # Initial noise
        x = torch.randn(B, 1, device=device)

        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B, 1), i / n_steps, device=device)
            v = model(x, t, c)
            x = x + dt * v

        return x

def sample_flow_repeated(model, embedder, X_test, n_steps=100, n_samples=50, device='cuda'):
    all_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples = sample_flow(model, embedder, X_test, n_steps=n_steps, device=device)
            all_samples.append(samples)
        return all_samples

def sample_flow_mean(
    model,
    embedder,
    X_test,
    n_samples=50,
    n_steps=100,
    device=device
):
    with torch.no_grad():
        B = X_test.shape[0]
        S = n_samples

        # Condition (B, C)
        c = embedder(X_test.to(device))

        # Expand condition to (B, S, C)
        c = c.unsqueeze(1).expand(B,S,c.shape[-1])

        # Ininial noise (B,S,1)
        x  = torch.randn(B,S,1, device=device)

        dt = 1.0 / n_steps

        for i in range (n_steps):
            t_val = i / n_steps
            t = torch.full((B, S, 1), t_val / n_steps, device=device)

            # Flatten batch and predict initial velocity
            v1 = model(x.reshape(B*S, 1), t.reshape(B*S, 1), c.reshape(B*S, -1)).reshape(B, S, 1)
            
            # Predict half-step position and velocity
            x_half = x + 0.5 * dt * v1
            t_half = torch.full((B, S, 1), t_val + 0.5 * dt, device=device)
            v2 = model(x_half.reshape(B*S, 1), t_half.reshape(B*S, 1), c.reshape(B*S, -1)).reshape(B, S, 1)

            # Take the step using the midpoint velocity
            x = x + dt * v2
        
        return x # (B, S, 1)

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

### ------------------------------ Define optimiser and scheduler ------------------------------ ###

learning_rate = 0.001
optimiser = torch.optim.AdamW(list(VelNet.parameters()) + list(Embedder.parameters()), 
    lr=learning_rate
    )

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

### ------------------------------ Run Training Loop ------------------------------ ###

# Run a training loop

print("Beginning Training Loop")
print("="*60)

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch
times = [] # Keep track of times


N_epochs = 250 # Number of epochs we iterate over

for epoch in range(N_epochs):

    start_time = time.time()

    VelNet.train()
    Embedder.train()

    epoch_train_loss = 0.0

    for batch_x,batch_y in train_loader:
        # Ensure batch is on GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimiser.zero_grad()

        # Forward pass and calculate losses
        train_loss = conditional_flow_matching_loss(
            VelNet,
            Embedder,
            batch_x,
            batch_y
        )
        
        # Perform a backward pass + Optimisation
        train_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(VelNet.parameters()) + list(Embedder.parameters()), 
            max_norm=1.0 
            )

        optimiser.step()

        epoch_train_loss += train_loss.item() * batch_x.size(0)

    avg_train_loss = epoch_train_loss / len(train_loader.dataset)
    losses.append(avg_train_loss)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    VelNet.eval()
    Embedder.eval()

    epoch_val_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # Ensure batch is on GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            val_loss = conditional_flow_matching_loss(
            VelNet,
            Embedder,
            batch_x,
            batch_y
        )
            epoch_val_loss += val_loss.item() * batch_x.size(0)
    
    avg_val_loss = epoch_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # Call scheduler outside batch loop 
    scheduler.step(avg_val_loss)

    tot_secs = np.sum(times)
    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... Training
        print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s | Total Time: {tot_secs//60:.0f} minutes {tot_secs%60:.2f}s")
        print("="*60)

    final_epoch = epoch+1
    
    if epoch >= 50: # Starts the early stop loss checks after the warmup epochs 
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            print(f'Final Epoch before early stop [{epoch+1}/{N_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            break

    torch.cuda.empty_cache()
print("Training Complete")
print("="*60)

### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler info
with h5py.File("../train_inputs/larger_scaler_info.h5", "r") as f:
    scaler_Y_mean = f["Y_mean"][()]
    scaler_Y_scale = f["Y_scale"][()]

# Load test targets directly from H5
with h5py.File("../train_inputs/larger_ttbar_test.h5", "r") as f:
    Y_test_scaled = f["Y"][:]

VelNet.eval()
Embedder.eval()

list_of_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device) 
        outputs = sample_flow_mean(VelNet, Embedder, inputs, n_steps=250)
        #outputs = sample_flow_repeated(VelNet, Embedder, inputs, n_steps=250)
        pred_mean = torch.mean(outputs,dim=1)

        list_of_predictions.append(pred_mean.cpu())

pred = torch.concatenate(list_of_predictions)
Y_pred = pred.detach().cpu().numpy().flatten()

from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

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

# 4. Calculate KL Divergence
KLD = np.sum(rel_entr(P, Q))

# Inverse transform using saved scaler
Y_pred_geV = ((Y_pred * scaler_Y_scale) + scaler_Y_mean).flatten()
Y_test_geV = ((Y_test_scaled * scaler_Y_scale) + scaler_Y_mean).flatten()

correlation_matrix = np.corrcoef(Y_test_geV, Y_pred_geV)
MSE_GeV = mean_squared_error(Y_test_geV, Y_pred_geV)
RMS_GeV = root_mean_squared_error(Y_test_geV,Y_pred_geV)
MAE_GeV = mean_absolute_error(Y_test_geV,Y_pred_geV)
R2_GeV = r2_score(Y_test_geV,Y_pred_geV)
WD = wasserstein_distance(Y_test_geV, Y_pred_geV)

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
    f"Epochs: {final_epoch}/{N_epochs}\nBatch size: {batch_size}\nLR: {learning_rate}\nMSE in GeV: {MSE_GeV:.4f}\nRMSE in GeV: {RMS_GeV:.4f}\nMAE in GeV: {MAE_GeV:.4f}\nR^2 in GeV: {R2_GeV:.4f}\nPearson correlation coeff : {correlation_matrix[0, 1]:.4f}",
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
axes[1,0].set_title("Predicted Mass Distribution")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Text box with KL Divergence
axes[1,0].text(
    0.98, 0.98,
    f"KL Divergence: {KLD:.4f}\nWasserstein Distance: {WD:.4f}",
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
plt.savefig("../plots/ttbar_mass_flowmatch.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    "../train_outputs/ttbar_mass_predictions_flowmatch.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

print("Saved predictions to ../data/ttbar_mass_predictions_flowmatch.txt")

print("---------------Metrics---------------")
print(f"Epochs: {final_epoch}/{N_epochs}")
print(f"Batch size: {batch_size}")
print(f"LR: {learning_rate}")
print(f"MSE in GeV: {MSE_GeV:.4f}")
print(f"RMSE in GeV: {RMS_GeV:.4f}")
print(f"MAE in GeV: {MAE_GeV:.4f}")
print(f"R^2 in GeV: {R2_GeV:.4f}")
print(f"KL Divergence: {KLD:.4f}")
print(f"Pearson correlation coeff : {correlation_matrix[0, 1]:.4f}")
print(f"Wasserstein Distance: {WD:.4f}")