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

N_inputs = dataset_train[0][0].shape[0]

with h5py.File("../train_inputs/feature_labels.h5", "r") as f:
    feature_names = f["Feature_labels"][:].astype(str)

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

### ------------------------------ Create Model Architecture ------------------------------ ###

import torch.nn.functional as F
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
            nn.Linear(Nhidden,8)
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

    for batch_x, batch_y, batch_m in train_loader:
        # Ensure batch is on GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_m = batch_m.to(device)

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
    scaler_M_mean = torch.tensor(f["M_mean"][:], device=device, dtype=torch.float32)
    scaler_M_scale = torch.tensor(f["M_scale"][:], device=device, dtype=torch.float32) 

# Load test targets directly from H5
with h5py.File("../train_inputs/larger_ttbar_test.h5", "r") as f:
    Y_test_scaled = f["Y"][:]

VelNet.eval()
Embedder.eval()

list_of_predictions = []
with torch.no_grad():
    for inputs, targets, _ in test_loader:
        inputs = inputs.to(device) 
        outputs = sample_flow_mean(VelNet, Embedder, inputs, n_steps=250)
        pred_mean = torch.mean(outputs,dim=1)

        list_of_predictions.append(pred_mean.cpu())

Y_pred = torch.concatenate(list_of_predictions).detach().cpu().numpy().flatten()

from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

# Compute and display metrics for each target feature

# Store target feature names in an array
target_names = ['top_px', 'top_py', 'top_pz', 
                'antitop_px', 'antitop_py', 'antitop_pz', 'top_E', 'antitop_E']

print("\n")
print("="*60)
print("TARGET FEATURE METRICS")
print("="*60)

epsilon = 1e-10
r2_per_dim = []
KLD_per_dim = []
for i in range(8):
    bins = np.linspace(0, max(np.max(Y_test_scaled[:, i]),np.max(Y_pred[:, i])), num=100) # Adjust range to your P_T spectrum
    p_counts, _ = np.histogram(Y_test_scaled[:, i], bins=bins)
    q_counts, _ = np.histogram(Y_pred[:, i], bins=bins)

    P = (p_counts + epsilon) / (np.sum(p_counts) + epsilon)
    Q = (q_counts + epsilon) /( np.sum(q_counts) + epsilon)

    KLD = np.sum(rel_entr(P, Q))
    KLD_per_dim.append(KLD)
    mse = mean_squared_error(Y_test_scaled[:, i], Y_pred[:, i])
    r2 = r2_score(Y_test_scaled[:, i], Y_pred[:, i])
    r2_per_dim.append(r2)
    mae = mean_absolute_error(Y_test_scaled[:,i], Y_pred[:,i])
    print(f"{target_names[i]}: MSE={mse:.4f}, R²={r2:.4f}, MAE = {mae:.4f}, KLD={KLD:.4f}\n")
print("="*60)

# Inverse transform
Y_pred_geV = Y_pred * scaler_Y_scale + scaler_Y_mean
Y_test_geV = Y_test_scaled * scaler_Y_scale + scaler_Y_mean

# ------------------------------ Plotting ------------------------------ #
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
