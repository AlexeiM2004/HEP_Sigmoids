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

save_prefix = "step_50_train_30"

print(f"Saving to {save_prefix} prefixed files")

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

class TransformerVelocityNet(nn.Module):
    def __init__(self, TimeEmbedder, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.TimeEmbedder = TimeEmbedder

        # Time and xt projections
        self.time_proj = nn.Linear(d_model, d_model)
        self.xt_proj = nn.Linear(1, d_model)

        # Project each feature group to d_model
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
    
    def forward(self, xt, t, c):

        t_emb = self.TimeEmbedder(t)
        t_token = self.time_proj(t_emb).unsqueeze(1)
        xt_token = self.xt_proj(xt).unsqueeze(1)


        leading_order_jet_features = c[:, 0:8]
        second_order_jet_features = c[:, 8:16]
        third_order_features = c[:, 16:24]
        fourth_order_features = c[:, 24:32]
        muon_features = c[:, 32:42]
        electron_features = c[:, 42:52]
        met_features = c[:, 52:54]
        

        leading_order_jet_token = self.leading_order_jet_proj(leading_order_jet_features).unsqueeze(1)
        second_order_jet_token = self.second_order_jet_proj(second_order_jet_features).unsqueeze(1)
        third_order_jet_token = self.third_order_jet_proj(third_order_features).unsqueeze(1)
        fourth_order_jet_token = self.fourth_order_jet_proj(fourth_order_features).unsqueeze(1)
        muon_token = self.muon_proj(muon_features).unsqueeze(1)
        electron_token = self.electron_proj(electron_features).unsqueeze(1)
        met_token = self.met_proj(met_features).unsqueeze(1)
        
        tokens = torch.cat([t_token, 
                            xt_token,
                            leading_order_jet_token,
                            second_order_jet_token,
                            third_order_jet_token,
                            fourth_order_jet_token,
                            muon_token, 
                            electron_token, 
                            met_token], dim=1)
        
        # Transformer
        tokens = self.transformer(tokens)

        xt_out = tokens[:, 1, :]
        
        return self.classifier(xt_out)

def conditional_flow_matching_loss(VelocityNet, X_train_batch, Y_train_batch, sigma_min=1e-4):
    y1 = Y_train_batch.unsqueeze(1)
    sample_batch_size = X_train_batch.shape[0]
    
    y0 = torch.randn_like(y1)
    t = torch.rand(sample_batch_size, 1, device=device)

    xt = (1 - t) * y0 + t * y1

    # Pass xt, t, and context (physics features)
    v_pred = VelocityNet(xt, t, X_train_batch)
    v_target = y1 - y0

    return ((v_pred - v_target) ** 2).mean()

embed_dim = 64
Sinusoidembed = SinusoidalPositionEmbeddings(dim=embed_dim).to(device)

nhead = 4
nlayer = 4

VelNet = TransformerVelocityNet(
    TimeEmbedder=Sinusoidembed,
    d_model=embed_dim, 
    nhead=nhead, 
    num_layers=nlayer, 
    dropout=dropout_rate
).to(device)

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

def sample_flow_mean(model, X_test, n_samples=50, n_steps=100, device=device):
    with torch.no_grad():
        B = X_test.shape[0]
        S = n_samples

        # Condition (B, 54) -> expand to (B*S, 54)
        c = X_test.to(device)
        c = c.unsqueeze(1).expand(B, S, c.shape[-1]).reshape(B * S, -1)

        x = torch.randn(B * S, 1, device=device)
        dt = 1.0 / n_steps

        for i in range (n_steps):
            t_val = i / n_steps
            t = torch.full((B * S, 1), t_val, device=device)

            v1 = model(x, t, c)
            
            x_half = x + 0.5 * dt * v1
            t_half = torch.full((B * S, 1), t_val + 0.5 * dt, device=device)
            
            v2 = model(x_half, t_half, c)

            x = x + dt * v2
        
        return x.reshape(B, S, 1)

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
optimiser = torch.optim.AdamW(VelNet.parameters(), 
    lr=learning_rate)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

### ------------------------------ Run Training Loop ------------------------------ ###

# Run a training loop

print("Beginning Training Loop")
print("="*60)

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch
times = [] # Keep track of times


N_epochs = 30 # Number of epochs we iterate over

for epoch in range(N_epochs):

    start_time = time.time()

    VelNet.train()

    epoch_train_loss = 0.0

    for batch_x,batch_y in train_loader:
        # Ensure batch is on GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimiser.zero_grad()

        # Forward pass and calculate losses
        train_loss = conditional_flow_matching_loss(VelNet, batch_x, batch_y)
        
        # Perform a backward pass + Optimisation
        train_loss.backward()

        optimiser.step()

        epoch_train_loss += train_loss.item() * batch_x.size(0)

    avg_train_loss = epoch_train_loss / len(train_loader.dataset)
    losses.append(avg_train_loss)

    epoch_time = time.time() - start_time
    times.append(epoch_time)

    VelNet.eval()

    epoch_val_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # Ensure batch is on GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            val_loss = conditional_flow_matching_loss(VelNet, batch_x, batch_y)

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

### ------------------------------ Save Model ------------------------------ ###
save_model = True
if save_model == True:
    # Combine models, optimizer, and architecture params into a single model dict
    model_dict = {
        'velnet_state_dict': VelNet.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'hyperparameters': {
            'embed_dim': embed_dim,
            'Nhead': nhead,
            'Nlayer': nlayer,
        }
    }

    torch.save(model_dict, f"../trained_models/{save_prefix}_direct_flowmatch_model.pt")
    print(f"Saved model to ../trained_models/{save_prefix}_direct_flowmatch_model.pt")

### ------------------------------ Evaluate Model ------------------------------ ###

# Load scaler info
with h5py.File("../train_inputs/larger_scaler_info.h5", "r") as f:
    scaler_Y_mean = f["Y_mean"][()]
    scaler_Y_scale = f["Y_scale"][()]

# Load test targets directly from H5
with h5py.File("../train_inputs/larger_ttbar_test.h5", "r") as f:
    Y_test_scaled = f["Y"][:]

VelNet.eval()

list_of_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device) 
        outputs = sample_flow_mean(VelNet, inputs, n_steps=50, n_samples=1)

        pred_mean = torch.mean(outputs,dim=1)

        list_of_predictions.append(pred_mean.cpu())

pred = torch.concatenate(list_of_predictions)
Y_pred = pred.detach().cpu().numpy().flatten()

from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance,energy_distance

MSE = mean_squared_error(Y_test_scaled, Y_pred)
RMS = root_mean_squared_error(Y_test_scaled,Y_pred)
MAE = mean_absolute_error(Y_test_scaled,Y_pred)
R2 = r2_score(Y_test_scaled,Y_pred)

# 1. Histogram the data to turn event regression into a PDF
bins = np.linspace(0, max(np.max(Y_test_scaled),np.max(Y_pred)), num=100) 
p_counts, _ = np.histogram(Y_test_scaled, bins=bins)
q_counts, _ = np.histogram(Y_pred, bins=bins)

# 2. Normalise
epsilon = 1e-10
P = (p_counts + epsilon) / np.sum(p_counts + epsilon)
Q = (q_counts + epsilon) / np.sum(q_counts + epsilon)

# 3. Calculate KL Divergence
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
ED = energy_distance(Y_test_geV, Y_pred_geV)

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
    f"KL Divergence: {KLD:.4f}\nWasserstein Distance: {WD:.4f}\nEnergy Distance: {ED:.4f}",
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
plt.savefig(f"../plots/{save_prefix}_ttbar_mass_flowtransform.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    f"../train_outputs/{save_prefix}_ttbar_mass_predictions_flowtransform.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

print(f"Saved predictions to ../train_outputs/{save_prefix}_ttbar_mass_predictions_flowtransform.txt")

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
print(f"Energy Distance: {ED:.4f}")