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
    def __init__(self, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Project each group to d_model
        self.jet_proj = nn.Linear(13, d_model) 
        self.muon_proj = nn.Linear(5, d_model)     
        self.electron_proj = nn.Linear(5, d_model) 
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
            nn.Linear(128, 1)
        )

        # ----------New Stuff----------
        self.norm = nn.LayerNorm(d_model)

        self.type_emb = nn.Embedding(4, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("j_idx", torch.tensor(0))
        self.register_buffer("mu_idx", torch.tensor(1))
        self.register_buffer("el_idx", torch.tensor(2))
        self.register_buffer("met_idx", torch.tensor(3))
    
    def pool(self, x):
        return x.mean(dim=1)
        
    def forward(self, x):
        # Split 126 features into groups
        jets = x[:, :104].reshape(-1, 8, 13)       # Automatically handles 2477 or 4096 events
        muons = x[:, 104:114].reshape(-1, 2, 5)    # Automatically handles batch dimension
        electrons = x[:, 114:124].reshape(-1, 2, 5)
        met = x[:, 124:126].reshape(-1, 1, 2)

        current_batch_size = jets.size(0)
        
        # Project each group to token and concatenate
        j_tokens = self.jet_proj(jets)
        mu_tokens = self.muon_proj(muons)
        el_tokens = self.electron_proj(electrons)
        met_tokens = self.met_proj(met)

        j_tokens += self.type_emb(self.j_idx)
        mu_tokens += self.type_emb(self.mu_idx)
        el_tokens += self.type_emb(self.el_idx)
        met_tokens += self.type_emb(self.met_idx)
        
        obj_tokens = torch.cat([j_tokens, mu_tokens, el_tokens, met_tokens], dim=1)
        
        obj_tokens = self.norm(obj_tokens)

        cls_tokens = self.cls_token.expand(current_batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, obj_tokens], dim=1)
        tokens = self.dropout(tokens)

        # Transformer
        transformer_out = self.transformer(tokens)
        
        # Global pooling
        cls_summary = transformer_out[:, 0]
        
        return self.classifier(cls_summary)

model = GroupedTransformer(d_model=64,nhead=4,num_layers=4,dropout=0.1).to(device)

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

N_epochs = 800 # Number of epochs we iterate over

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
plt.savefig("../plots/ttbar_Mass_embed.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    "../train_outputs/ttbar_mass_predictions_embed.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

print("Saved predictions to ../data/ttbar_mass_predictions_embed.txt")

print("---------------Metrics---------------")
print(f"Epochs: {N_epochs}")
print(f"Batch size: {batch_size}")
print(f"LR: {learning_rate}")
print(f"MSE in GeV: {MSE_GeV:.4f}")
print(f"RMSE in GeV: {RMS_GeV:.4f}")
print(f"MAE in GeV: {MAE_GeV:.4f}")
print(f"R^2 in GeV: {R2_GeV:.4f}")