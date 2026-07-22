### ------------------------------ Imports ------------------------------ ###

import uproot # Read in ROOT file format 
import awkward as ak # Used to perform awkward operations on jagged arrays
import matplotlib.pyplot as plt # Used to plot graphs 
import os
import numpy as np
import torch

### ------------------------------ Device Usage ------------------------------ ###

print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(f"Number of GPUs: {torch.cuda.device_count()}")

### ------------------------------ File Download ------------------------------ ###

filename = "ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root"

size = os.path.getsize("ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root")
print(f"File size: {size / (1024**3):.2f} GB")

### ------------------------------ Data Preperation ------------------------------ ###

# Open main file
file_ttbar = uproot.open("ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root")

# Access reco branch
tree = file_ttbar["reco"]
print(tree.keys())

# Get branch names from TTree
all_branches = tree.keys()

# Define column groups
jet_cols = [col for col in all_branches if col.startswith('jet_')]
el_cols = [col for col in all_branches if col.startswith('el_')]
mu_cols = [col for col in all_branches if col.startswith('mu_')]
met_cols = [col for col in all_branches if col.startswith('met_')]

print(f"Jet columns: {len(jet_cols)}")
print(f"Electron columns: {len(el_cols)}")
print(f"Muon columns: {len(mu_cols)}")
print(f"Met columns: {len(met_cols)}")

# Load needed columns into awkward array
needed_cols = jet_cols + el_cols + mu_cols + met_cols + ['parton_ttbar_m']
tree = tree.arrays(needed_cols)

# Access ttbar_m column
ttbar_mass = tree["parton_ttbar_m"]

# Pad and fill all data to their respective max
jet_cols_padded_filled = []
for i in range(8):
    padded_jet = ak.pad_none(tree[jet_cols[i]], 8, clip=True)
    filled_jet = ak.fill_none(padded_jet, 0.0)
    jet_cols_padded_filled.append(filled_jet)

el_cols_padded_filled = []
mu_cols_padded_filled = []
for i in range(5):
    padded_el = ak.pad_none(tree[el_cols[i]], 2, clip=True)
    padded_mu = ak.pad_none(tree[mu_cols[i]], 2, clip=True)
    filled_el = ak.fill_none(padded_el, 0.0)
    filled_mu = ak.fill_none(padded_mu, 0.0)
    el_cols_padded_filled.append(filled_el)
    mu_cols_padded_filled.append(filled_mu)

# Stack input columns individually from each array
X_jets = np.column_stack(jet_cols_padded_filled)
X_el = np.column_stack(el_cols_padded_filled)
X_mu = np.column_stack(mu_cols_padded_filled)
X_met = np.column_stack([tree['met_met_NOSYS'],tree['met_phi_NOSYS']])

# Concatenate X data
X = np.concatenate([X_jets, X_mu, X_el, X_met], axis=1)
target = np.array(ttbar_mass)

# Delete large objects to free memory
del tree, ttbar_mass, jet_cols_padded_filled, el_cols_padded_filled, mu_cols_padded_filled, met_cols
del X_jets, X_el, X_mu, X_met

### ------------------------------ Split into train/validation/test (80:10:10) ------------------------------ ###

from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
X_train, X_temp, Y_train, Y_temp = train_test_split(X, target, test_size=0.2, random_state=42)

# Second split: 10% validation, 10% test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

### ------------------------------ Scale the data ------------------------------ ###

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()


### ------------------------------ Convert Into Tensors ------------------------------ ###

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

Y_train_tensor = torch.tensor(Y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
Y_val_tensor = torch.tensor(Y_val_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
Y_test_tensor = torch.tensor(Y_test_scaled.reshape(-1, 1), dtype=torch.float32, device=device)

print(X_train_tensor.size())
print(X_val_tensor.size())
print(X_test_tensor.size())

print(Y_train_tensor.size())
print(Y_val_tensor.size())
print(Y_test_tensor.size())

# ------------------------------ DataLoaders ------------------------------ #

from torch.utils.data import TensorDataset, DataLoader

batch_size = 4096

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

### ------------------------------ Create Model Architecture (MLP for now) ------------------------------ ###

import torch.nn as nn

# Create a DNN ( 4 layers, 128 neurons )

model = nn.Sequential(
    nn.Linear(86, 128),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1)
).to(device)

model = nn.DataParallel(model)

### ------------------------------ Early stopping mechanism ------------------------------ ###

class EarlyStopping:
    def __init__(self, patience=40, min_delta=0):
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
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate) # Use the ADAM optimiser

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)

### ------------------------------ Run Training Loop ------------------------------ ###

# Run a training loop

print("Beginning Training Loop")
print("="*60)
import time

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch
times = [] # Keep track of times

N_epochs = 100 # Number of epochs we iterate over

for epoch in range(N_epochs):

    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0

    for batch_x,batch_y in train_loader:
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
            y_pred = model(batch_x)
            val_loss = loss(y_pred, batch_y)
            epoch_val_loss += val_loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Call scheduler outside batch loop 
    scheduler.step(avg_val_loss)

    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... Training
        print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f} | Epoch Time: {epoch_time:.2f}s | Total Time: {np.sum(times):.2f}s")

    if epoch >= 50:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            print(f'Final Epoch before early stop [{epoch+1}/{N_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            break

    torch.cuda.empty_cache()

### ------------------------------ Evaluate Model ------------------------------ ###

model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        list_of_predictions.append(outputs)

pred = torch.concatenate(list_of_predictions)
Y_pred = pred.detach().cpu().numpy()

from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score

MSE = mean_squared_error(Y_test_scaled, Y_pred)
RMS = root_mean_squared_error(Y_test_scaled,Y_pred)
MAE = mean_absolute_error(Y_test_scaled,Y_pred)
R2 = r2_score(Y_test_scaled,Y_pred)

Y_pred_geV = scaler_Y.inverse_transform(Y_pred.reshape(-1, 1)).flatten()
Y_test_geV = scaler_Y.inverse_transform(Y_test_scaled.reshape(-1, 1)).flatten()

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
plt.savefig("TTBar_Mass.png")
plt.show()

# ------------------------------ Save Predictions to File (Use for ORIGIN) ------------------------------ #

# Create a 2D array with true and predicted masses
results = np.column_stack([Y_test_geV, Y_pred_geV, Y_pred_geV - Y_test_geV])

# Save to file
np.savetxt(
    "ttbar_mass_predictions.txt", 
    results,
    header="True_Mass_GeV  Predicted_Mass_GeV  Resolution_GeV",
    fmt="%.2f",
    delimiter="  "
)

print("Saved predictions to ttbar_mass_predictions.txt")
