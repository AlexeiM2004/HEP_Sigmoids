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

### ------------------------------ File Download ------------------------------ ###

filename = "ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root"

size = os.path.getsize("ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root")
print(f"File size: {size / (1024**3):.2f} GB")

### ------------------------------ Data Preperation ------------------------------ ###

# Open main file
file_ttbar  = uproot.open("ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root") 

# Access reco branch

tree = file_ttbar["reco"] 

# Access all jet_ , el_, and mu_ columns

jet_cols = [col for col in tree.keys() if col.startswith('jet_')]
el_cols = [col for col in tree.keys() if col.startswith('el_')]
mu_cols = [col for col in tree.keys() if col.startswith('mu_')]


# List concatenate all colums into one input column

input_cols = jet_cols + el_cols + mu_cols 

# Access ttbar_m column

ttbar_mass = tree["parton_ttbar_m"]

# Pad and fill all data to their respective max, using hard coded maxima

jet_cols_padded_filled = []

for i in range(8):
    padded = ak.pad_none(tree[jet_cols[i]].array(),13,clip=True)
    filled = ak.fill_none(padded, 0.0)
    jet_cols_padded_filled.append(filled)

el_cols_padded_filled = []
mu_cols_padded_filled = []

for i in range(5):
    padded_el = ak.pad_none(tree[el_cols[i]].array(),2,clip=True)
    padded_mu = ak.pad_none(tree[mu_cols[i]].array(),2,clip=True)
    filled_el = ak.fill_none(padded_el, 0.0)
    filled_mu = ak.fill_none(padded_mu, 0.0)
    el_cols_padded_filled.append(filled_el)
    mu_cols_padded_filled.append(filled_mu)

# Stack input columns individually from each arrays

X_jets = ak.to_numpy(ak.concatenate(jet_cols_padded_filled, axis=1))
X_el = ak.to_numpy(ak.concatenate(el_cols_padded_filled, axis=1))
X_mu = ak.to_numpy(ak.concatenate(mu_cols_padded_filled, axis=1))

# Concatenate X data

X = np.hstack([X_jets, X_el, X_mu])

target = np.array(ttbar_mass)

### ------------------------------ Split into train/validation/test (80:10:10) ------------------------------ ###

from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
X_train, X_temp, Y_train, Y_temp = train_test_split(X, target, test_size=0.2, random_state=42)

# Second split: 10% validation, 10% test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print("----Data Splitting----")
print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")
print("--------------------")

### ------------------------------ Scale the data ------------------------------ ###

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_X.fit(X_train)

X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_Y = StandardScaler()
scaler_Y.fit(Y_train.reshape(-1, 1))
Y_train_scaled = scaler_Y.transform(Y_train.reshape(-1, 1)).flatten()
Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()

### ------------------------------ Convert Into Tensors ------------------------------ ###

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

Y_train_tensor = torch.tensor(Y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
Y_val_tensor = torch.tensor(Y_val_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
Y_test_tensor = torch.tensor(Y_test_scaled.reshape(-1, 1), dtype=torch.float32, device=device)

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
import torch.nn.functional as F

class ttbar_mass_regression(nn.Module):
    def __init__(self, in_dim):
        super(ttbar_mass_regression, self).__init__()
        self.lin1 = nn.Linear(in_dim, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 32)
        self.lin4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        identity = F.leaky_relu(self.lin1(x))

        x1 = F.leaky_relu(self.lin2(identity))
        x1 = x1 + identity
        x2 = self.dropout(F.leaky_relu(self.lin3(x1)))
        return self.lin4(x2)

model = ttbar_mass_regression(X_train_tensor.shape[1]).to(device)

### ------------------------------ Define loss func, optimiser, and scheduler ------------------------------ ###

loss = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001) # Use the ADAM optimiser

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)


### ------------------------------ Run Training Loop ------------------------------ ###

# Run a training loop

import time

train_losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch
times = [] # Keep track of times

N_epochs = 150 # Number of epochs we iterate over

train_time_start = time.perf_counter() # Tracking overall training time

print()
print("-----Beginning training-----")
for epoch in range(N_epochs):

    start_time = time.time()
    model.train()
    epoch_train_loss = 0.0

    for batch_x,batch_y in train_loader:
        # Perform a forward pass
        y_pred = model(batch_x)

        batch_y = batch_y.view(-1, 1)

        train_loss = loss(y_pred,batch_y)
        
        # Perform a backward pass + Optimisation

        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()
        epoch_train_loss += train_loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

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

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{N_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Loss Diff : {np.abs(avg_val_loss - avg_train_loss):.4f}") 
        print(f"Epoch Time: {epoch_time:.2f}s | Total Time: {(np.sum(times) // 60):.0f} minutes {(np.sum(times)) % 60:.2f} seconds")
        print("-----------------------------------------------------")

    torch.cuda.empty_cache()

# -------------------------------------------------------------------------------------------------- #

# Testing

model.eval()
list_of_predictions = []

with torch.no_grad():
  for inputs, targets in test_loader:
    outputs = model(inputs)
    list_of_predictions.append(outputs.cpu().numpy())

y_pred_scaled = np.vstack(list_of_predictions)
y_pred_physical = scaler_Y.inverse_transform(y_pred_scaled)

# -------------------------------------------------------------------------------------------------- #

# Plotting

def plot_loss_curve(train_losses,val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss") 
    plt.title("Loss curve")
    plt.legend()
    plt.plot()

    plt.savefig("ttbar_reg_losses_plot.png")

def plot_regression(y_pred, y_true):
    if hasattr(y_pred, "cpu"): y_pred = y_pred.cpu().numpy()
    if hasattr(y_true, "cpu"): y_true = y_true.cpu().numpy()
    
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    plt.figure(figsize=(8, 6))
    
    # Plot the predictions vs true values
    plt.scatter(y_pred, y_true, alpha=0.5, s=5, label="Events")

    max_val = max(max(y_pred), max(y_true))
    min_val = min(min(y_pred), min(y_true))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label="Perfect Regression")
    
    plt.xlabel("Predicted System Mass [GeV]")
    plt.ylabel("True System Mass [GeV]")
    plt.title("T-Tbar System Mass Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("ttbar_mass_reg_plot.png")


plot_regression(y_pred_physical, Y_test)
plot_loss_curve(train_losses,val_losses)
# -------------------------------------------------------------------------------------------------- #

# Metric calculations

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Standard ML Metrics
rmse = np.sqrt(mean_squared_error(Y_test, y_pred_physical))
mae = mean_absolute_error(Y_test, y_pred_physical)
r2 = r2_score(Y_test, y_pred_physical)

print(f"RMSE: {rmse:.2f} GeV")
print(f"MAE:  {mae:.2f} GeV")
print(f"R2:   {r2:.4f}")

# -------------------------------------------------------------------------------------------------- #

# Importances

def make_prediction(model, data_loader):
    model.eval()
    list_of_predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            list_of_predictions.append(outputs.cpu().numpy())
    
    return np.vstack(list_of_predictions)

from sklearn.metrics import accuracy_score

def permutation_importance(model, X_tensor, Y_tensor, Y_vals, batch_size):
    
    physical_prediction = lambda x: scaler_Y.inverse_transform(make_prediction(model, x))

    # Make dataloader for baseline
    baseline_dataset = TensorDataset(X_tensor, Y_tensor)
    baseline_loader = DataLoader(baseline_dataset, batch_size=batch_size, shuffle=False)

    baseline_score = r2_score(Y_vals, physical_prediction(baseline_loader))
    importances = []

    for feature_idx in range(X_tensor.shape[1]):
        # Shuffle the values of the current feature
        X_tensor_shuffled = X_tensor.clone()
        X_tensor_shuffled[:, feature_idx] = X_tensor_shuffled[:, feature_idx][torch.randperm(X_tensor.shape[0])]

        # Make shuffled dataloader
        shuffled_dataset = TensorDataset(X_tensor_shuffled, Y_tensor)
        shuffled_dataloader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False)

        # Calculate Shuffled score
        shuffled_score = r2_score(Y_vals, physical_prediction(shuffled_dataloader))

        # Subtract scores for importance

        importances.append(baseline_score - shuffled_score)

    return np.array(importances)

def plot_importances(importances, jet_names, el_names, mu_names):
    importances = np.asarray(importances)
    
    # 1. Rebuild the 124 feature names
    true_feature_names = []
    for col in jet_names:
        for jet_idx in range(13):
            true_feature_names.append(f"{col}_{jet_idx}")
            
    for col in el_names:
        for el_idx in range(2):
            true_feature_names.append(f"{col}_{el_idx}")
            
    for col in mu_names:
        for mu_idx in range(2):
            true_feature_names.append(f"{col}_{mu_idx}")
            
    features_array = np.asarray(true_feature_names)
    
    # 2. Sort high-to-low
    idx = np.argsort(importances)[::-1]
    sorted_features_descending = features_array[idx]
    sorted_importances = importances[idx]
    
    # 3. Slice to Top 25
    top_n = 25
    bot_n = 0

    sorted_features_descending = sorted_features_descending[bot_n:top_n]
    sorted_importances = sorted_importances[bot_n:top_n]

    plt.figure(figsize=(12, 8))
    # Draw 25 bars
    plt.barh(range(top_n), sorted_importances, align='center', color='royalblue', edgecolor='k')
    
    # FIX: Changed range(len(importances)) to range(top_n) to perfectly match 25 labels
    plt.yticks(range(top_n), sorted_features_descending)
    
    plt.xlabel('Importance (Drop in R2 Score)')
    plt.ylabel('Feature Mapped Column')
    plt.title(f'Top {top_n} Feature Importances for T-Tbar Mass Regression')
    plt.gca().invert_yaxis()  # Keep highest importance at the top
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("ttbar_importance_plot.png", dpi=300)

print("-----Calculating Importances-----")
importances = permutation_importance(model, X_val_tensor, Y_val_tensor, Y_val, batch_size)
plot_importances(importances, jet_cols, el_cols, mu_cols)

# -------------------------------------------------------------------------------------------------- #