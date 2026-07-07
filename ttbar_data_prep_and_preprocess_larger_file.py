### ------------------------------ Code Brief ------------------------------ ###

# Accesses root file
# Accesses the "reco" branch
# Extracts all the keys; jet_*, mu_*, el_*, and met_*
# Pads and fills jagged arrays
# Stacks and concatenates X information
# Converts target to array
# Performs an 80:10:10 Train:Validate:Test split
# Scales (Normalises) the data
# Writes all data to H5 file

### ------------------------------ Imports ------------------------------ ###

import uproot # Read in ROOT file format 
import awkward as ak # Used to perform awkward operations on jagged arrays
import matplotlib.pyplot as plt # Used to plot graphs 
import os # Used to find filepath
import numpy as np

### ------------------------------ File Download ------------------------------ ###

filename = "../data/ttbar_2L_mc20eTrain300_240426C_410472_mc20e_fullsim.root"

size = os.path.getsize(filename)
print(f"File size: {size / (1024**3):.2f} GB")

### ------------------------------ Data Preperation ------------------------------ ###

# Open main file
file_ttbar = uproot.open(filename)

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

# Build the flat, ordered feature-name list to match X's columns exactly
feature_names = []
feature_names += [f"{col}_{i}" for col in jet_cols for i in range(8)]
feature_names += [f"{col}_{i}" for col in el_cols for i in range(5)]
feature_names += [f"{col}_{i}" for col in mu_cols for i in range(5)]
feature_names += [f"{col}_{i}" for col in met_cols for i in range(1)]

feature_names = np.array(feature_names)

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

### ------------------------------ Save to HDF5 ------------------------------ ###

import h5py

with h5py.File("../data/larger_ttbar_train.h5", "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)

with h5py.File("../data/larger_ttbar_val.h5", "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)

with h5py.File("../data/larger_ttbar_test.h5", "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)

with h5py.File("../data/larger_scaler_info.h5", "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_[0])
    f.create_dataset("Y_scale", data=scaler_Y.scale_[0])

with h5py.File("../data/feature_labels.h5","w") as f:
    f.create_dataset("Feature_labels", data=feature_names.astype("S"))

print("Saved to larger_ttbar_train.h5")
print("Saved to larger_ttbar_val.h5")
print("Saved to larger_ttbar_test.h5")
print("Saved scaler info to larger_scaler_info.h5")
print("Saved features to larger_feature_labels.h5")
