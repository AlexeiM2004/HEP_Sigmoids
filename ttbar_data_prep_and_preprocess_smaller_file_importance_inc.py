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

filename = "../data/ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root"

size = os.path.getsize(filename)
print(f"File size: {size / (1024**3):.2f} GB")

### ------------------------------ Access TTree ------------------------------ ###

# Open main file
file_ttbar = uproot.open(filename)

# Access reco branch
tree = file_ttbar["reco"]

# Get branch names from TTree
all_branches = tree.keys()

### ------------------------------ Feature Dictionary ------------------------------ ###

# Create a dictionary with all 126 features. 

feature_dict = {
    'jet_features': { # 8 Categories of jet features
        'columns': ['jet_pt_NoOverlap_NOSYS', 'jet_eta_NoOverlap_NOSYS', 'jet_phi_NoOverlap_NOSYS', 
                    'jet_e_NoOverlap_NOSYS', 'jet_GN2v01_FixedCutBEff_65_select_NoOverlap_NOSYS',
                    'jet_GN2v01_FixedCutBEff_70_select_NoOverlap_NOSYS', 
                    'jet_GN2v01_FixedCutBEff_77_select_NoOverlap_NOSYS',
                    'jet_GN2v01_FixedCutBEff_85_select_NoOverlap_NOSYS'],
        'n_jets': 13,
        'indices_to_keep': [0, 1, 2, 3] # Change to include more features
    },
    
    'electron_features': { # 5 Categories of electron features
        'columns': ['el_pt_NOSYS', 'el_eta', 'el_phi', 'el_e_NOSYS', 'el_charge'],
        'n_objects': 2,
        'indices_to_keep': [0, 1]
    },
    
    'muon_features': { # 5 Categories of muon features
        'columns': ['mu_pt_NOSYS', 'mu_eta', 'mu_phi', 'mu_e_NOSYS', 'mu_charge'],
        'n_objects': 2,
        'indices_to_keep': [0, 1] 
    },
    
    'met_features': { # 2 Categories of met features
        'columns': ['met_met_NOSYS', 'met_phi_NOSYS'],
    }
}

### ------------------------------ Build Features List ------------------------------ ###

features_to_keep = []

# Jets
for col in feature_dict['jet_features']['columns']:
    for idx in feature_dict['jet_features']['indices_to_keep']:
        features_to_keep.append(f"{col}_{idx}")

# Electrons
for col in feature_dict['electron_features']['columns']:
    for idx in feature_dict['electron_features']['indices_to_keep']:
        features_to_keep.append(f"{col}_{idx}")

# Muons
for col in feature_dict['muon_features']['columns']:
    for idx in feature_dict['muon_features']['indices_to_keep']:
        features_to_keep.append(f"{col}_{idx}")

# MET
for col in feature_dict['met_features']['columns']:
    features_to_keep.append(col)

print(f"Keeping {len(features_to_keep)} features out of 126")

### ------------------------------ Extract Features from Reco ------------------------------ ###

columns_to_load = []
columns_to_load.extend(feature_dict['jet_features']['columns'])
columns_to_load.extend(feature_dict['electron_features']['columns'])
columns_to_load.extend(feature_dict['muon_features']['columns'])
columns_to_load.extend(feature_dict['met_features']['columns'])
columns_to_load.append('parton_ttbar_m')

tree_awk = tree.arrays(columns_to_load)

X_features = []

for col in feature_dict['jet_features']['columns']:
    data = tree_awk[col]
    padded = ak.pad_none(data, feature_dict['jet_features']['n_jets'], clip=True)
    filled = ak.fill_none(padded, 0.0)
    selected_features = filled[:, feature_dict['jet_features']['indices_to_keep']]
    X_features.append(selected_features)

for col in feature_dict['electron_features']['columns']:
    data = tree_awk[col]
    padded = ak.pad_none(data, feature_dict['electron_features']['n_objects'], clip=True)
    filled = ak.fill_none(padded, 0.0)
    selected_features = filled[:, feature_dict['electron_features']['indices_to_keep']]
    X_features.append(selected_features)

for col in feature_dict['muon_features']['columns']:
    data = tree_awk[col]
    padded = ak.pad_none(data, feature_dict['muon_features']['n_objects'], clip=True)
    filled = ak.fill_none(padded, 0.0)
    selected_features = filled[:, feature_dict['muon_features']['indices_to_keep']]
    X_features.append(selected_features)

for col in feature_dict['met_features']['columns']:
    data = tree_awk[col] 
    X_features.append(data)

X = np.column_stack([ak.to_numpy(part) for part in X_features])
print(f"X shape: {X.shape}")

ttbar_mass = tree_awk["parton_ttbar_m"]
target = np.array(ttbar_mass)
print(f"Target shape: {target.shape}")

### ------------------------------ Clean Memory ------------------------------ ###

del tree, file_ttbar, ttbar_mass
del X_features, data, padded, filled

### ------------------------------ Split into train/validation/test (80:10:10) ------------------------------ ###

from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
X_train, X_temp, Y_train, Y_temp = train_test_split(X, target, test_size=0.2, random_state=67)

# Second split: 10% validation, 10% test
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=67)

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

with h5py.File("../data/smaller_importance_trimmed_ttbar_train.h5", "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)

with h5py.File("../data/smaller_importance_trimmed_ttbar_val.h5", "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)

with h5py.File("../data/smaller_importance_trimmed_ttbar_test.h5", "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)

with h5py.File("../data/smaller_importance_trimmed_scaler_info.h5", "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_[0])
    f.create_dataset("Y_scale", data=scaler_Y.scale_[0])

print("Saved to smaller_importance_trimmed_ttbar_train.h5")
print("Saved to smaller_importance_trimmed_ttbar_val.h5")
print("Saved to smaller_importance_trimmed_ttbar_test.h5")
print("Saved scaler info to smaller_importance_trimmed_scaler_info.h5")

