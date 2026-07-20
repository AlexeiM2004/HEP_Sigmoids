### ------------------------------ Code Brief ------------------------------ ###

# Accesses root file
# Accesses the "reco" branch
# Extracts all the keys; jet_*, mu_*, el_*, and met_*
# Pads and fills jagged arrays
# Stacks and concatenates X information
# Converts target kinematic features into an array
# Performs an 80:10:10 Train:Validate:Test split
# Scales (Normalises) the data
# Writes all X,Y,M data to H5 file

### ------------------------------ Imports ------------------------------ ###

import uproot # Read in ROOT file format 
import awkward as ak # Used to perform awkward operations on jagged arrays
import matplotlib.pyplot as plt # Used to plot graphs 
import os # Used to find filepath
import numpy as np

### ------------------------------ File Download ------------------------------ ###

filename = "ttbar_2L_mc20eTrain300_240426C_410472_mc20e_fullsim.root"

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
        'indices_to_keep': [0,1,2,3] # Change to include more features
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

# Define columns to load
columns_to_load = []
columns_to_load.extend(feature_dict['jet_features']['columns'])
columns_to_load.extend(feature_dict['electron_features']['columns'])
columns_to_load.extend(feature_dict['muon_features']['columns'])
columns_to_load.extend(feature_dict['met_features']['columns'])
columns_to_load.append('parton_top_pt')
columns_to_load.append('parton_antitop_pt')
columns_to_load.append('parton_top_phi')
columns_to_load.append('parton_antitop_phi')
columns_to_load.append('parton_top_eta')
columns_to_load.append('parton_antitop_eta')
columns_to_load.append('parton_top_m')
columns_to_load.append('parton_antitop_m')

# Load specific columns (this is so the whole tree isnt loaded)
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

### ------------------------------ Kinematic features for target ------------------------------ ###

top_pt = tree_awk["parton_top_pt"]
antitop_pt = tree_awk["parton_antitop_pt"]
top_phi = tree_awk["parton_top_phi"]
antitop_phi = tree_awk["parton_antitop_phi"]
top_eta = tree_awk["parton_top_eta"]
antitop_eta = tree_awk["parton_antitop_eta"]
top_m = tree_awk["parton_top_m"]
antitop_m = tree_awk["parton_antitop_m"]

top_px = top_pt*np.cos(top_phi)
top_py = top_pt*np.sin(top_phi)
top_pz = top_pt*np.sinh(top_eta)
antitop_px = antitop_pt*np.cos(antitop_phi)
antitop_py = antitop_pt*np.sin(antitop_phi)
antitop_pz = antitop_pt*np.sinh(antitop_eta)

import vector

# Predicted vectors
top_vec = vector.Array(
    ak.zip({
        'px': top_px,
        'py': top_py,
        'pz': top_pz,
        'M': top_m
    })
)

antitop_vec = vector.Array(
    ak.zip({
        'px': antitop_px,
        'py': antitop_py,
        'pz': antitop_pz,
        'M': antitop_m
    })
)

top_E = top_vec.energy
antitop_E = antitop_vec.energy

target = np.column_stack([
    ak.to_numpy(top_px),
    ak.to_numpy(top_py),
    ak.to_numpy(top_pz),
    ak.to_numpy(antitop_px),
    ak.to_numpy(antitop_py),
    ak.to_numpy(antitop_pz),
    ak.to_numpy(top_E),
    ak.to_numpy(antitop_E)
])

mass = np.column_stack([
    ak.to_numpy(top_m),
    ak.to_numpy(antitop_m)
])

print(f"Target shape: {target.shape}")

print(f"Mass shape : {mass.shape}")

### ------------------------------ Clean Memory ------------------------------ ###

del X_features, tree, file_ttbar, top_pt, top_eta, top_phi, antitop_pt, antitop_eta, antitop_phi, top_px, top_py, top_pz, antitop_px, antitop_py, antitop_pz, top_m, antitop_m

### ------------------------------ Split into train/validation/test (80:10:10) ------------------------------ ###

from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
X_train, X_temp, Y_train, Y_temp, M_train, M_temp = train_test_split(X, target, mass, test_size=0.2, random_state=67)

# Second split: 10% validation, 10% test
X_val, X_test, Y_val, Y_test, M_val, M_test = train_test_split(X_temp, Y_temp, M_temp, test_size=0.5, random_state=67)

### ------------------------------ Scale the data ------------------------------ ###

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_val_scaled = scaler_Y.transform(Y_val)
Y_test_scaled = scaler_Y.transform(Y_test)

scaler_M = StandardScaler()
M_train_scaled = scaler_M.fit_transform(M_train)
M_val_scaled = scaler_M.transform(M_val)
M_test_scaled = scaler_M.transform(M_test)

### ------------------------------ Save to HDF5 ------------------------------ ###

import h5py

with h5py.File("kinematic_features_train.h5", "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)
    f.create_dataset("M", data=M_train_scaled)

with h5py.File("kinematic_features_val.h5", "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)
    f.create_dataset("M", data=M_val_scaled)

with h5py.File("kinematic_features_test.h5", "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)
    f.create_dataset("M", data=M_test_scaled)

with h5py.File("kinematic_features_scaler_info.h5", "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_)
    f.create_dataset("Y_scale", data=scaler_Y.scale_)
    f.create_dataset("M_mean", data=scaler_M.mean_)
    f.create_dataset("M_scale", data=scaler_M.scale_)

print("Saved to kinematic_features_train.h5")
print("Saved to kinematic_features_val.h5")
print("Saved to kinematic_features_test.h5")
print("Saved scaler info to kinematic_features_scaler_info.h5")

