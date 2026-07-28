### ------------------------------ Code Brief ------------------------------ ###

# Accesses root file
# Accesses the "reco" branch
# Extracts all the keys; jet_*, mu_*, el_*, and met_*
# Pads and fills jagged arrays
# Stacks and concatenates X information
# Converts target spin observable features into an array
# Performs an 80:10:10 Train:Validate:Test split
# Scales (Normalises) the data
# Writes all X,Y data to H5 file

### ------------------------------ Imports ------------------------------ ###

import uproot
import awkward as ak
import os
import numpy as np
import vector
import h5py

### ------------------------------ File Download ------------------------------ ###

filename = "ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root"

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

### ------------------------------ Load Features from Reco ------------------------------ ###

# Define empty array for columns to load
columns_to_load = []

# Load in X features
columns_to_load.extend(feature_dict['jet_features']['columns'])
columns_to_load.extend(feature_dict['electron_features']['columns'])
columns_to_load.extend(feature_dict['muon_features']['columns'])
columns_to_load.extend(feature_dict['met_features']['columns'])

# Load in target features
columns_to_load.extend([f'parton_{p}_{s}' for p in ['top','antitop','lepton_plus','lepton_minus'] for s in ['pt','phi','eta','m']])
columns_to_load.append('parton_ttbar_m')
tree_awk = tree.arrays(columns_to_load)

### ------------------------------ Extract X Features from Reco ------------------------------ ###

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

### ------------------------------ Calculate and Extract Target Features From Reco ------------------------------ ###

Target_features = []

top_pt, top_phi, top_eta, top_m = [tree_awk[f"parton_top_{s}"] for s in ["pt", "phi", "eta", "m"]]

antitop_pt, antitop_phi, antitop_eta, antitop_m = [tree_awk[f"parton_antitop_{s}"] for s in ["pt", "phi", "eta", "m"]]

lepton_plus_pt, lepton_plus_phi, lepton_plus_eta, lepton_plus_m = [tree_awk[f"parton_lepton_plus_{s}"] for s in ["pt", "phi", "eta", "m"]]

lepton_minus_pt, lepton_minus_phi, lepton_minus_eta, lepton_minus_m = [tree_awk[f"parton_lepton_minus_{s}"] for s in ["pt", "phi", "eta", "m"]]

def momentum_energy_calculator(pt,phi,eta,m):
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)

    vec = vector.Array(
    ak.zip({
        'px': px,
        'py': py,
        'pz': pz,
        'M': m }))

    E = vec.energy

    return px,py,pz,E

Target_features.extend(momentum_energy_calculator(top_pt, top_phi, top_eta, top_m))
Target_features.extend(momentum_energy_calculator(antitop_pt, antitop_phi, antitop_eta, antitop_m))
Target_features.extend(momentum_energy_calculator(lepton_plus_pt, lepton_plus_phi, lepton_plus_eta, lepton_plus_m))
Target_features.extend(momentum_energy_calculator(lepton_minus_pt, lepton_minus_phi, lepton_minus_eta, lepton_minus_m))

#mttbar = tree_awk["parton_ttbar_m"]
#mttbar_np = ak.to_numpy(mttbar)

target = np.column_stack([ak.to_numpy(part) for part in Target_features])
print(f"Target shape: {target.shape}")

### ------------------------------ Clean Memory ------------------------------ ###

del X_features, tree, file_ttbar, Target_features, top_pt, top_phi, top_eta, top_m, antitop_pt, antitop_phi, antitop_eta, antitop_m, lepton_plus_pt, lepton_plus_phi, lepton_plus_eta, lepton_plus_m, lepton_minus_pt, lepton_minus_phi, lepton_minus_eta, lepton_minus_m

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
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_val_scaled = scaler_Y.transform(Y_val)
Y_test_scaled = scaler_Y.transform(Y_test)

### ------------------------------ Save to HDF5 ------------------------------ ###

with h5py.File("kinematic_spin_observable_features_train.h5", "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)

with h5py.File("kinematic_spin_observable_features_val.h5", "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)

with h5py.File("kinematic_spin_observable_features_test.h5", "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)

with h5py.File("kinematic_spin_observable_features_scaler_info.h5", "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_)
    f.create_dataset("Y_scale", data=scaler_Y.scale_)

print("Saved to kinematic_spin_observable_features_train.h5")
print("Saved to kinematic_spin_observable_features_val.h5")
print("Saved to kinematic_spin_observable_features_test.h5")
print("Saved scaler info to kinematic_spin_observable_features_scaler_info.h5")


