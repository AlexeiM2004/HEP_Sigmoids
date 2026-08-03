### ------------------------------ Imports ------------------------------ ###

import uproot
import awkward as ak
import os
import numpy as np
import vector
import h5py
from dataclasses import dataclass, field, asdict

### ------------------------------ Control Panels ------------------------------ ###

@dataclass
class Data_Configuration:
    use_large_reco : bool = True            # Reco file sizes
    use_mass_loss : bool = True             # Use mass loss
    use_truth_lepton : bool = True          # Use truth lepton
    use_pred_lepton : bool = True           # Use predicted lepton
    use_direct_regression : bool = False     # Use direct or kinematic regression

control_panel = Data_Configuration()

def display_config(config):
    print("="*60)
    print("CONTROL PANEL - DATA CONFIGURATION")
    print("="*60)
    
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

display_config(control_panel)

### ------------------------------ File Upload ------------------------------ ###

if control_panel.use_large_reco == True:
    filename = "ttbar_2L_mc20eTrain300_240426C_410472_mc20e_fullsim.root"
else:
    filename = "ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root"

size = os.path.getsize(filename)
print(f"\nFile size: {size / (1024**3):.2f} GB")

### ------------------------------ Access TTree ------------------------------ ###

# Open reco file

file_ttbar = uproot.open(filename)

# Access reco branch
tree = file_ttbar["reco"]

### ------------------------------ Feature Dictionary ------------------------------ ###

# Create a dictionary with all 126 features. 

feature_dict = {
    'jet_features': { # 8 Categories of jet features, 13 jets (104 Overall)
        'columns': ['jet_pt_NoOverlap_NOSYS', 'jet_eta_NoOverlap_NOSYS', 'jet_phi_NoOverlap_NOSYS', 
                    'jet_e_NoOverlap_NOSYS', 'jet_GN2v01_FixedCutBEff_65_select_NoOverlap_NOSYS',
                    'jet_GN2v01_FixedCutBEff_70_select_NoOverlap_NOSYS', 
                    'jet_GN2v01_FixedCutBEff_77_select_NoOverlap_NOSYS',
                    'jet_GN2v01_FixedCutBEff_85_select_NoOverlap_NOSYS'],
        'n_jets': 13,
        'indices_to_keep': [0,1,2,3] # Change to include more jets
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

# Array of features to include
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

print(f"\nKeeping {len(features_to_keep)} features for X")
for i, feat_name in enumerate(features_to_keep):
    print(f"  Column {i:02d}: {feat_name}")

del features_to_keep

### ------------------------------ Load Features from Tree ------------------------------ ###

# Define empty array for columns of features to load (The whole tree doesnt have to be loaded then)
columns_to_load = []

# Load in X features
columns_to_load.extend(feature_dict['jet_features']['columns'])
columns_to_load.extend(feature_dict['electron_features']['columns'])
columns_to_load.extend(feature_dict['muon_features']['columns'])
columns_to_load.extend(feature_dict['met_features']['columns'])

if control_panel.use_direct_regression == False:
    # Load in Parton ttbar features (TARGETS)
    columns_to_load.extend([f'parton_{p}_{s}' for p in ['top','antitop'] for s in ['pt','phi','eta','m']])

if control_panel.use_truth_lepton == True:
    # Load in Parton (truth) lepton features (For BOOST FUNCTION)
    columns_to_load.extend([f'parton_{p}_{s}' for p in ['lepton_plus','lepton_minus'] for s in ['pt','phi','eta','m']])

if control_panel.use_pred_lepton == True:
    # Load in RECO (pred) lepton features (For BOOST FUNCTION)
    columns_to_load.extend([f'lep_{p}' for p in ['pt_NOSYS', 'eta_NOSYS', 'phi_NOSYS','e_NOSYS','charge_NOSYS']])

if control_panel.use_direct_regression == True:
    # Load in mttbar feature
    columns_to_load.append('parton_ttbar_m')

# Convert loaded features into arrays
tree_awk = tree.arrays(columns_to_load)

del file_ttbar, tree, columns_to_load

### ------------------------------ Extract X Features from Reco ------------------------------ ###

# Define empty array for X features
X_features = []

# For X features with jagged arrays, pad and fill with none then append to X features array

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

# Stack
X = np.column_stack([ak.to_numpy(part) for part in X_features])

print(f"\nX shape: {X.shape}")

del X_features, feature_dict

### ------------------------------ Calculate and Extract Target Features From Reco ------------------------------ ###

if control_panel.use_direct_regression == False or control_panel.use_mass_loss == True:
    # Build t, tbar and ttbar 4-vectors
    top_vec = vector.zip({s: tree_awk[f"parton_top_{s}"] for s in ["pt", "phi", "eta", "m"]})
    antitop_vec = vector.zip({s: tree_awk[f"parton_antitop_{s}"] for s in ["pt", "phi", "eta", "m"]})

if control_panel.use_direct_regression == False:
    # Extract parton targets
    target = np.column_stack([
        ak.to_numpy(getattr(v, attr))
        for v in [top_vec, antitop_vec]
        for attr in ["px", "py", "pz", "E"]
    ])

    print(f"Target shape: {target.shape}")

if control_panel.use_mass_loss == True:
    ttbar_vec = top_vec + antitop_vec

if control_panel.use_direct_regression == True:
    # Add mttbar feature to target array
    mttbar = tree_awk["parton_ttbar_m"]

    target = np.column_stack([ak.to_numpy(mttbar)])

    print(f"Target shape: {target.shape}")

# Stack
if control_panel.use_mass_loss == True:
    mass = np.column_stack([
        ak.to_numpy(top_vec.mass),
        ak.to_numpy(antitop_vec.mass),
        ak.to_numpy(ttbar_vec.mass)
    ])

    print(f"Mass shape : {mass.shape}")
    del ttbar_vec

if control_panel.use_direct_regression == False:
    del top_vec, antitop_vec

### ------------------------------ Extract Truth Lepton Features from Tree ------------------------------ ###

if control_panel.use_truth_lepton == True:
    # Build lepton plus and minus parton (Truth) 4-vectors
    lepP_truth = vector.zip({s: tree_awk[f"parton_lepton_plus_{s}"] for s in ["pt", "phi", "eta", "m"]})
    lepM_truth = vector.zip({s: tree_awk[f"parton_lepton_minus_{s}"] for s in ["pt", "phi", "eta", "m"]})

    # Stack
    lepton_features = np.column_stack([
        ak.to_numpy(getattr(v, attr))
        for v in [lepP_truth, lepM_truth]
        for attr in ["px", "py", "pz", "E"]
    ])

    print(f"Lepton features shape: {lepton_features.shape}")

    del lepP_truth, lepM_truth

### ------------------------------ Extract Reco Lepton Features ------------------------------ ###

if control_panel.use_pred_lepton == True:
    lep_vecs = vector.zip({
        'pt':  tree_awk["lep_pt_NOSYS"] / 1000,
        'eta': tree_awk["lep_eta_NOSYS"],
        'phi': tree_awk["lep_phi_NOSYS"],
        'E':   tree_awk["lep_e_NOSYS"] / 1000,
        'charge': tree_awk["lep_charge_NOSYS"]
    })

    # Isolate strictly by charge, regardless of whether they are index 0 or 1
    leptonP_vec = lep_vecs[lep_vecs.charge > 0][:, 0]
    leptonM_vec = lep_vecs[lep_vecs.charge < 0][:, 0]

    reco_arrays = []
    for vec in [leptonP_vec, leptonM_vec]:
        for attr in ["px", "py", "pz", "E"]:
            reco_arrays.append(ak.to_numpy(getattr(vec, attr)))

    reco_features = np.column_stack(reco_arrays)

    print(f"Reco lepton features shape: {reco_features.shape}")
    del lep_vecs, leptonP_vec, leptonM_vec

del tree_awk

### ------------------------------ Split into train/validation/test (80:10:10) ------------------------------ ###

from sklearn.model_selection import train_test_split

# First split X,Y,M,L & R into 80% (Train) : 20% (Temp)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, target, test_size=0.2, random_state=67)

if control_panel.use_mass_loss == True:
    M_train, M_temp = train_test_split(mass, test_size=0.2, random_state=67)
    M_val, M_test = train_test_split(M_temp, test_size=0.5, random_state=67)
    del M_temp, mass

if control_panel.use_truth_lepton == True:
    L_train, L_temp = train_test_split(lepton_features, test_size=0.2, random_state=67)
    L_val, L_test = train_test_split(L_temp, test_size=0.5, random_state=67)
    del L_temp, lepton_features

if control_panel.use_pred_lepton == True:
    R_train, R_temp = train_test_split(reco_features, test_size=0.2, random_state=67)
    R_val, R_test  = train_test_split(R_temp, test_size=0.5, random_state=67)
    del R_temp, reco_features

# Second split X,Y,M,L & R into 10% (Validation) : 10% (Test)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=67)

del X_temp, Y_temp, target, X

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

del X_train, X_val, X_test, Y_train, Y_val, Y_test

### ------------------------------ Save to HDF5 ------------------------------ ###

# Save X,Y,M,L, & R data to train,test,val & scaler files

with h5py.File("kinematic_withM_spin_observable_features_train.h5", "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)

    if control_panel.use_mass_loss == True:
        f.create_dataset("M", data=M_train)
    if control_panel.use_truth_lepton == True:
        f.create_dataset("L", data=L_train)
    if control_panel.use_pred_lepton == True:
        f.create_dataset("R", data=R_train)

with h5py.File("kinematic_withM_spin_observable_features_val.h5", "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)

    if control_panel.use_mass_loss == True:
        f.create_dataset("M", data=M_val)
    if control_panel.use_truth_lepton == True:
        f.create_dataset("L", data=L_val)
    if control_panel.use_pred_lepton == True:
        f.create_dataset("R", data=R_val)

with h5py.File("kinematic_withM_spin_observable_features_test.h5", "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)

    if control_panel.use_mass_loss == True:
        f.create_dataset("M", data=M_test)
    if control_panel.use_truth_lepton == True:
        f.create_dataset("L", data=L_test)
    if control_panel.use_pred_lepton == True:
        f.create_dataset("R", data=R_test)

with h5py.File("kinematic_withM_spin_observable_features_scaler_info.h5", "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_)
    f.create_dataset("Y_scale", data=scaler_Y.scale_)

print("Saved to kinematic_withM_spin_observable_features_train.h5")
print("Saved to kinematic_withM_spin_observable_features_val.h5")
print("Saved to kinematic_withM_spin_observable_features_test.h5")
print("Saved scaler info to kinematic_withM_spin_observable_features_scaler_info.h5")

