### ------------------------------ Imports ------------------------------ ###

import uproot
import awkward as ak
import os
import numpy as np
import vector
import h5py
import yaml

### ------------------------------ Load YAML ------------------------------ ###

with open("config_data.yaml", "r") as f:
    config_data = yaml.safe_load(f)

with open("config_model.yaml", "r") as f:
    config_model = yaml.safe_load(f)

# ------------------------------ Print Configs ------------------------------ #

def print_yaml_config(config, title):
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")

print_yaml_config(config_data, "DATA CONFIGURATION")

# ------------------------------ Extract All Values ------------------------------ #

# Data
train_file = config_data["data"]["train_file"]
val_file = config_data["data"]["val_file"]
test_file = config_data["data"]["test_file"]
scaler_file = config_data["data"]["scaler_file"]

# Preprocessing
use_large_reco = config_data["preprocessing"]["use_large_reco"]
use_direct_regression = config_data["preprocessing"]["use_direct_regression"]
use_mass_loss = config_data["preprocessing"]["use_mass_loss"]
use_truth_lepton = config_data["preprocessing"]["use_truth_lepton"]
use_pred_lepton = config_data["preprocessing"]["use_pred_lepton"]
direct_targets = config_data["preprocessing"]["direct_targets"]

# Saving
loss_curve = config_data["saving"]["loss_curve"]
evaluation_results = config_data["saving"]["evaluation_results"]

### ------------------------------ File Upload ------------------------------ ###

if use_large_reco == True:
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
    },

    'reco_EM_features' : {
        'columns' : ['reco_EM_Wminus_eta_NOSYS', 'reco_EM_Wminus_m_NOSYS', 'reco_EM_Wminus_phi_NOSYS',
                    'reco_EM_Wminus_pt_NOSYS', 'reco_EM_Wplus_eta_NOSYS', 'reco_EM_Wplus_m_NOSYS',
                    'reco_EM_Wplus_phi_NOSYS', 'reco_EM_Wplus_pt_NOSYS', 'reco_EM_antitop_eta_NOSYS',
                    'reco_EM_antitop_m_NOSYS', 'reco_EM_antitop_phi_NOSYS', 'reco_EM_antitop_pt_NOSYS', 
                    'reco_EM_nu_antitop_eta_NOSYS', 'reco_EM_nu_antitop_m_NOSYS', 'reco_EM_nu_antitop_phi_NOSYS',
                    'reco_EM_nu_antitop_pt_NOSYS', 'reco_EM_nu_top_eta_NOSYS', 'reco_EM_nu_top_m_NOSYS',
                    'reco_EM_nu_top_phi_NOSYS', 'reco_EM_nu_top_pt_NOSYS', 'reco_EM_top_eta_NOSYS', 
                    'reco_EM_top_m_NOSYS', 'reco_EM_top_phi_NOSYS', 'reco_EM_top_pt_NOSYS'],
    },

    'reco_NW_features' : {
        'columns' : ['reco_NW_Wminus_eta_NOSYS', 'reco_NW_Wminus_m_NOSYS', 'reco_NW_Wminus_phi_NOSYS',
                    'reco_NW_Wminus_pt_NOSYS', 'reco_NW_Wplus_eta_NOSYS', 'reco_NW_Wplus_m_NOSYS', 
                    'reco_NW_Wplus_phi_NOSYS', 'reco_NW_Wplus_pt_NOSYS', 'reco_NW_antitop_eta_NOSYS',
                    'reco_NW_antitop_m_NOSYS', 'reco_NW_antitop_phi_NOSYS', 'reco_NW_antitop_pt_NOSYS',
                    'reco_NW_nu_antitop_eta_NOSYS', 'reco_NW_nu_antitop_m_NOSYS', 'reco_NW_nu_antitop_phi_NOSYS', 
                    'reco_NW_nu_antitop_pt_NOSYS', 'reco_NW_nu_top_eta_NOSYS', 'reco_NW_nu_top_m_NOSYS',
                    'reco_NW_nu_top_phi_NOSYS', 'reco_NW_nu_top_pt_NOSYS', 'reco_NW_top_eta_NOSYS', 
                    'reco_NW_top_m_NOSYS', 'reco_NW_top_phi_NOSYS', 'reco_NW_top_pt_NOSYS'],
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

# MET (Missing Energy Transverse)
for col in feature_dict['met_features']['columns']:
    features_to_keep.append(col)

# RECO EM (Reconstructed electromagnetic)
for col in feature_dict['reco_EM_features']['columns']:
    features_to_keep.append(col)

# RECO NW (Reconstructed neutrino weighting)
for col in feature_dict['reco_NW_features']['columns']:
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
columns_to_load.extend(feature_dict['reco_EM_features']['columns'])
columns_to_load.extend(feature_dict['reco_NW_features']['columns'])


if (not use_direct_regression) or (any(t in ["cos_theta_star", "cos_theta_ttbar", "cos_x_plus_minus", "cos_N_plus_minus"] for t in direct_targets)):
    columns_to_load.extend([f'parton_{p}_{s}' for p in ['top','antitop'] for s in ['pt','phi','eta','m']])

if use_truth_lepton or any(t in ["cos_x_plus_minus", "cos_N_plus_minus"] for t in direct_targets):
    # Load in Parton (truth) lepton features (For BOOST FUNCTION)
    columns_to_load.extend([f'parton_{p}_{s}' for p in ['lepton_plus','lepton_minus'] for s in ['pt','phi','eta','m']])

if use_pred_lepton:
    # Load in RECO (pred) lepton features (For BOOST FUNCTION)
    columns_to_load.extend([f'lep_{p}' for p in ['pt_NOSYS', 'eta_NOSYS', 'phi_NOSYS','e_NOSYS','charge_NOSYS']])

if use_direct_regression:
    # Load in mttbar feature
    columns_to_load.append('parton_ttbar_m')

# Convert loaded features into arrays
tree_awk = tree.arrays(columns_to_load)

del file_ttbar, tree, columns_to_load

### ------------------------------ Extract X Features from Reco ------------------------------ ###

# Define empty array for X features
X_features = []

# For X features with jagged arrays, pad and fill with none then append to X features array

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

for col in feature_dict['reco_EM_features']['columns']:
    data = tree_awk[col] 
    X_features.append(data)

for col in feature_dict['reco_NW_features']['columns']:
    data = tree_awk[col] 
    X_features.append(data)

for col in feature_dict['jet_features']['columns']:
    data = tree_awk[col]
    padded = ak.pad_none(data, feature_dict['jet_features']['n_jets'], clip=True)
    filled = ak.fill_none(padded, 0.0)
    selected_features = filled[:, feature_dict['jet_features']['indices_to_keep']]
    X_features.append(selected_features)

# Stack
X = np.column_stack([ak.to_numpy(part) for part in X_features])

print(f"\nX shape: {X.shape}")

del X_features, feature_dict

### ------------------------------ Extract Truth Lepton Features from Tree ------------------------------ ###

if use_truth_lepton or any(t in ["cos_x_plus_minus", "cos_N_plus_minus"] for t in direct_targets):
    # Build lepton plus and minus parton (Truth) 4-vectors
    lepP_truth = vector.zip({s: tree_awk[f"parton_lepton_plus_{s}"] for s in ["pt", "phi", "eta", "m"]})
    lepM_truth = vector.zip({s: tree_awk[f"parton_lepton_minus_{s}"] for s in ["pt", "phi", "eta", "m"]})
    
    if use_truth_lepton:
        # Stack
        lepton_features = np.column_stack([
            ak.to_numpy(getattr(v, attr))
            for v in [lepP_truth, lepM_truth]
            for attr in ["px", "py", "pz", "E"]
        ])

        print(f"Lepton features shape: {lepton_features.shape}")

        del lepP_truth, lepM_truth

### ------------------------------ Extract Reco Lepton Features ------------------------------ ###

if use_pred_lepton:
    # Build vectors for electrons and muons (converting MeV -> GeV for pt & E)
    el_vecs = vector.zip({
        'pt':  tree_awk["el_pt_NOSYS"] / 1000.0,
        'eta': tree_awk["el_eta"],
        'phi': tree_awk["el_phi"],
        'E':   tree_awk["el_e_NOSYS"] / 1000.0,
        'charge': tree_awk["el_charge"]
    })

    mu_vecs = vector.zip({
        'pt':  tree_awk["mu_pt_NOSYS"] / 1000.0,
        'eta': tree_awk["mu_eta"],
        'phi': tree_awk["mu_phi"],
        'E':   tree_awk["mu_e_NOSYS"] / 1000.0,
        'charge': tree_awk["mu_charge"]
    })

    # Concatenate electron and muon collections per event natively along axis=1
    all_lep_vecs = ak.concatenate([el_vecs, mu_vecs], axis=1)

    # Filter strictly by physical electric charge (+1 vs -1)
    lepP_filtered = all_lep_vecs[all_lep_vecs.charge > 0]
    lepM_filtered = all_lep_vecs[all_lep_vecs.charge < 0]

    # Safely extract the single positive and negative lepton vector per event
    leptonP_vec = ak.firsts(lepP_filtered)
    leptonM_vec = ak.firsts(lepM_filtered)

    reco_arrays = []
    for vec in [leptonP_vec, leptonM_vec]:
        for attr in ["px", "py", "pz", "E"]:
            reco_arrays.append(ak.to_numpy(getattr(vec, attr)))

    reco_features = np.column_stack(reco_arrays)

    print(f"Reco lepton features shape (from el+mu): {reco_features.shape}")
    del el_vecs, mu_vecs, all_lep_vecs, lepP_filtered, lepM_filtered, leptonP_vec, leptonM_vec

### ------------------------------ Calculate and Extract Target Features From Reco ------------------------------ ###

if use_direct_regression:
    # Construct direct regression targets
    target_columns = []

    if any(t in ["cos_theta_star", "cos_theta_ttbar", "cos_x_plus_minus", "cos_N_plus_minus"] for t in direct_targets):
        # Construct t, tbar and ttbar vectors
        top_vec = vector.zip({s: tree_awk[f"parton_top_{s}"] for s in ["pt", "phi", "eta", "m"]})
        antitop_vec = vector.zip({s: tree_awk[f"parton_antitop_{s}"] for s in ["pt", "phi", "eta", "m"]})
        ttbar = top_vec + antitop_vec

        # Frame kinematics
        top_in_CoM = top_vec.boostCM_of(ttbar)
        antitop_in_CoM = antitop_vec.boostCM_of(ttbar)
        Kdirection = top_in_CoM.to_beta3().unit()
        z = vector.obj(x=0, y=0, z=1)
        cos_T = z.dot(Kdirection)

    # mttbar
    if "mttbar" in direct_targets:
        mttbar = tree_awk["parton_ttbar_m"]
        target_columns.append(ak.to_numpy(mttbar))

    # cos_theta_star
    if "cos_theta_star" in direct_targets:
        target_columns.append(ak.to_numpy(cos_T))

    # cos_theta_ttbar
    if "cos_theta_ttbar" in direct_targets:
        norm_top = top_vec.p
        norm_anti = antitop_vec.p
        dot_3 = top_vec.px * antitop_vec.px + top_vec.py * antitop_vec.py + top_vec.pz * antitop_vec.pz
        cos_theta_ttbar = dot_3 / (norm_top * norm_anti)
        target_columns.append(ak.to_numpy(cos_theta_ttbar))

    # cos_x_plus_minus
    if "cos_x_plus_minus" in direct_targets:
        # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
        leptonP_in_CoM     = lepP_truth.boostCM_of(ttbar)
        leptonM_in_CoM     = lepM_truth.boostCM_of(ttbar)
        leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
        leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

        # Compute cos_x_plus/minus
        cos_x_minus = -Kdirection.dot(leptonM_in_antitop.to_beta3().unit())
        cos_x_plus = Kdirection.dot(leptonP_in_top.to_beta3().unit())

        target_columns.append(ak.to_numpy(cos_x_minus))
        target_columns.append(ak.to_numpy(cos_x_plus))

        del leptonP_in_CoM, leptonM_in_CoM, leptonP_in_top, leptonM_in_antitop

    # cos_N_plus_minus
    if "cos_N_plus_minus" in direct_targets:
        # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
        leptonP_in_CoM     = lepP_truth.boostCM_of(ttbar)
        leptonM_in_CoM     = lepM_truth.boostCM_of(ttbar)
        leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
        leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

        leptonP_direction = leptonP_in_top.to_beta3().unit()
        leptonM_direction = leptonM_in_antitop.to_beta3().unit()

        sin_T = (1 - cos_T**2)**0.5
        mask = 1*(cos_T > 0) -1*(cos_T < 0)
        Ndirection = z.cross(Kdirection)/sin_T
        cos_N_minus = -mask*Ndirection.dot(leptonM_direction) 
        cos_N_plus  = mask*Ndirection.dot(leptonP_direction)

        target_columns.append(ak.to_numpy(cos_N_minus))
        target_columns.append(ak.to_numpy(cos_N_plus))

        del leptonP_in_CoM, leptonM_in_CoM, leptonP_in_top, leptonM_in_antitop

    # Stack all selected targets
    target = np.column_stack(target_columns)
    print(f"Target shape: {target.shape}")

    if any(t in ["cos_theta_star", "cos_theta_ttbar"] for t in direct_targets):
        del top_vec, antitop_vec, ttbar, top_in_CoM, Kdirection

else:
    # Kinematic Regression

    # Construct t, tbar and ttbar vectors
    top_vec = vector.zip({s: tree_awk[f"parton_top_{s}"] for s in ["pt", "phi", "eta", "m"]})
    antitop_vec = vector.zip({s: tree_awk[f"parton_antitop_{s}"] for s in ["pt", "phi", "eta", "m"]})

    # Add to target
    target = np.column_stack([
        ak.to_numpy(top_vec.px), ak.to_numpy(top_vec.py), ak.to_numpy(top_vec.pz), ak.to_numpy(top_vec.E),
        ak.to_numpy(antitop_vec.px), ak.to_numpy(antitop_vec.py), ak.to_numpy(antitop_vec.pz), ak.to_numpy(antitop_vec.E)
    ])
    print(f"Target shape: {target.shape}")

    # Mass Loss
    if use_mass_loss:
        ttbar_vec = top_vec + antitop_vec
        mass = np.column_stack([
            ak.to_numpy(top_vec.mass),
            ak.to_numpy(antitop_vec.mass),
            ak.to_numpy(ttbar_vec.mass)
        ])
        print(f"Mass shape: {mass.shape}")
        del ttbar_vec

    del top_vec, antitop_vec

del tree_awk

### ------------------------------ Split into train/validation/test (80:10:10) ------------------------------ ###

from sklearn.model_selection import train_test_split

# First split X,Y,M,L & R into 80% (Train) : 20% (Temp)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, target, test_size=0.2, random_state=67)

if use_mass_loss == True:
    M_train, M_temp = train_test_split(mass, test_size=0.2, random_state=67)
    M_val, M_test = train_test_split(M_temp, test_size=0.5, random_state=67)
    del M_temp, mass

if use_truth_lepton == True:
    L_train, L_temp = train_test_split(lepton_features, test_size=0.2, random_state=67)
    L_val, L_test = train_test_split(L_temp, test_size=0.5, random_state=67)
    del L_temp, lepton_features

if use_pred_lepton == True:
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

with h5py.File(train_file, "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)

    if use_mass_loss == True:
        f.create_dataset("M", data=M_train)
    if use_truth_lepton == True:
        f.create_dataset("L", data=L_train)
    if use_pred_lepton == True:
        f.create_dataset("R", data=R_train)

with h5py.File(val_file, "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)

    if use_mass_loss == True:
        f.create_dataset("M", data=M_val)
    if use_truth_lepton == True:
        f.create_dataset("L", data=L_val)
    if use_pred_lepton == True:
        f.create_dataset("R", data=R_val)

with h5py.File(test_file, "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)

    if use_mass_loss == True:
        f.create_dataset("M", data=M_test)
    if use_truth_lepton == True:
        f.create_dataset("L", data=L_test)
    if use_pred_lepton == True:
        f.create_dataset("R", data=R_test)

with h5py.File(scaler_file, "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_)
    f.create_dataset("Y_scale", data=scaler_Y.scale_)

print(f"Saved training data to {train_file}")
print(f"Saved validation data to {val_file}")
print(f"Saved test data to {test_file}")
print(f"Saved scaling data to {scaler_file}")

