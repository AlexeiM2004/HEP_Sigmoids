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

import uproot # Read in ROOT file format 
import awkward as ak # Used to perform awkward operations on jagged arrays
import matplotlib.pyplot as plt # Used to plot graphs 
import os # Used to find filepath
import numpy as np

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

### ------------------------------ Spin Observable Boost Function ------------------------------ ###

def boost(top,anti_top,leptonP,leptonM):
    # Build ttbar system
    ttbar = top + anti_top 

    # Boost tops lab --> ttbar_CoM
    top_in_CoM      = top.boostCM_of(ttbar)
    antitop_in_CoM  = anti_top.boostCM_of(ttbar) 

    # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
    leptonP_in_CoM     = leptonP.boostCM_of(ttbar)
    leptonM_in_CoM     = leptonM.boostCM_of(ttbar)
    leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
    leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

    # Compute the unit 3-vector directions
    Kdirection        = top_in_CoM.to_beta3().unit()
    leptonP_direction = leptonP_in_top.to_beta3().unit()
    leptonM_direction = leptonM_in_antitop.to_beta3().unit()

    return Kdirection, leptonP_direction, leptonM_direction

### ------------------------------ Helicity Basis Function ------------------------------ ###

def helicity_basis_observables(Kdirection, leptonP_direction, leptonM_direction):
    # Kinematics
    z     = vector.obj(x=0,y=0,z=1)
    cos_T = z.dot(Kdirection)
    sin_T = (1 - cos_T**2)**0.5
    mask = 1*(cos_T > 0) -1*(cos_T < 0)

    # Helicity basis observables
    cos_K_plus  = Kdirection.dot(leptonP_direction)
    cos_K_minus = -Kdirection.dot(leptonM_direction) 

    Ndirection = z.cross(Kdirection)/sin_T
    cos_N_plus  = mask*Ndirection.dot(leptonP_direction)
    cos_N_minus = -mask*Ndirection.dot(leptonM_direction) 

    Rdirection  =  1/sin_T * (z - cos_T*Kdirection)
    cos_R_plus  = mask*Rdirection.dot(leptonP_direction)
    cos_R_minus = -mask*Rdirection.dot(leptonM_direction) 

    cos_phi     = leptonP_direction.dot(leptonM_direction)

    helicity_observables = ak.zip({
    "cos_K_plus"  :  cos_K_plus,
    "cos_K_minus" :  cos_K_minus,
    "cos_N_plus"  :  cos_N_plus,
    "cos_N_minus" :  cos_N_minus,
    "cos_R_plus"  :  cos_R_plus,
    "cos_R_minus" :  cos_R_minus,
    "cos_phi"     :  cos_phi 
    }, depth_limit=1, with_name="Event")

    return helicity_observables

### ------------------------------ Extract Features from Reco ------------------------------ ###

# Define columns to load
columns_to_load = []
columns_to_load.extend(feature_dict['jet_features']['columns'])
columns_to_load.extend(feature_dict['electron_features']['columns'])
columns_to_load.extend(feature_dict['muon_features']['columns'])
columns_to_load.extend(feature_dict['met_features']['columns'])
columns_to_load.extend([f'parton_{p}_{s}' for p in ['top','antitop','lepton_plus','lepton_minus'] for s in ['pt','phi','eta','m']])
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

### ------------------------------ Kinematic features for target ------------------------------ ###

import vector

Parton_Top = vector.zip({
    "pt": tree_awk["parton_top_pt"],
    "eta": tree_awk["parton_top_eta"],
    "phi": tree_awk["parton_top_phi"],
    "mass": tree_awk["parton_top_m"]})

Parton_AntiTop = vector.zip({
    "pt": tree_awk["parton_antitop_pt"],
    "eta": tree_awk["parton_antitop_eta"],
    "phi": tree_awk["parton_antitop_phi"],
    "mass": tree_awk["parton_antitop_m"]})

Parton_leptonPlus = vector.zip({
    "pt": tree_awk["parton_lepton_plus_pt"],
    "eta": tree_awk["parton_lepton_plus_eta"],
    "phi": tree_awk["parton_lepton_plus_phi"],
    "mass": tree_awk["parton_lepton_plus_m"]})

Parton_leptonMinus = vector.zip({
    "pt": tree_awk["parton_lepton_minus_pt"],
    "eta": tree_awk["parton_lepton_minus_eta"],
    "phi": tree_awk["parton_lepton_minus_phi"],
    "mass": tree_awk["parton_lepton_minus_m"]})

dic_of_spin_observables = helicity_basis_observables(*boost(Parton_Top,
                                                            Parton_AntiTop,
                                                            Parton_leptonPlus,
                                                            Parton_leptonMinus))

mttbar = tree_awk["parton_ttbar_m"]
mttbar_np = ak.to_numpy(mttbar)
obs_fields = ['cos_phi']

target_cos = np.column_stack([ak.to_numpy(dic_of_spin_observables[field]) for field in obs_fields])
target = np.column_stack([target_cos, mttbar_np])


print(f"Target shape: {target.shape}")


### ------------------------------ Clean Memory ------------------------------ ###

del X_features, tree, file_ttbar, Parton_Top, Parton_AntiTop, Parton_leptonPlus, Parton_leptonMinus

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

import h5py

with h5py.File("spin_observable_features_train.h5", "w") as f:
    f.create_dataset("X", data=X_train_scaled)
    f.create_dataset("Y", data=Y_train_scaled)

with h5py.File("spin_observable_features_val.h5", "w") as f:
    f.create_dataset("X", data=X_val_scaled)
    f.create_dataset("Y", data=Y_val_scaled)

with h5py.File("spin_observable_features_test.h5", "w") as f:
    f.create_dataset("X", data=X_test_scaled)
    f.create_dataset("Y", data=Y_test_scaled)

with h5py.File("spin_observable_features_scaler_info.h5", "w") as f:
    f.create_dataset("Y_mean", data=scaler_Y.mean_)
    f.create_dataset("Y_scale", data=scaler_Y.scale_)

print("Saved to spin_observable_features_train.h5")
print("Saved to spin_observable_features_val.h5")
print("Saved to spin_observable_features_test.h5")
print("Saved scaler info to spin_observable_features_scaler_info.h5")

