import h5py
import pandas as pd
import numpy as np
import torch
from torch import nn
import sklearn
import requests
import io
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------- #

url = "https://cernbox.cern.ch/remote.php/dav/public-files/icjK5HWChdTcdb2/WW_vs_TT_dataset.h5"

# Download file
response = requests.get(url)
response.raise_for_status() # Check for download errors

# Load file into a in-memory binary stream
H_vs_TT_dataset = io.BytesIO(response.content)

# Load file
file = h5py.File(H_vs_TT_dataset, 'r')

# Extract data
df_signal       = pd.DataFrame(file['Signal'][:])
df_background   = pd.DataFrame(file['Background'][:])

# Inspect the data
inspect = False
if inspect == True:
    print("Background file shape", df_background.shape)
    print("Signal file shape",df_signal.shape,"\n")
    print("Background column metadata",df_background.columns)
    print("Signal column metadata",df_signal.columns)

# Visualise the data
def compare_distributions(signal_data, background_data, variable_name):
    plt.figure(figsize=(10,6),dpi=100)
    plt.hist(signal_data[variable_name], bins=40, histtype='step',label='Signal',density=True)
    plt.hist(background_data[variable_name], bins=40, histtype='step',label='Background',density=True)
    plt.xlabel(variable_name)
    plt.ylabel('Density')
    plt.title(f'Distribution of {variable_name}')
    plt.legend()
    plt.show()

# -------------------------------------------------------------------------------------------------- #

# Plot all distributions to compare from

inspect_lepton0 = False
if inspect_lepton0 == True:
    compare_distributions(df_background,df_signal,'lepton0_px')
    compare_distributions(df_background,df_signal,'lepton0_py')
    compare_distributions(df_background,df_signal,'lepton0_pz')
    compare_distributions(df_background,df_signal,'lepton0_energy')

inspect_lepton1 = False
if inspect_lepton1 == True:
    compare_distributions(df_background,df_signal,'lepton1_px')
    compare_distributions(df_background,df_signal,'lepton1_py')
    compare_distributions(df_background,df_signal,'lepton1_pz')
    compare_distributions(df_background,df_signal,'lepton1_energy')

inspect_jet0 = False
if inspect_jet0 == True:
    compare_distributions(df_background,df_signal,'jet0_px')
    compare_distributions(df_background,df_signal,'jet0_py')
    compare_distributions(df_background,df_signal,'jet0_pz')
    compare_distributions(df_background,df_signal,'jet0_energy')

inspect_jet1 = False
if inspect_jet1 == True:
    compare_distributions(df_background,df_signal,'jet1_px')
    compare_distributions(df_background,df_signal,'jet1_py')
    compare_distributions(df_background,df_signal,'jet1_pz')
    compare_distributions(df_background,df_signal,'jet1_energy')

inspect_misc = False
if inspect_misc == True:
    compare_distributions(df_background,df_signal,'Njets')
    compare_distributions(df_background,df_signal,'HT_all')
    compare_distributions(df_background,df_signal,'MissingEnergy')

# -------------------------------------------------------------------------------------------------- #

### Create list of features to train on 

features = []
for feature in features:
    compare_distributions(df_background,df_signal,feature) # Plots a comparison graph of selected features

# Create two arrays based on input features to train on

lepton0_px_signal = np.array(df_signal['lepton0_px']) # Signal lepton0
lepton0_px_bkgrd = np.array(df_background['lepton0_px']) # Background lepton0
lepton0_energy_signal = np.array(df_signal['lepton0_energy']) # Signal lepton0 
lepton0_energy_bkgrd = np.array(df_background['lepton0_energy']) # Background lepton 0

# Extra arrays for potential multiclassification 

lepton0_py_signal = np.array(df_signal['lepton0_py']) # Signal lepton0
lepton0_py_bkgrd = np.array(df_background['lepton0_py']) # Background lepton0
lepton0_pz_signal = np.array(df_signal['lepton0_pz']) # Signal lepton0
lepton0_pz_bkgrd = np.array(df_background['lepton0_pz']) # Background lepton0
combined_leptons_mass_signal = np.array(df_signal['combined_leptons_mass']) # Signal  
combined_leptons_mass_bkgrd = np.array(df_background['combined_leptons_mass']) # Background
angle_between_leptons_signal = np.array(df_signal['angle_between_leptons']) # Signal
angle_between_leptons_bkgrd = np.array(df_background['angle_between_leptons']) # Background
missing_energy_signal = np.array(df_signal['MissingEnergy']) # Signal lepton0
missing_energy_bkgrd = np.array(df_background['MissingEnergy']) # Background lepton0

### Create targets for learning

target_1 = np.ones(len(lepton0_energy_signal)) # Signal is 1
target_0 = np.zeros(len(lepton0_energy_bkgrd)) # Background is 0
targets = np.concatenate((target_1,target_0), axis = 0)
targets = targets.reshape(-1,1)

# -------------------------------------------------------------------------------------------------- #

### Concat input data into input_data array (later utilise features to concatenate all arrays into one input data)

lepton_px = np.concatenate((lepton0_px_bkgrd,lepton0_px_signal), axis=0)
lepton_py = np.concatenate((lepton0_py_bkgrd,lepton0_py_signal), axis=0)
lepton_pz = np.concatenate((lepton0_pz_bkgrd,lepton0_pz_signal), axis=0)
combined_leptons_mass = np.concatenate((combined_leptons_mass_bkgrd,combined_leptons_mass_signal), axis=0)
angle_between_leptons = np.concatenate((angle_between_leptons_bkgrd,angle_between_leptons_signal), axis=0)
missing_energy = np.concatenate((missing_energy_bkgrd,missing_energy_signal), axis=0)


input_data = np.column_stack([lepton_px,lepton_py,lepton_pz,combined_leptons_mass,angle_between_leptons,missing_energy])


from sklearn.model_selection import train_test_split

# First split data into 80% train, 20% temp (to split)

input_data_train,input_data_split,targets_train,targets_split = train_test_split(input_data,targets, test_size=0.2, shuffle=True)

# Use 20% split to convert into 10% validation and 10% testing

input_data_validate,input_data_test,targets_validate,targets_test = train_test_split(input_data_split,targets_split, test_size=0.5, shuffle=True)

# We now have 6 arrays, two arrays (80% of data respectively) are to train, two arrays (10%) are validation, two arrays (10%) are for testing

# -------------------------------------------------------------------------------------------------- #

### Normalise the data

# Use SKLearn standardiser

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # SKLearn's in house z score standardiser function

input_data_train_std = scaler.fit_transform(input_data_train) # Standardise the training data, using the training data's mean and std
input_data_test_std = scaler.transform(input_data_test) # Standardises the test data using the training data's mean and std
input_data_validate_std = scaler.transform(input_data_validate) # Standardises the test data using the training data's mean and std

# -------------------------------------------------------------------------------------------------- #

### Convert all 6 arrays into tensors

input_data_train_tens = torch.tensor(input_data_train_std, dtype=torch.float32)
targets_train_tens = torch.tensor(targets_train, dtype=torch.float32) 
input_data_validate_tens = torch.tensor(input_data_validate_std, dtype=torch.float32)
targets_validate_tens = torch.tensor(targets_validate, dtype=torch.float32)
input_data_test_tens = torch.tensor(input_data_test_std, dtype=torch.float32)
targets_test_tens = torch.tensor(targets_test, dtype=torch.float32)

# Inspect the array shapes to make sure they look sensible

if inspect == True:
    print(input_data_train_tens.shape)
    print(targets_train_tens.shape)
    print(input_data_validate_tens.shape)
    print(targets_validate_tens.shape)
    print(input_data_test_tens.shape)
    print(targets_test_tens.shape)

# -------------------------------------------------------------------------------------------------- #

# Batch training

from torch.utils.data import TensorDataset, DataLoader

# Combine features and lables into datasets

train_dataset = TensorDataset(input_data_train_tens,targets_train_tens) # Data paired with corresponding labels
validate_dataset = TensorDataset(input_data_validate_tens,targets_validate_tens)
test_dataset =TensorDataset(input_data_test_tens,targets_test_tens)

batch_size = 1024 # Define batch size for dataloaders

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------------------------------------------------- #

### Training the model

# Create a DNN ( 2 layers, 32 neurons )

model = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# Define loss function

loss = nn.BCELoss()

# Define optimiser

optimiser = torch.optim.Adam(model.parameters(), lr=0.025) # Use the ADAM optimiser

# Inspect multi-layer perceptron model
print(model)

# Run a training loop

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch

N_epochs = 50 # Number of epochs we iterate over

for epoch in range(N_epochs):

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

    model.eval() # Use this to keep track of the validation loss at each step
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in validate_loader:
            y_pred = model(batch_x)
            val_loss = loss(y_pred, batch_y)
            epoch_val_loss += val_loss.item()
    
    avg_val_loss = epoch_val_loss / len(validate_loader)
    val_losses.append(avg_val_loss)

    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... Training
        print(f'Epoch [{epoch + 1}/{N_epochs}], Training Loss : {avg_train_loss:.4f},  Validation Loss : {avg_val_loss:.4f}')

# -------------------------------------------------------------------------------------------------- #

# Display loss curve

def plot_loss_curve(losses,val_losses):
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss") 
    plt.title("Loss curve")
    plt.legend()
    plt.show()

plot_loss_curve(losses,val_losses)

# -------------------------------------------------------------------------------------------------- #

