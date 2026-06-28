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

features = df_background.columns.tolist()
if inspect == True:
    for feature in features:
        compare_distributions(df_background,df_signal,feature) # Plots a comparison graph of selected features

# Create two arrays based on input features to train on

lepton0_energy_signal = np.array(df_signal['lepton0_energy']) # Signal lepton0 
lepton0_energy_bkgrd = np.array(df_background['lepton0_energy']) # Background lepton 0

### Create targets for learning

target_1 = np.ones(len(lepton0_energy_signal)) # Signal is 1
target_0 = np.zeros(len(lepton0_energy_bkgrd)) # Background is 0
targets = np.concatenate((target_1,target_0), axis = 0)
targets = targets.reshape(-1,1)

signal_features = df_signal[features].values
background_features = df_background[features].values

input_data = np.vstack([signal_features, background_features])

# -------------------------------------------------------------------------------------------------- #

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

### Early stopping mechanism 

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience        # How many epochs to wait
        self.min_delta = min_delta      # Minimum improvement to count
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping()

# -------------------------------------------------------------------------------------------------- #

### Training the model

# Create a DNN ( 4 layers, 64 neurons )

model = nn.Sequential(
    nn.Linear(26, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.LeakyReLU(),
    nn.Linear(16, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# Define loss function

loss = nn.BCELoss()

# Define optimiser

optimiser = torch.optim.Adam(model.parameters(), lr=0.05) # Use the ADAM optimiser

# Inspect multi-layer perceptron model
print(model)

# Run a training loop

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes
val_losses = [] # Keeps track of validation loss @ each epoch

# Run a training loop (no batches)
losses = []
val_losses = []

N_epochs = 1000

for epoch in range(N_epochs):
    # Training
    model.train()
    y_pred = model(input_data_train_tens)
    train_loss = loss(y_pred, targets_train_tens)
    
    optimiser.zero_grad()
    train_loss.backward()
    optimiser.step()
    
    losses.append(train_loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        y_val = model(input_data_validate_tens)
        val_loss = loss(y_val, targets_validate_tens)
        val_losses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{N_epochs}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        print(f'Final Epoch before early stop [{epoch+1}/{N_epochs}], Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        break

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

# Utilise predictive power

y_pred = model(input_data_test_tens)
plt.hist(y_pred.detach().numpy()) # Plot a histogram from 0 to 1, number of counts at 0 or 1 shows identification of penguins
plt.show()

y_final = y_pred.detach().numpy() # Extracts the predictions using .detach and converts into a numpy array
predicted_class = np.round(y_final) # Converts the predictions into probabilities, by rounding to either 0 or 1

# Analyse models predictive power

from sklearn.metrics import accuracy_score # Import an accuracy scorer from SKLearn
accuracy = accuracy_score(targets_test_tens, predicted_class) # Feed in y_true and predicted class
print("Given accuracy (from SKLearn)= ",accuracy) # This says "Out of all penguins, what fraction did the model classify correctly"


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Import a confusion matrix from SKLearn
conf_matrix = confusion_matrix(targets_test_tens, predicted_class, normalize='true')
display_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
display_matrix.plot()
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score # Import more methods to test the accuracy of the model
precision = precision_score(targets_test_tens, predicted_class, average='weighted')
recall = recall_score(targets_test_tens, predicted_class, average='weighted')
f1 = f1_score(targets_test_tens, predicted_class, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Compute ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(targets_test_tens.numpy(), y_pred.detach().numpy())
roc_auc = auc(fpr, tpr)
print("ROC-AUC value =",roc_auc)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')  
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------------------------------- #


import numpy as np
from sklearn.metrics import accuracy_score

def permutation_importance(model, input_data_train_tens, targets_train_tens):

    detach_to_binary = lambda x: np.round(model(x).detach().numpy())

    baseline_score = accuracy_score(targets_train_tens, detach_to_binary(input_data_train_tens))  # For a classification task
    importances = []

    for feature_idx in range(input_data_train_tens.shape[1]):
        # Shuffle the values of the current feature
        X_val_shuffled = input_data_train_tens.clone()
        X_val_shuffled[:, feature_idx] = X_val_shuffled[:, feature_idx][torch.randperm(input_data_train_tens.shape[0])]

        # Recalculate the performance
        score_shuffled = accuracy_score(targets_train_tens, detach_to_binary(X_val_shuffled))
        print(score_shuffled)

        # The difference in performance is the importance of this feature
        importances.append(baseline_score - score_shuffled)

    return np.array(importances)

# Usage example
importances = permutation_importance(model, input_data_train_tens, targets_train_tens)
# sort features according to importance
idx = np.argsort(np.asarray(importances))

sorted_features_descending = df_background.columns.to_numpy()[idx]

for i in range(len(sorted_features_descending)):
    print(f"Feature: {sorted_features_descending[i]}, Importance: {importances[idx[i]]:.4f}")

importances = importances[idx]
plt.figure(figsize=(12, 6))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), sorted_features_descending)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()