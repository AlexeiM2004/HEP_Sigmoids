import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    print(f"Using device: {device}")
    print()

df_signal = pd.read_parquet("signal.parquet")
df_background = pd.read_parquet("background.parquet")

# define the features you are interested in
input_features = ['lepton0_px', 'lepton0_py', 'lepton0_pz', 'lepton0_energy',
'lepton1_px', 'lepton1_py', 'lepton1_pz', 'lepton1_energy',
'jet0_px', 'jet0_py', 'jet0_pz', 'jet0_energy',
'jet1_px', 'jet1_py', 'jet1_pz', 'jet1_energy',
'Njets', 'HT_all', 'MissingEnergy',
'jet0_mass', 'jet1_mass', 'combined_leptons_mass',
'angle_between_jets', 'angle_between_leptons']

df_signal_filtered = df_signal[input_features]
df_background_filtered = df_background[input_features]

# Set targets for training
y_signal     = np.ones(len(df_signal_filtered))
y_background = np.zeros(len(df_background_filtered))

# Combine the dataframes as one big numpy array
input_data = np.concatenate((df_signal_filtered, df_background_filtered), axis=0)
target     = np.concatenate((y_signal, y_background), axis=0)

# split data into train, validation, and test sets (You can also do the shuffle here, if not shuffled before)
X_train, X_test, Y_train, Y_test = train_test_split(input_data, target, test_size = 0.2, shuffle=True, random_state=24)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, shuffle=True, random_state=24)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Convert all the data sets into the PyTorch tensor format, e.g.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).reshape(-1,1)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).reshape(-1,1)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).reshape(-1,1)

# Just write this as nn.Sequential
N_features = len(input_features)

class SimpleDNN(nn.Module):
    def __init__(self, N_input_features): # You can add more parameters here, such that the size of all layers can be
        # defined in the constructor
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleDNN, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(N_input_features, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32,1)

    def forward(self, x):
        # Compute the forward pass.
        x1     = F.relu(self.linear1(x))
        x2     = self.dropout(F.relu(self.linear2(x1)))
        
        y_pred = F.sigmoid(self.linear3(x2))
        return y_pred
    
N_features = len(input_features)
model = SimpleDNN(N_features)
model = model.to(device)

'''
model = nn.Sequential(nn.Linear(N_features,128),
                      nn.ReLU(),
                      nn.Linear(128,32),
                      nn.ReLU(),
                      nn.Dropout(0.3),
                      nn.Linear(32,1),
                      nn.Sigmoid())
'''

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

#Customised scheduler class that adds tolerance for spiking losses

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, tolerance = 100):
        self.patience = patience        # How many epochs to wait
        self.min_delta = min_delta      # Minimum improvement to count
        self.tolerance = tolerance      # Minimum spiking for counter to not increase
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement
        else:
            if val_loss - self.best_loss < self.tolerance:
                self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

from torch.utils.data import TensorDataset, DataLoader

# Create TensorDatasets that combine input features with labels
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset =   TensorDataset(X_val_tensor, Y_val_tensor)


# Create DataLoader objects for training, validation, and testing in batches
BATCH_SIZE = 8192

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,      # Uses multiple CPU cores to prepare batches
    pin_memory=True
)

val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=4,      # Uses multiple CPU cores to prepare batches
    pin_memory=True
)

def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

model.apply(reset_weights)

train_losses = []
val_losses   = []

N_epochs = 1000

Earl = EarlyStopping(30,0.0003, 0.015)

def Batch_Train(model):
    for epoch in range(N_epochs):

        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{N_epochs}], Loss: {train_loss:.4f}')

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                val_predictions = model(inputs)

                batch_val_loss = criterion(val_predictions, targets).item()

                running_val_loss += batch_val_loss * inputs.size(0)

            val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

        Earl(val_loss)
        if Earl.early_stop:
            print("-------------------------")
            print("Early stopping triggered.")
            print(f'Final Epoch [{epoch +1}/{N_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print("-------------------------")
            break
    return

def Train(model, X_train_t, Y_train_t, X_val_t, Y_val_t):
    # Move the ENTIRE dataset to GPU once before starting the loop
    X_train_device = X_train_t.to(device)
    Y_train_device = Y_train_t.to(device)
    X_val_device = X_val_t.to(device)
    Y_val_device = Y_val_t.to(device)

    for epoch in range(N_epochs):
        model.train()
        optimizer.zero_grad()

        # Feed everything forward at once
        predictions = model(X_train_device)
        loss = criterion(predictions, Y_train_device)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{N_epochs}], Train Loss: {train_loss:.4f}')

        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_device)
            val_loss = criterion(val_predictions, Y_val_device).item()
            val_losses.append(val_loss)

        Earl(val_loss)
        if Earl.early_stop:
            print("-------------------------")
            print("Early stopping triggered.")
            print(f'Final Epoch [{epoch + 1}/{N_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print("-------------------------")
            break
    return

if __name__ == '__main__':
    train_losses = []
    val_losses   = []
    Earl.counter = 0  
    Earl.best_loss = float('inf')
    Earl.early_stop = False
    model.apply(reset_weights)

    #Batch_Train(model)
    Train(model, X_train_tensor, Y_train_tensor, X_val_tensor, Y_val_tensor)

    # Plot loss function for training and validation sets
    plt.figure(figsize=(8, 6))

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Validation Loss'])


    # Evaluate the model on the test dataset
    # per default  PyTorch will add the predicted values to the computation graph
    # Call the detach() method to remove them
    model.eval()
    with torch.no_grad():
        test_pred_tensor = model(X_test_tensor.to(device)) 
        test_pred = test_pred_tensor.cpu().numpy()

    # Filter the predicted events for true Signal and true Background
    sig_scores = test_pred[Y_test == 1]
    back_scores = test_pred[Y_test == 0]

    # Plot the predicted scores for both true Signal and true Background events
    plt.figure(figsize=(8, 6))
    plt.hist(sig_scores,
             bins=40,
             range=(0,1),
             density=True,
             histtype='step',
             label='True Signal',
             alpha = 0.5)

    plt.hist(back_scores,
             bins=40,
             range=(0,1),
             density=True,
             histtype='step',
             label='True Background',
             alpha = 0.5)


    plt.legend()
    plt.xlabel("Predicted Scores")
    plt.ylabel("Density")
    plt.xlim((0,1))


    final_prediction = np.round(test_pred)

    # Compute the accuracy_score

    accuracy = accuracy_score(Y_test, final_prediction)
    print(f'Accuracy rating: {accuracy:.4f}%')

    # Caclulate some other metrics that are suitable for a classification
    cm = confusion_matrix(Y_test, final_prediction, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", 'Signal'])
    disp.plot()


    # Compute the ROC

    false_pos, true_pos, thresh = roc_curve(Y_test, test_pred)

    # Compute AUC (Area Under Curve)

    area_u_curve = auc(false_pos,true_pos)

    # Plot ROC curve
    lin_x = np.linspace(0,1,2)
    lin_y = lin_x

    plt.figure(figsize=(8, 6))

    plt.plot(false_pos,true_pos)
    plt.plot(lin_x,lin_x, 'r--', alpha =0.4)

    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.annotate(f'Area under curve: {area_u_curve:.4f}',(0.6,0.2),xycoords='axes fraction')

    def permutation_importance(model, X_val_tensor, y_val):
        model.eval()

        detach_to_binary = lambda x: np.round(model(x.to(device)).detach().cpu().numpy())

        baseline_score = accuracy_score(y_val, detach_to_binary(X_val_tensor))  # For a classification task
        importances = []

        for feature_idx in range(X_val_tensor.shape[1]):
            # Shuffle the values of the current feature
            X_val_shuffled = X_val_tensor.clone()
            X_val_shuffled[:, feature_idx] = X_val_shuffled[:, feature_idx][torch.randperm(X_val.shape[0])]

            # Recalculate the performance
            score_shuffled = accuracy_score(y_val, detach_to_binary(X_val_shuffled))
            #print(score_shuffled)

            # The difference in performance is the importance of this feature
            importances.append(baseline_score - score_shuffled)

        return np.array(importances)

    # Usage example
    importances = permutation_importance(model, X_val_tensor, Y_val_tensor)

    # sort features according to importance
    idx = np.argsort(np.asarray(importances))

    sorted_features_descending = np.array(input_features)[idx]

    '''
    for i in range(len(sorted_features_descending)):
        print(f"Feature: {sorted_features_descending[i]}, Importance: {importances[idx[i]]:.4f}")
    '''

    # Plot feature importances
    importances = importances[idx]
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), sorted_features_descending)
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()