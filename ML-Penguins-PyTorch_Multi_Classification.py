import pandas as pd # Library used for data handling
import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
import os # Library used to check if file is present
import urllib.request # Library used to download
import torch
import torch.nn as nn

# Check if penguins is downloaded, if not download it (using urllib & os)

filename = "penguins_downloaded.csv"
url = "https://cernbox.cern.ch/s/wh34GhKCOv0Umh7/download"

if not os.path.exists(filename):
    print("Downloading", filename,".")
    urllib.request.urlretrieve(url, "penguins_downloaded.csv")
    print("Download complete.")
else:
    print(filename, "file found.")

## Read penguins into panda

input_penguins_df = pd.read_csv(filename) # Load penguin dataset into "pandas" module

penguins_df = input_penguins_df.dropna(inplace=False) # Rows with entries containing "N/A" or "none" are removed

### Plot bill depth vs bill length, catagorised into the 3 different penguin species
# - Create 3 penguin specific dataframes, using a mask 
# - Plot each species separately
# - Display graph

fig, ax = plt.subplots()

def plot_catagorical_problem(ax, xlow=29, xhigh=61, ylow=12,yhigh=22):
    
    # Create 3 separate penguin species dataframes , using the target array as a mask
    df_adelie = penguins_df[penguins_df["species"] == "Adelie"] # Sets all non-adelie penguins to false, and only reads true (masks out all non-adelie penguins)
    df_gentoo = penguins_df[penguins_df["species"] == "Gentoo"] # Sets all non-gentoo penguins to false, and only reads true (masks of all non-gentoo penguins)
    df_chinstrap = penguins_df[penguins_df["species"] == "Chinstrap"] # Sets all non-chinstrap penguins to false, and only reads true (masks out all non-chinsrap penguins)

    # Plot each species separately with invidiual colours for clarity
    ax.scatter(df_adelie["bill_length_mm"],df_adelie["bill_depth_mm"], color="blue", label="Adelie")
    ax.scatter(df_gentoo["bill_length_mm"],df_gentoo["bill_depth_mm"], color="red", label="Gentoo")
    ax.scatter(df_chinstrap["bill_length_mm"],df_chinstrap["bill_depth_mm"], color="green", label="Chinstrap")

    # Set plot params
    ax.set_xlim(xlow,xhigh)
    ax.set_ylim(ylow,yhigh)
    ax.set_xlabel("bill length (mm)")
    ax.set_ylabel("bill depth (mm)")

    ax.legend(loc="lower left", framealpha=1)

plot_catagorical_problem(ax)
plt.show()

### Main task - PyTorch Multi-Classification, including all penguins

X = penguins_df[["bill_length_mm","bill_depth_mm"]].values # Extract bill length and depth from masked penguins

X_tens = torch.tensor(X,dtype=torch.float32)

target = pd.factorize(penguins_df["species"])[0] # Creates two variables, target (0,1 depending on species type) and species name, chinstrap or adelie 
y_multiclass_tens = torch.tensor(target,dtype=torch.long) 

# Create multi layered perceptron (shallow DNN)

import torch.nn.functional as F

multiclass_model = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,3)
)

# Define loss function

loss = nn.CrossEntropyLoss()

# Define optimiser

optimiser = torch.optim.Rprop(multiclass_model.parameters(),lr=0.1)

# Inspect multi-layer perceptron model
print(multiclass_model)

# Run a training loop

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes.

N_epochs = 1000 # Number of epochs we iterate over

for epoch in range(N_epochs):

    # Use mopdel as a prediction function
    y_pred = multiclass_model(X_tens)

    # Compute loss (chi squared) between predictions and true values
    loss_value = loss(y_pred,y_multiclass_tens)

    # Backward pass + Optimisation

    optimiser.zero_grad()
    loss_value.backward()
    optimiser.step()

    losses.append(loss_value.item()) # Update the empty losses array used for visualisation of loss at each epoch
    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... 
        print(f'Epoch [{epoch + 1}/{N_epochs}], Loss : {loss_value.item():.4f}')

# Display loss curve

def plot_loss_curve(losses):
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss") 
    plt.title("Loss curve")
    plt.show()

plot_loss_curve(losses)

# Utilise predictive power

y_out_tens = multiclass_model(X_tens)
probabilities = F.softmax(y_out_tens,dim=1)
y_out = y_out_tens.detach().numpy()
predictions = np.argmax(y_out,axis=1)

print(predictions)

# Analyse models predictive power

print("Number of misclassifications =",np.count_nonzero(target!=predictions)) # Check how many values match up between the predictions and the actual data

from sklearn.metrics import accuracy_score # Import an accuracy scorer from SKLearn
accuracy = accuracy_score(target, predictions) # Feed in y_true and predicted class
print("Given accuracy (from SKLearn)= ",accuracy) # This says "Out of all penguins, what fraction did the model classify correctly"


from sklearn.metrics import confusion_matrix # Import a confusion matrix from SKLearn
conf_matrix = confusion_matrix(target, predictions)
print("Confusion matrix (from SKLearn)= ",conf_matrix)

# The confusion matrix, given in the form
# | TP FN |
# | FP TN |
# TP = True positive, pulled adelie as adelie
# FN = False negative, pulled chinstrap as adelie
# FP = False Positive, pulled adelie as chinstrap
# TN = True negative, pulled chinstrap as chinstrap
# Confusion matrix explains exactly where the mistatkes are being made

from sklearn.metrics import precision_score, recall_score, f1_score # Import more methods to test the accuracy of the model
precision = precision_score(target, predictions, average='weighted')
recall = recall_score(target, predictions, average='weighted')
f1 = f1_score(target, predictions, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Precision quantifies, of all the times the model predicted a class, what fraction was correct (Minimise false positives)
# Recall quantifies, of all the adelie penguins, how many did the model correctly identify (Minimise false negatives)
# F1-score, the harmonic mean of precision and recall, balanced measure.

# Misclassifications - How many the model got wrong
# Accuracy - What percentage the model got right
# Confusion matrix - Which specific classes got confused
# Precision - How often was adelie right
# Recall - How many adelie was found
# F1 - Whats the balance between precision and recall

# Use the model to group the points

x_data = penguins_df[["bill_length_mm","bill_depth_mm"]].values

x_min,x_max = x_data[:,0].min() -1, x_data[:,0].max() + 1
y_min,y_max = x_data[:,1].min() -1, x_data[:,1].max() + 1

xx,yy = np.meshgrid(np.linspace(x_min,x_max,300),np.linspace(y_min,y_max,300))

grid = np.c_[xx.ravel(),yy.ravel()]
grid_tensor = torch.tensor(grid,dtype=torch.float32)

with torch.no_grad():
    logits = multiclass_model(grid_tensor)
    preds = torch.argmax(logits,dim=1)

preds = preds.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.contour(xx,yy,preds,alpha=0.3,cmap="coolwarm")

plt.scatter(x_data[:,0],x_data[:,1],c=target,cmap="coolwarm",edgecolor="k")

plt.xlabel("Bill length (mm)")
plt.ylabel("Bill depth (mm)")
plt.title("Decision Boundaries of Penguin classifier")
plt.show()


