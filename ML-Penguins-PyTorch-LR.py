import pandas as pd # Library used for data handling
import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
import os # Library used to check if file is present
import urllib.request # Library used to download
import torch
from torch import nn, optim

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

### Generic standardiser function 

def standardiser(standardise, desired_std=1.0):
    standardise = (standardise - standardise.mean()) / standardise.std() # Shift the mean to be 0, this standardises the mean to 0 and standard deviation to 1
    standardise = standardise * desired_std # Sets the std to desired std
    return standardise

### Load data into PyTorch Tensors, not using dataclasses this time

# Assign penguin's data to tensors 
input_data = torch.tensor(penguins_df[["flipper_length_mm"]].values, dtype =torch.float32)
target = torch.tensor(penguins_df["body_mass_g"].values, dtype =torch.float32)

# Calculate mean and std for standardisation
mean_input = torch.mean(input_data)
mean_output = torch.mean(target)

std_input = torch.std(input_data)
std_output = torch.std(target)


# Standardise penguin data
input_data = standardiser(input_data)
target = standardiser(target)

# Reshape both as 2D column vectors

target = target.reshape(-1,1)

print(input_data.shape)
print(target.shape)

### Apply neural network

# Model is a "linear layer" for a simple linear regression case
# It maps two values Xi (input) to a single value yi(output)

model = nn.Linear(1,1) # This is the key model defined

# The model can be examined via the following

print(model) 
# This displays the model, its linear, it has one input feature, one output feature, and bias turned on (y intercept)
print(list(model.parameters()))
# This displays two random parameters (with grad on for backward pass)

### Training the neural network

# In the SKLearn case, an analytic solution was implicitly used to solve the regression problem
# PyTorch solves the same problem iteratively by minimising the loss function and updating the parameters (5 steps)
# We define a loss function and optimiser

selected_loss_function = nn.MSELoss()
selected_optimiser = optim.Rprop(model.parameters())

# Each iteration now requires a for loop which;
# - Makes a prediction based on current model
# - Computes loss (via selected loss function), i.e. diff btwn predicted and true value
# - Updates model parameters

losses = [] # Keeps track of loss @ every epoch, this is for visualisation purposes.

N_epochs = 100 # Number of epochs we iterate over

for epoch in range(N_epochs):
    # Tell the optimiser to begin an optimisation step
    selected_optimiser.zero_grad()

    # Use mopdel as a prediction function
    predictions = model(input_data)

    # Compute loss (chi squared) between predictions and true values
    loss = selected_loss_function(predictions,target)

    # tell loss function and optimise to end optimisation step
    loss.backward()
    selected_optimiser.step()

    losses.append(loss.item()) # Update the empty losses array used for visualisation of loss at each epoch
    if (epoch + 1) % 10 == 0: # If epoch number + 1 is divisible by 10, print ... 
        print(f'Epoch [{epoch + 1}/{N_epochs}], Loss : {loss.item():.4f}')

#Plot
plt.plot(losses, label = "True curve", color='green', linewidth =5)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss Curve, Epochs vs Loss')
plt.legend()
plt.show()

# Reset model's parameters
reset_parameters = False
if reset_parameters == True:
    model.reset_parameters()
    optimizer = optim.Rprop(model.parameters())

print("Weight (Scaled):", model.weight)
print("Bias (Scaled):", model.bias)

### Evaluating the model 

y_out = model(input_data)
y_pred = y_out.detach()

plt.scatter(input_data,target,color='blue',label='Data Points')
plt.plot(input_data, y_pred,color='red',label='Linear Regression line')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear Regression example')
plt.legend()
plt.show()


### Calculate R squared value 

# R-squared metric
def r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r_squared_value = r_squared(target, y_pred)
print("R squared value",r_squared_value.item())


### Predicting specific values using the model

example_flipper_length_mm = 300.0

standardised_example = (example_flipper_length_mm - mean_input.item()) / std_input.item()

print(example_flipper_length_mm)
print(standardised_example)

example_tensor = torch.tensor([[standardised_example]], dtype=torch.float32)

prediction = model(example_tensor)
print(prediction.item())

predicted_body_mass = prediction.item()*std_output + mean_output 
print(predicted_body_mass)