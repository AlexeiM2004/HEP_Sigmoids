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
    # Column-wise standardisation
    mean = standardise.mean(dim=0, keepdim=True)  # Mean of each column
    std = standardise.std(dim=0, keepdim=True)    # Std of each column
    standardise = (standardise - mean) / std
    standardise = standardise * desired_std
    return standardise

### Load data into PyTorch Tensors, not using dataclasses this time

# Assign penguin's data to tensors 
input_data = torch.tensor(penguins_df[["flipper_length_mm","bill_depth_mm","bill_length_mm","year"]].values, dtype =torch.float32)
target = torch.tensor(penguins_df["body_mass_g"], dtype =torch.float32)

# Inspect penguin data
debug = False
if debug == True:
    print(input_data.shape)
    print(target.shape)

if debug == True:
    print("=== INPUT DATA DEBUG ===")
    print("Shape:", input_data.shape)
    print("First 5 rows:")
    print(input_data[:5])
    print("Min:", input_data.min().item())
    print("Max:", input_data.max().item())
    print("Mean:", input_data.mean().item())
    print("Std:", input_data.std().item())

# Calculate mean and std for standardisation

mean_input_1 = torch.mean(input_data[:,0])
mean_input_2 = torch.mean(input_data[:,1])
mean_input_3 = torch.mean(input_data[:,2])
mean_input_4 = torch.mean(input_data[:,3])
mean_output = torch.mean(target)


std_input_1 = torch.std(input_data[:,0])
std_input_2 = torch.std(input_data[:,1])
std_input_3 = torch.std(input_data[:,2])
std_input_4 = torch.std(input_data[:,3])
std_output = torch.std(target)

# Standardise penguin data
input_data = standardiser(input_data)
target = standardiser(target)

# Reshape target as a 2D column vector

target = target.reshape(-1,1)

### Apply neural network

# Model is a "linear layer" for a simple linear regression case
# It maps a single value Xi (input) to a single value yi(output)

model = nn.Linear(4,1) # This is the key model defined

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

N_epochs = 1000 # Number of epochs we iterate over

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

### Calculate R squared value 

# R-squared metric
def r_squared(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r_squared_value = r_squared(target, y_pred)
print("R squared value =",r_squared_value.item())

### Predicting specific values using the model

example_flipper_length_mm = 190.0
example_bill_depth_mm = 20.0
example_bill_length_mm = 37.8
example_year = 2009

# Standardise the example values

standardised_example_flipper = (example_flipper_length_mm - mean_input_1.item()) / std_input_1.item()
standardised_example_depth = (example_bill_depth_mm - mean_input_2.item()) / std_input_2.item()
standardised_example_length = (example_bill_length_mm - mean_input_3.item()) / std_input_3.item()
standardised_example_year = (example_year - mean_input_4.item()) / std_input_4.item()

# Convert example values into tensor such that they can be fed into the model

example_tensor = torch.tensor([[standardised_example_flipper,standardised_example_depth,standardised_example_length,standardised_example_year]], dtype=torch.float32)

# Feed into model to predict

prediction = model(example_tensor)

if debug == True:
    print("=== MODEL WEIGHTS ===")
    print("Weight 1 (flipper):", model.weight[0, 0].item())
    print("Weight 2 (bill depth):", model.weight[0, 1].item())
    print("Bias:", model.bias.item())
    print("=== END ===")

    print("=== DEBUG ===")
    print("mean_output:", mean_output.item())   # Should be ~4200
    print("std_output:", std_output.item())     # Should be ~800
    print("prediction (scaled):", prediction.item())
    print("=== END ===")

# Unstandardise to yield answer

def unstandardiser(unstandardise):
    predicted_body_mass = unstandardise.item()*std_output + mean_output
    return predicted_body_mass

print("Predicted body mass from an example flipper length of",example_flipper_length_mm,",bill depth of", example_bill_depth_mm,",bill length of", example_bill_length_mm,",in year", example_year,"gives a body mass = ", unstandardiser(prediction).item())
