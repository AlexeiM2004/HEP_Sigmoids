import pandas as pd # Library used for data handling
import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
import os # Library used to check if file is present
import urllib.request # Library used to download
from dataclasses import dataclass # Library used to create dataclass to store the penguin input data
from sklearn.linear_model import LinearRegression # Library used for linear regression analysis

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

# Create a class to store the penguin's features such as mass, age, sex and length measurements

@dataclass
class penguin_features:
    flipper_length_mm: list
    body_mass_g: list
    bill_length_mm: list 
    bill_depth_mm: list
    sex: list
    year: list

features = penguin_features( 
    flipper_length_mm = penguins_df["flipper_length_mm"].values,
    body_mass_g = penguins_df["body_mass_g"].values,
    bill_length_mm = penguins_df["bill_length_mm"].values,
    bill_depth_mm = penguins_df["bill_depth_mm"].values,
    sex = penguins_df["sex"].values,
    year = penguins_df["year"].values
)

# Create a class to store the penguin's general data, such as species and island 

@dataclass
class penguin_general:
    species: list
    island: list
    
general = penguin_general(
    species = penguins_df["species"].values,
    island = penguins_df["island"].values 
)

### Multiple Linear (Multilinear) Regression Task

# - Create variable x, now a stacked array containing flipper length and bill depth (multivariable)
# - Create variable y_true, "true" body mass
# - Perform a linear regression fit on the x and y data, and predict
# - Retrieve coefficients from gradients, and intercept
# - Display gradients, intercept, and R squared.
# - Create scatter plot of X[:,0] vs body mass, keeping X[:,1] as a constant mean
# - Create grids and column stack, predict and plot
# - Repeat for plot of X[:,1] vs body mass, keeping X[:,0] as constant mean

multiple_lr_analysis = True
if multiple_lr_analysis == True:

    # Split data into inputs (X features) and body mass (Y)

    X = np.column_stack((features.flipper_length_mm,features.bill_depth_mm)) # Arrays must be stacked
    y_true = features.body_mass_g

    # Create and fit LR model

    model = LinearRegression()
    model.fit(X,y_true)

    # Use model to make predictions
    y_pred = model.predict(X)

    # Retrieve coefficients
    slope_flipper = model.coef_[0]
    slope_bill = model.coef_[1]
    intercept = model.intercept_

    print(f"Equation : body_mass = {slope_flipper:.2f} * flipper_length + {slope_bill:.2f} * bill_depth + {intercept:.2f}")
    print(f"R^2 = {model.score(X,y_true):.2f}")

    # Create scatter plot of flipper length vs body mass

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.scatter(X[:,0], y_true, color = 'blue', alpha = 0.5, label = 'Data points')

    # Create a prediciton for flipper length 
    bill_mean = X[:, 1].mean() # Hold Bill depth at its mean
    flipper_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # Create a grid of evenly spaced flipper points
    X_flipper_grid = np.column_stack((flipper_grid, np.full_like(flipper_grid, bill_mean))) # Stack the arrays, with an array of bill depth set to the mean 
    y_pred_flipper = model.predict(X_flipper_grid)
    ax1.plot(flipper_grid, y_pred_flipper, color='red', label='Predicted (bill depth held at mean)')
    
    ax1.set_xlabel("Flipper length (mm)")
    ax1.set_ylabel("Body mass (g)")
    ax1.set_title('Flipper Length vs Body Mass (2-feature model)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Create scatter plot of bill depth vs mody mass
    fig, ax2 = plt.subplots(figsize=(7, 5))
    
    # Scatter plot
    ax2.scatter(X[:, 1], y_true, color='green', alpha=0.5, label='Data Points')
    
    # Create a prediction for bill dpeth
    flipper_mean = X[:, 0].mean() # Hold flipper length at its mean
    bill_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 100) # Create a grid of evenly spaced bill points
    X_bill_grid = np.column_stack((np.full_like(bill_grid, flipper_mean), bill_grid)) # Stack the arrays,  with an array of flipper length set to the mean 
    y_pred_bill = model.predict(X_bill_grid)
    ax2.plot(bill_grid, y_pred_bill, color='red', label='Predicted (flipper length held at mean)')
    
    ax2.set_xlabel("Bill depth (mm)")
    ax2.set_ylabel("Body mass (g)")
    ax2.set_title('Bill Depth vs Body Mass (2-feature model)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.show()
