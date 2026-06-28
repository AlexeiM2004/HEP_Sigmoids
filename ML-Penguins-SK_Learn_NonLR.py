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

### Non-Linear regression analysis task 

# - Create generic polynomial, 
# - Generate data points around that polynomial
# - Add noise to the data points for "randomness"
# - Use Scikit-learn to build polynomial terms and recast for multilinear regression
# - Fit, visualise and predict, similar to linear regression

non_linear_regression = True

if non_linear_regression == True:
    # As previously mentioned, linear regression refers to rltn btwn predictions (outputs) and the parameters (not inouts)
    # Linear regression can also be performed on non-linear functions provided outputs are linear in params

    np.random.seed(4024662462)
    # Generate synthetic data to play with
    x = np.linspace(-2,3,1000).reshape(-1,1) # Generate 1000 samples between -2 and 3. 
    y_true = np.sin(x) # Example polynomial

    # Add noise to true values
    y = y_true + 0.2 * np.random.randn(*y_true.shape) # Generate random noise

    # Use scikit-learn to build polynomial terms and recast for multilinear regression

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly.fit_transform(x)

    # Fit, visualise and predict similarly.

    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict
    y_pred = model.predict(X_poly)

    #Plot
    plt.scatter(x,y,label="Noisy Data")
    plt.plot(x, y_true, label = "True curve", color='green', linewidth =5)
    plt.plot(x, y_pred, label = "Predicted curve", color='blue')
    plt.legend()
    plt.show()

    # We can use linear regression here because the parameters are linear, eve though the polynomial isnt linear
    # x can be defined as different variables and we can then recast the problem as multilinear regression