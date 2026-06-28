import pandas as pd # Library used for data handling
import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
import os # Library used to check if file is present
import urllib.request # Library used to download
from dataclasses import dataclass # Library used to create dataclass to store the penguin input data
from sklearn.linear_model import LogisticRegression # Library used for linear regression analysis

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

df_filtered = penguins_df[penguins_df["species"] != "Gentoo"] # Sets all chinstrap penguins to false, and only reads true (masks out all non-adelie penguins)
target,species_names = pd.factorize(df_filtered["species"]) # Creates two variables, target (0,1 depending on species type) and species name, chinstrap or adelie 

X = df_filtered[["bill_length_mm","bill_depth_mm"]].values
y_true = target

# Creatre linear regression model from SKL

model = LogisticRegression()
best_fit = model.fit(X,y_true) # Apply best fit by appling parameters to best fit

random_datapoint_features = X[213].reshape(-1,2) # Reshape such that its a double column vector, or (1,2)
random_datapoint_probabilities = best_fit.predict(random_datapoint_features)
print(X[213])
print("Correct class = ",random_datapoint_probabilities)

