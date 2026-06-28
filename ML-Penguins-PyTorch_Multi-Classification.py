import pandas as pd # Library used for data handling
import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
import os # Library used to check if file is present
import urllib.request # Library used to download
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

### Pre-Task - Classification intro

# Use a mask to filter out any non adelie / chinstrap penguins

target,species_names = pd.factorize(penguins_df["species"]) # Creates three variables, target (0,1 depending on species type) and species name, chinstrap or adelie 

# Debug inspection

inspect = True
if inspect == True:
    print(target,"\n")
    print(species_names,"\n")

# Optionally one can interate through the entire array, replacing all Adelie penguins with 0 and all Chinstrap penguins with 1
# Factorise is however, much more efficient (and easier)
# target = []
# for species in df_filtered["species"]:
#     if species == "Adelie":
#         target.append(0)
#     else:
#         target.append(1)
# target = np.array(target)


X = penguins_df[["bill_length_mm","bill_depth_mm"]].values
y_true = target

# Creatre linear regression model from SKL

model = LogisticRegression()
best_fit = model.fit(X,y_true) # Apply best fit by appling parameters to best fit

random_datapoint_features = X[330].reshape(-1,2) # Reshape such that its a double column vector, or (1,2)
random_datapoint_probabilities = best_fit.predict(random_datapoint_features)
print(X[330])
print("Correct class = ",random_datapoint_probabilities)
