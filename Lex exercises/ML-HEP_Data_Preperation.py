### ------------------------------ Imports ------------------------------ ###

import urllib.request # Used for downloading the ttbar dataset from CERNBox.
import uproot # Read in ROOT file format 
import awkward as ak # Used to perform awkward operations on jagged arrays
import matplotlib.pyplot as plt # Used to plot graphs 
import vector # Used to inspect vectors of leptons
import os


### ------------------------------ Clear Code Stream ------------------------------ ###

clear = True
if clear == True:
    os.system('cls')

### ------------------------------ File Download ------------------------------ ###

filename = "ttbar_from_cernbox.root"
url = "https://cernbox.cern.ch/s/nrBbuO7bu4wi82W/download"

if not os.path.exists(filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")
else:
    print(f"{filename} file found.")

size = os.path.getsize("ttbar_from_cernbox.root")
print(f"File size: {size / (1024**3):.2f} GB")

### ------------------------------ Retrieve and Inspect Bulk Data ------------------------------ ###

inspect = False

file_tt  = uproot.open("ttbar_from_cernbox.root") # Access main file

tree = file_tt["mini"] # Access mini branch

jet_pt_array = tree["jet_pt"].array() # Access jet pt array

if inspect == True:
    print(file_tt.keys(),"\n") # Print the main tree to inspect its keys (branches)
    print(tree.keys(),"\n") # Print the branch "mini" to inspect its keys
    jet_pt_array.show(20) # Print the first 20 entries in the jet transverse momentum array

### ------------------------------ Using Awkward Operations on Arrays ------------------------------ ###

jet_pt_array = tree["jet_pt"].array()
Njets = ak.num(jet_pt_array) # Counts the number of jets in each event
# Njets.show(20)

# Plot a histogram to inspect the distribution of the number of jets

import matplotlib.pyplot as plt

def plot_histogram(variable_name):
    plt.figure(figsize=(10,6),dpi=100)
    plt.hist(variable_name, bins=10, histtype='step',label='Number of Jets',density=True)
    plt.xlabel('Test')
    plt.ylabel('Test')
    plt.title(f'Histogram')
    plt.legend()
    plt.show()

# plot_histogram(Njets) # Looks like a Boltzmann / Gaussian Distribution

# jet_pt_array.show(10) # Before
jet_pt_array_GeV = jet_pt_array/1e3 # Convert the jet_pt_array to GeV
# jet_pt_array_GeV.show(10) # After

leading_jet = ak.max(jet_pt_array, axis=1) # Computing the maximum jet pT in each event a.k.a the pT of the leading/hardest jet
# leading_jet.show(10)

### ------------------------------ Applying Selection Cuts ------------------------------ ###

## Masking Data, i.e. construct an array of booleans and apply that mask to the data

# There are generally two types of selections to apply
# Per-object, "should individual events pass the cut?" No of events remains unchanged, but No of entries changes
# Per-event, "does a whole entry in an array pass selection?" No of entries remains unchanged, but No of events changes

## Example usage of "Per-object" cut, No of events remains unchanged

pt_greater_than_25_mask = jet_pt_array > 25e3
filtered_jet_pt_array = jet_pt_array[pt_greater_than_25_mask] # All entries lower than 25 GeV have been masked out
# filtered_jet_pt_array.show(10)

# Now mask such that we only include entires above 25 GeV and below 100 GeV (one step)

pt_lesser_than_100_mask = jet_pt_array < 100e3
filtered_jet_pt_array = jet_pt_array[pt_greater_than_25_mask & pt_lesser_than_100_mask]
# filtered_jet_pt_array.show(10)

## Alternatively, one can define the second mask based on the resultant array from the first filter (two step)

# Define the first mask and apply to the original array
pt_get_25_mask = jet_pt_array >= 25e3
jet_pt_array_get_25 = jet_pt_array[pt_get_25_mask]

# Define the second mask from the filtered array and apply it as a subsequent filter
pt_get_100_mask = jet_pt_array_get_25 <= 100e3
jet_pt_array_between_25_and_100 = jet_pt_array_get_25[pt_get_100_mask]

## Example usage of "Per-Event" cut, No of objects remains unchanged

Njets_mask = Njets >= 2 # Must be more than 2 jets (Events) in each row otherwise masked out
masked_Njets = Njets[Njets_mask]
# masked_Njets.show(10)

### ------------------------------ Dealing with Multiple Branches ------------------------------ ###

####################################################################################################
#            RETURN TO THIS SECTION TO PRACTICE USING A DICTIONARY BASED FILING SYSTEM             #
####################################################################################################

# In reality, one must apply any filter to any branch of interest
# This can be done by loading all relevant branches into an array (and applying the filter)

jet_array = tree.arrays(["jet_pt","jet_eta","jet_phi","jet_E","jet_MV2c10"])

# Create a dictionary of aliases so that one can extract individual branches through a dict-like syntax

jet_branch_names = {
    "pt" : "jet_pt",
    "eta" : "jet_eta",
    "phi" : "jet_phi",
    "E" : "jet_E",
    "b_tag" : "jet_MV2c10"
}

jet_array3 = tree.arrays(jet_branch_names.keys(), aliases = jet_branch_names)

pt_get_25_mask =  jet_array3["pt"] >= 25e3
pt_lesser_than_100_mask = jet_array3["pt"] < 100e3
filtered_jet_pt_array = jet_array3[pt_greater_than_25_mask & pt_lesser_than_100_mask] # Apply the object cuts 

Njets = ak.count(filtered_jet_pt_array["pt"], axis=1)
Njets_mask = Njets >= 2 
filtered_events = filtered_jet_pt_array[Njets_mask] # Apply the event cuts

### ------------------------------ Operations with Four-Momenta ------------------------------ ###

lepton_array = tree.arrays(["lep_pt", "lep_eta", "lep_phi", "lep_E"])

# Create four-vector arrays using the dictionary constructor
lepton_vectors = vector.zip({"pt" : lepton_array["lep_pt"],
                             "eta" : lepton_array["lep_eta"],
                             "phi" : lepton_array["lep_phi"],
                             "E" : lepton_array["lep_E"]})

# Display the pt of the leptons
print(lepton_vectors.pt)

# Count the number of leptons in each event
print("Number of leptons in each event,", ak.num(lepton_vectors))

# Add vectors to give combined systems 
combined_lepton_system = lepton_vectors[:,0] + lepton_vectors[:,1] # Sums the first two leptons in each event

print("Combined first two leptons from each event,", combined_lepton_system.m)

## One can also perform boosting (Lorentz transforms). In particular, boosting to the COM (0 mom) frame
l1_in_CMF = lepton_vectors[:,0].boostCM_of(combined_lepton_system) # Boost 1st lepton to COM frame of combined system
l2_in_CMF = lepton_vectors[:,1].boostCM_of(combined_lepton_system) # Boost 2nd lepton "                             "

# (l1_in_CMF+l2_in_CMF).show(20) # Check sum of boosted leptons in CMF

### ------------------------------ Histogramming and Plotting ------------------------------ ###

# One can fast plot histograms straight from arrays of data

# First remove entries with no events by masking out =< 0

Njets = ak.num(jet_array3["pt"]) # Counts the number of jets in each event
Njets_mask = Njets > 0 # Must be more than 0 jets (Events) in each row otherwise masked out
leading_jet_pt = jet_array3["pt"][Njets_mask][:, 0]  # Get the leading jet pT (first jet in each event)

# Optionally one can use "Hist" a more powerful histogram plotting tool

from hist import Hist, axis
import mplhep as hep
jet_pt_hist = Hist(axis.Regular(bins=10, start=0, stop=1e5, name="x"))
jet_pt_hist.fill(leading_jet_pt)
hep.histplot(jet_pt_hist)

####################################################################################################
#              RETURN TO THIS SECTION TO PRACTICE USING HISTOGRAMS FOR VISUAL ANALYSIS             #
####################################################################################################

### ------------------------------ Histogramming and Plotting ------------------------------ ###

