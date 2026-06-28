### ------------------------------ Imports ------------------------------ ###

import numpy as np
import urllib.request # Used for downloading the ttbar dataset from CERNBox.
import uproot # Read in ROOT file format 
import awkward as ak # Used to perform awkward operations on jagged arrays
import matplotlib.pyplot as plt # Used to plot graphs 
import vector # Used to inspect vectors of leptons
import os
import torch

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

file_tt  = uproot.open("ttbar_from_cernbox.root") # Access main file

tree = file_tt["mini"] # Access mini branch
print(tree.keys())

ak_array = tree.arrays(["jet_n","jet_pt","jet_eta","jet_phi","jet_E","jet_jvt","jet_trueflav","jet_truthMatched"])

jet_n = tree['jet_n'].array()
jet_pt = tree['jet_pt'].array()
jet_eta = tree['jet_eta'].array()
jet_phi = tree['jet_phi'].array()
jet_E = tree['jet_E'].array()
jet_jvt = tree['jet_jvt'].array()
jet_trueflav = tree['jet_trueflav'].array()
jet_truthMatched = tree['jet_truthMatched'].array()

lep_type = tree['lep_type'].array()
print(lep_type)

# These all seem to do the exact same thing (wtf)
# print(tree.show)
# print(tree.items)
# print(tree.values)

