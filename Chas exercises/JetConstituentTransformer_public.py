import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

with h5py.File("part_transformer.h5","r") as f:
  signal_data = f["signal"][:]
  bkg_data = f["bkg"][:]

signal = structured_to_unstructured(signal_data)
background = structured_to_unstructured(bkg_data)

X = np.concatenate([signal,background])
X.shape #Already in the transformer shape we need i.e. Ndata x Ntokens x Nfeatures

target = np.concatenate([np.ones(signal.shape[0]), np.zeros(background.shape[0])], axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size=0.15, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, 4))
X_scaled = scaler.transform(X_train.reshape(-1, 4))
X_scaled = X_scaled.reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 4))
X_test_scaled = X_test_scaled.reshape(X_test.shape)