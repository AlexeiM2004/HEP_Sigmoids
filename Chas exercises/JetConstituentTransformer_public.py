import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
scaler.fit(X_train.reshape(-1, 5))
X_scaled = scaler.transform(X_train.reshape(-1, 5))
X_scaled = X_scaled.reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 5))
X_test_scaled = X_test_scaled.reshape(X_test.shape)

X_train_tensor = torch.tensor(X_scaled,dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled,dtype=torch.float32)

Y_train_tensor = torch.tensor(Y_train.reshape(-1,1), dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.reshape(-1,1), dtype=torch.float32)

class JetTransformer(nn.Module):
  def __init__(self, in_dim, nhead, num_layers):
    super(JetTransformer, self).__init__()

    self.input_embedder = nn.Sequential(
      nn.Linear(in_dim, 16),
      nn.ReLU(),
      nn.Linear(16,32)
    )

    self.transformer_encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(d_model=32, nhead=nhead,batch_first=True, dim_feedforward = 128),
      num_layers=num_layers,enable_nested_tensor=False)
    
    self.output_classifier_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

  def mean_pooling(self,x):
    """
    Mean pooling implementation
    """
    # x: (batch_size, num_tokens, embed_dim)
    return x.mean(dim=1)
    
  def forward(self, x):
    x = self.input_embedder(x)
    x = self.transformer_encoder(x)
    x = self.mean_pooling(x)
    x = self.output_classifier_head(x)
    return torch.sigmoid(x)

# Load the train and test datasets in TensorDataset
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False, pin_memory=True)
    
model = JetTransformer(in_dim=5, nhead=2, num_layers=2)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

N_epochs = 30
train_losses = []

print("Beginning training")
print()

for epoch in range(N_epochs):
  model.train()
  running_loss = 0.0
  for inputs, targets in train_loader:
    X_batch, Y_batch = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, Y_batch)
    loss.backward()
    optimizer.step()
    running_loss += loss.item() * X_batch.size(0)

  train_loss = running_loss / len(train_loader.dataset)
  train_losses.append(train_loss)
  print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
  torch.cuda.empty_cache()

import matplotlib.pyplot as plt
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # if using GPU

        outputs = model(inputs)
        list_of_predictions.append(outputs)
        loss = criterion(outputs, targets)

pred = torch.concatenate(list_of_predictions)
Y_pred = torch.round(pred).detach().cpu().numpy()

from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_test.reshape(-1,1),Y_pred))