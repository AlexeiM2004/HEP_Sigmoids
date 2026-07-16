### ------------------------------ Imports ------------------------------ ###

import matplotlib.pyplot as plt # Used to plot graphs 
import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import time

from datetime import datetime

### ------------------------------ Print Current Timestamp ------------------------------ ###

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Job started at :", formatted_time)

### ------------------------------ Device Usage ------------------------------ ###

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device with number of GPUs: {torch.cuda.device_count()}")

### ------------------------------ Load Preprocessed Data ------------------------------ ###

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as f:
            self.X = torch.tensor(f["X"][:], dtype=torch.float32)
            self.Y = torch.tensor(f["Y"][:], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ------------------------------ DataLoaders ------------------------------ #

from torch.utils.data import TensorDataset, DataLoader

batch_size = 4096

dataset_train = CustomDataset("../train_inputs/smaller_ttbar_train.h5")
dataset_val = CustomDataset("../train_inputs/smaller_ttbar_val.h5")
dataset_test = CustomDataset("../train_inputs/smaller_ttbar_test.h5")

with h5py.File("../train_inputs/smaller_ttbar_train.h5", "r") as f:
    X_train= f["X"][:]
    Y_train_scaled= f["Y"][:]

with h5py.File("../train_inputs/smaller_ttbar_test.h5", "r") as f:
    X_test = f["X"][:]
    Y_test_scaled = f["Y"][:]

# ------------------------------ Flow-Matching ------------------------------ #

class ContextEmbeddor(nn.Module):
    def __init__(self, Ninputs, Nembed):
        super(ContextEmbeddor, self).__init__()
        self.fc1 = nn.Linear(Ninputs, 128)
        self.fc2 = nn.Linear(128, Nembed)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConditionalVelocityNet(nn.Module):
    def __init__(self,  Ninputs, Ncontext, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Ninputs + 1 + Ncontext, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, t, c):
        inp = torch.cat([x, t, c], dim=1)
        return self.net(inp)

def conditional_flow_matching_loss(VelocityNet, Embedder, train_X_batch, train_y_batch):
    
    # Conditional target sampling
    x1 = train_y_batch.unsqueeze(1)

    # Sample batch
    batch_size = train_X_batch.shape[0]
    x0 = torch.randn_like(x1)
    
    # Use context embeddor to get context from train_X_batch
    c = Embedder(train_X_batch)
    
    # Sample time t
    t = torch.rand(batch_size,1,device=device)
    
    # Interpolate between x0 and x1
    xt = (1 - t) * x0 + t * x1

    v_pred = VelocityNet(xt, t, c)
    v_target = x1 - x0

    return ((v_pred - v_target) ** 2).mean()

size = 100000

X_train = torch.tensor(X_train[0:size], dtype=torch.float32)
Y_train_scaled = torch.tensor(Y_train_scaled[0:size], dtype=torch.float32)

N = X_train.shape[0]
batch_size = 256
num_epochs = 800

Embedder = ContextEmbeddor(Ninputs=X_train.shape[1],Nembed=32).to(device)
VelNet = ConditionalVelocityNet(Ninputs=1, Ncontext=32).to(device)

optimiser = torch.optim.Adam(
    list(VelNet.parameters()) + list(Embedder.parameters()),
    lr=1e-3
)

losses = []
for epoch in range(num_epochs):
    perm = torch.randperm(N)

    total_loss = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]

        X_batch = X_train[idx].to(device)
        y_batch = Y_train_scaled[idx].to(device)

        optimiser.zero_grad()

        loss = conditional_flow_matching_loss(
            VelNet,
            Embedder,
            X_batch,
            y_batch, 
        )

        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(idx)

    avg_loss = total_loss / N
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("../plots/flowmatching_loss.png")

def sample_flow(model, embedder, test_X, n_steps=100, device='cuda'):
    with torch.no_grad():
        B = test_X.shape[0]

        # Condition
        c = embedder(test_X.to(device))

        # Initial noise
        x = torch.randn(B, 1, device=device)

        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B, 1), i / n_steps, device=device)
            v = model(x, t, c)
            x = x + dt * v

        return x

def sample_flow_repeated(model, embedder, test_X, n_steps=100, n_samples=1, device='cuda'):
    all_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples = sample_flow(model, embedder, test_X, n_steps=n_steps, device=device)
            all_samples.append(samples)
        return all_samples

def sample_flow_many(
    model,
    embedder,
    test_X,
    n_samples=50,
    n_steps=100,
    device="cuda",
):
    with torch.no_grad():
        B = test_X.shape[0]
        S = n_samples

        # Condition (B, C)
        c = embedder(test_X.to(device))

        # Expand condition to (B, S, C)
        c = c.unsqueeze(1).expand(B, S, c.shape[-1])

        # Initial noise (B, S, 1)
        x = torch.randn(B, S, 1, device=device)

        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B, S, 1), i / n_steps, device=device)

            # Flatten batch for model call
            x_flat = x.reshape(B * S, 1)
            t_flat = t.reshape(B * S, 1)
            c_flat = c.reshape(B * S, c.shape[-1])

            v = model(x_flat, t_flat, c_flat)

            # Restore shape
            v = v.reshape(B, S, 1)

            x = x + dt * v

        return x  # (B, S, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test_scaled = torch.tensor(Y_test_scaled, dtype=torch.float32)

p = sample_flow_many(VelNet, Embedder, X_test, n_samples=2, n_steps=100, device=device)
x_pred_single = sample_flow(VelNet, Embedder, X_test, n_steps=200, device=device).squeeze()
x_pred = sample_flow_repeated(VelNet, Embedder, X_test, n_steps=200, n_samples=50, device=device)
# Turn into tensor
x_pred_tensor = torch.cat(x_pred, dim=1).squeeze()
x_pred_mean = torch.mean(x_pred_tensor,dim=1)

B = x_pred_tensor.shape[0]
idx = torch.randint(0, 50, (B,),device=device)
m_hat = x_pred_tensor[torch.arange(B,device=device), idx]

# 5. Plotting results
plt.figure(figsize=(8, 6))
plt.scatter(Y_test_scaled, x_pred_single.cpu(), alpha=0.4, label="Single Trajectory", color="blue")
#plt.scatter(Y_test_scaled, x_pred_mean.cpu(), alpha=0.4, label="Mean of 50 Trajectories", color="orange")
plt.plot([Y_test_scaled.min(), Y_test_scaled.max()], [Y_test_scaled.min(), Y_test_scaled.max()], 'r--', label="Perfect Prediction")

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Conditional Flow Matching Evaluation")
plt.legend()
plt.grid(True)
plt.savefig("../plots/flowmatching_scat.png")
plt.show()

# 6. Calculate Pearson Correlation cleanly using CPU tensors
correlation_matrix = torch.corrcoef(torch.stack([Y_test_scaled, x_pred_mean.cpu()]))
print(f"Pearson correlation coefficient (Mean Pred vs True): {correlation_matrix[0, 1]:.4f}")

import seaborn as sns

# plt.rcdefaults()
plt.figure(figsize=(8,6))

# Turn off white grid lines
sns.set_style("whitegrid", {'grid.linestyle': ''})

plt.hist2d(Y_test_scaled.numpy(), x_pred_single.cpu().numpy(), bins=np.arange(300,1000,25),cmap='viridis')
plt.xlabel('True ttbar Mass [GeV]',size=16)
plt.ylabel('Predicted ttbar Mass [GeV]',size=16)
plt.savefig("../plots/flowmatching_contour.png")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Counts',size=16)

# Plot histograms of the true and predicted masses
plt.figure(figsize=(8,6))
plt.hist(Y_test_scaled.numpy(),bins=50,range=(0,3000), label='True',histtype='step',linewidth=2)
plt.hist(x_pred_single.cpu().numpy(),bins=50,range=(0,3000), label='Predicted',histtype='step',linewidth=2)
plt.xlabel('ttbar Mass [GeV]',size=16)
plt.savefig("../plots/flowmatching_hist.png")
