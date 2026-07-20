# Transformer for Jet Classification
# Local Version

import os
import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import torch
import time

# ------------------------------ Check GPU ------------------------------ #
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# ------------------------------ File Loading (Local) ------------------------------ #
filename = "transformer_jet_classification.h5"

if not os.path.exists(filename):
    raise FileNotFoundError(f"File not found: {filename}. Make sure it's in the same directory.")

# ------------------------------ Load Data ------------------------------ #
with h5py.File(filename, "r") as f:
    signal_data = f["signal"][:]
    bkg_data = f["bkg"][:]

print(f"Signal shape: {signal_data.shape}")
print(f"Background shape: {bkg_data.shape}")

signal = structured_to_unstructured(signal_data)
background = structured_to_unstructured(bkg_data)

X = np.concatenate([signal, background])
target = np.concatenate([np.ones(signal.shape[0]), np.zeros(background.shape[0])], axis=0)

print(f"Total samples: {X.shape[0]}")

# ------------------------------ Train/Val/Test Split ------------------------------ #
from sklearn.model_selection import train_test_split

X_train, X_temp, Y_train, Y_temp = train_test_split(X, target, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ------------------------------ Scaling ------------------------------ #
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, 5))

X_train_scaled = scaler.transform(X_train.reshape(-1, 5)).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, 5)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)

# ------------------------------ Convert to Tensors ------------------------------ #
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

Y_train_tensor = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32, device=device)
Y_val_tensor = torch.tensor(Y_val.reshape(-1, 1), dtype=torch.float32, device=device)
Y_test_tensor = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32, device=device)

# ------------------------------ Model Definition ------------------------------ #
import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.1

class LatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) - inspired by DeepSeek
    Compresses Keys and Values into a lower-dimensional latent space
    """
    def __init__(self, d_model, nhead, latent_dim=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # If no latent_dim specified, use half of d_model
        self.latent_dim = latent_dim if latent_dim else d_model // 2
        
        # Query projection (standard)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        
        # Key and Value projections with latent compression
        # Input → latent space → K and V
        self.W_kv = nn.Linear(d_model, 2 * self.latent_dim, bias=False)
        self.W_k = nn.Linear(self.latent_dim, d_model, bias=False)
        self.W_v = nn.Linear(self.latent_dim, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 1. Query (standard)
        q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 2. Key/Value via latent compression
        kv = self.W_kv(x)  # (batch, seq, 2 * latent_dim)
        k_latent, v_latent = kv.chunk(2, dim=-1)  # Split into K and V latent
        
        # 3. Project latent back to full dimension
        k = self.W_k(k_latent).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.W_v(v_latent).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # 4. Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output)


class LatentTransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Latent Attention
    """
    def __init__(self, d_model, nhead, latent_dim=None, dropout=0.1):
        super().__init__()
        self.attention = LatentAttention(d_model, nhead, latent_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class JetTransformerWithLatentAttention(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, latent_dim=64, dropout_rate=0.1):
        super().__init__()
        
        self.input_embedder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU()
        )
        
        # Stack Latent Transformer layers
        self.layers = nn.ModuleList([
            LatentTransformerEncoderLayer(d_model, nhead, latent_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.output_classifier_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    
    def mean_pooling(self, x):
        return x.mean(dim=1)
    
    def forward(self, x):
        x = self.input_embedder(x)
        
        # Pass through each latent attention layer
        for layer in self.layers:
            x = layer(x)
        
        x = self.mean_pooling(x)
        x = self.output_classifier_head(x)
        return x


# Create the model
model = JetTransformerWithLatentAttention(
    input_dim=5,
    d_model=128,
    nhead=8,
    num_layers=6,
    latent_dim=96,  # Compressed latent dimension
    dropout_rate=0.01
)
model = model.to(device)

# ------------------------------ DataLoaders ------------------------------ #
from torch.utils.data import TensorDataset, DataLoader

batch_size = 2048

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------ Optimizer & Scheduler ------------------------------ #
from torch.optim.lr_scheduler import LambdaLR

learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

warmup_steps = 86
def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(warmup_steps) / float(current_step))

scheduler = LambdaLR(optimizer, lr_lambda)

# ------------------------------ Training Loop ------------------------------ #
print("\nBeginning training loop")
print("=" * 60)

train_losses = []
val_losses = []
times = []

num_epochs = 30

for epoch in range(num_epochs):
    start_time = time.time()
    
    # Training
    model.train()
    running_train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    epoch_time = time.time() - start_time
    times.append(epoch_time)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")
    
    torch.cuda.empty_cache()

# ------------------------------ Evaluation ------------------------------ #
model.eval()
list_of_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        list_of_predictions.append(outputs)

pred = torch.concatenate(list_of_predictions)
Y_pred = torch.round(torch.sigmoid(pred)).detach().cpu().numpy()

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
norm_conf_matrix = confusion_matrix(Y_test, Y_pred, normalize='true')
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')
roc_auc = roc_auc_score(Y_test.reshape(-1, 1), Y_pred)

print(f"\n========== Results ==========")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"=============================\n")

# ------------------------------ Plotting ------------------------------ #
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Loss plot
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Text box with metrics
axes[0].text(
    0.98, 0.98,
    f"Epochs: {num_epochs}\nBatch size: {batch_size}\nLR: {learning_rate}\nDropout: {dropout_rate}\n\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\nROC-AUC: {roc_auc:.4f}",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8),
    ha="right",
    va="top",
    transform=axes[0].transAxes
)

# Confusion matrix

ConfusionMatrixDisplay(confusion_matrix=norm_conf_matrix, display_labels=[0, 1]).plot(ax=axes[1], cmap='viridis')
axes[1].set_title('Normalized Confusion Matrix')

# ROC curve
probabilities = torch.sigmoid(pred).detach().cpu().numpy()
RocCurveDisplay.from_predictions(Y_test, probabilities, ax=axes[2])
axes[2].set_title('ROC Curve')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("loss_curve.png")
print("Plot saved as loss_curve.png")
plt.show()