# ============================================================================
#  IMPORTS AND SETUP
# ============================================================================

import pandas as pd
import pandapower as pp
import numpy as np
import torch
import sys
import os
from pathlib import Path

print("INFO: All imports successful")

# ============================================================================
# SECTION 2: CONFIGURATION AND HYPERPARAMETERS
# ============================================================================

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

# Training parameters (from original script)
case = 'cigre14'
folder = f'data/{case}/'
epochs = 600
batch_size = 64
learning_rate = 3e-3
save_dir = './trained_models'

# Model parameters (from original script)
phase_shift = True
num_nfeat = 8   # Node measurement features
num_efeat = 6   # Edge measurement features
num_nmeas = 4   # Number of node measurements
num_emeas = 2   # Number of edge measurements

print(f"Case Study: {case}")
print(f"Data Folder: {folder}")
print(f"Training Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Node Features: {num_nfeat}")
print(f"Edge Features: {num_efeat}")

# ============================================================================
# SECTION 3: MEASUREMENT INDICES SETUP
# ============================================================================

print("\n" + "="*60)
print("MEASUREMENT INDICES")
print("="*60)

# Set the measurement indices for each grid (from original script)
if 'cigre' in case:
    meas_v = np.array([0, 1, 12, 7, 11, 13])  # Fixed: removed index 14
    meas_pflow = np.array([0, 10])
else:
    meas_v = np.array([35, 16, 52, 47, 6, 48, 59, 27, 37, 56])
    meas_pflow = np.array([40, 43, 11, 21, 54, 57])

print(f"Voltage measurement indices: {meas_v}")
print(f"Power flow measurement indices: {meas_pflow}")

# ============================================================================
# SECTION 4: DATA LOADING
# ============================================================================

print("\n" + "="*60)
print("DATA LOADING")
print("="*60)

print(f"Loading data from pickle files in {folder}")

# Load data using original function
dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
    folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pflow
)

# Shuffle and split data (from original script)
random.shuffle(dataset)
split_coef = 0.9
train_dataset = dataset[0:int(split_coef*len(dataset))]
test_dataset = dataset[int(split_coef*len(dataset)):]

print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print example data structure for debugging
sample_batch = next(iter(train_loader))
print(f"\nExample data structure:")
print(f"  Node features shape: {sample_batch.x.shape}")
print(f"  Edge features shape: {sample_batch.edge_attr.shape}")
print(f"  Edge index shape: {sample_batch.edge_index.shape}")
print(f"  Targets shape: {sample_batch.y.shape}")
print(f"  Example node features (first 3): {sample_batch.x[:3]}")
print(f"  Example edge features (first 3): {sample_batch.edge_attr[:3]}")

# ============================================================================
# SECTION 5: MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*60)
print("MODEL INITIALIZATION")
print("="*60)


# Initialize DeepGAT model (using original parameters)
model = DeepGAT_DSSE(
    dim_feat=num_nfeat,     # 8 node measurement features
    dim_dense=64,           # Hidden dimension
    dim_out=2,              # Output: voltage magnitude and angle
    heads=4,                # Number of attention heads
    num_layers=3,           # Number of GNN layers
    edge_dim=num_efeat,     # 6 edge measurement features
    norm='lipschitznorm',   # Normalization type
    dropout=0.3,            # Dropout rate
    concat=False            # Average attention heads
)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized successfully")
print(f"Total parameters: {total_params:,}")
print(f"Model architecture:")
print(f"  Input features: {num_nfeat} nodes, {num_efeat} edges")
print(f"  Hidden dimension: 64")
print(f"  Output features: 2 (voltage magnitude and angle)")
print(f"  Attention heads: 4")
print(f"  GNN layers: 3")

# ============================================================================
# SECTION 6: TRAINING SETUP
# ============================================================================

print("\n" + "="*60)
print("TRAINING SETUP")
print("="*60)

# Setup optimizer (from original script)
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

# Regularization coefficients (from original script)
mu_v = 1e-1
reg_coefs = {
    'mu_v': mu_v,
    'mu_theta': mu_v,
    'lam_v': 1e-4,
    'lam_p': 1e-8,
    'lam_pf': 1e-6,
    'lam_reg': 1e2
}

print(f"Optimizer: Adamax")
print(f"Learning rate: {learning_rate}")
print(f"Regularization coefficients:")
for key, value in reg_coefs.items():
    print(f"  {key}: {value}")

# Create save directory
Path(save_dir).mkdir(parents=True, exist_ok=True)
print(f"Save directory created: {save_dir}")

# ============================================================================
# SECTION 7: TRAINING LOOP
# ============================================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

print(f"Training {case.upper()} case for {epochs} epochs")

# Training tracking
train_list = []
test_list = []
best_mae_v = float('inf')

# Main training loop (from original script)
for epoch in range(epochs):
    train_loss = 0
    model.train()

    for data in train_loader:
        num_samples = data.batch[-1] + 1

        optimizer.zero_grad()

        # Forward pass - use only measurement features
        out = model(data.x[:, :num_nfeat], data.edge_index, data.edge_attr[:, :num_efeat])

        # Calculate loss using original GSP-WLS function
        loss = gsp_wls_edge(
            input=data.x[:, :num_nfeat],
            edge_input=data.edge_attr[:, :num_efeat],
            output=out,
            x_mean=x_mean,
            x_std=x_std,
            edge_mean=pflow_mean,
            edge_std=pflow_std,
            edge_index=data.edge_index,
            reg_coefs=reg_coefs,
            num_samples=num_samples,
            node_param=data.x[:, num_nfeat:],
            edge_param=data.edge_attr[:, num_efeat:]
        )

        loss.backward()
        optimizer.step()
        train_loss += loss

    # Record training loss
    avg_train_loss = float((train_loss/len(train_loader)).detach().float().numpy())
    train_list.append(avg_train_loss)

    # Evaluate every 10 epochs or on the last epoch
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        
        evaluate_model(
            model, test_loader, num_nfeat=num_nfeat, num_efeat=num_efeat,
            x_mean=x_mean, x_std=x_std, epoch=epoch, epochs=epochs,
            avg_train_loss=avg_train_loss
        )


print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)


# ============================================================================
# SECTION 8: FINAL EVALUATION AND SAVING
# ============================================================================



