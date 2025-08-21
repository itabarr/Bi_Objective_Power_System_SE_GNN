"""
DeepGAT Training Script

Professional training script for DeepGAT model on power system state estimation.
Implements deep graph attention networks with Lipschitz normalization for stable training.
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.deepgat_dsse import DeepGAT_DSSE
from configs.deepgat_config import (
    DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG,
    LOSS_COEFFICIENTS, EVALUATION_CONFIG, DEVICE,
    get_measurement_indices
)
from utils.data_loader import create_data_loaders
from utils.training_utils import create_trainer
from dsml_loadsampling import progressBar

print("INFO: All imports successful")
print(f"INFO: Using device: {DEVICE}")

# ============================================================================
# SECTION 2: CONFIGURATION AND HYPERPARAMETERS
# ============================================================================

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

# Training parameters
case = 'cigre14'
epochs = 600
batch_size = 64
learning_rate = 3e-3
save_dir = './trained_models'
load_model_path = None  # Set to path if you want to resume training

print(f"Case Study: {case}")
print(f"Training Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Save Directory: {save_dir}")
print(f"Resume Training: {'Yes' if load_model_path else 'No'}")

# ============================================================================
# SECTION 3: MODEL ARCHITECTURE CONFIGURATION
# ============================================================================

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)

# Model architecture parameters
model_config = {
    'dim_nodes': 8,         # V, Theta, P, Q and their covariances
    'dim_lines': 6,         # P, Q and their Cov, B, G
    'dim_hid': 64,          # Hidden dimension
    'dim_out': 4,           # V, Theta outputs
    'heads': 4,             # Number of attention heads
    'gnn_layers': 3,        # Number of GNN layers
    'dropout_rate': 0.3,    # Dropout rate for regularization
    'norm': 'lipschitznorm' # Stable Lipschitz normalization
}

print("Model Configuration:")
for key, value in model_config.items():
    print(f"  {key}: {value}")

# ============================================================================
# SECTION 4: MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*60)
print("MODEL INITIALIZATION")
print("="*60)

# Initialize the DeepGAT model
model = DeepGAT_DSSE(
    dim_feat=model_config['dim_nodes'],
    dim_dense=model_config['dim_hid'],
    dim_out=model_config['dim_out'],
    heads=model_config['heads'],
    num_layers=model_config['gnn_layers'],
    edge_dim=model_config['dim_lines'],
    norm=model_config['norm'],
    dropout=model_config['dropout_rate']
)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model initialized successfully")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

# ============================================================================
# SECTION 5: DATA LOADING
# ============================================================================

print("\n" + "="*60)
print("DATA LOADING")
print("="*60)

# Setup data configuration
data_config = DATA_CONFIG.copy()
data_config['case'] = case
data_config['folder'] = f'data/{case}/'
data_config['batch_size'] = batch_size

print(f"Loading data from: {data_config['folder']}")
print(f"Batch size: {batch_size}")

# Load data from pickle files
try:
    train_loader, test_loader, data_info = create_data_loaders(data_config)

    print("Data loaded successfully")
    if data_info:
        print(f"  Training samples: {data_info['train_samples']:,}")
        print(f"  Test samples: {data_info['test_samples']:,}")
        print(f"  Node features: {data_info['num_node_features']}")
        print(f"  Edge features: {data_info['num_edge_features']}")
        print(f"  Target features: {data_info['num_target_features']}")

        # Print example data structure for debugging
        print("\nExample data structure:")
        sample_batch = next(iter(train_loader))
        print(f"  Batch node features shape: {sample_batch.x.shape}")
        print(f"  Batch edge features shape: {sample_batch.edge_attr.shape}")
        print(f"  Batch edge index shape: {sample_batch.edge_index.shape}")
        print(f"  Batch targets shape: {sample_batch.y.shape}")
        print(f"  Example node features (first 3): {sample_batch.x[:3]}")
        print(f"  Example edge features (first 3): {sample_batch.edge_attr[:3]}")

except Exception as e:
    print(f"ERROR: Failed to load data: {e}")
    print("NOTE: Make sure the pickle files exist in the data directory")
    exit(1)

# ============================================================================
# SECTION 6: TRAINING SETUP
# ============================================================================

print("\n" + "="*60)
print("TRAINING SETUP")
print("="*60)

# Setup training configuration
training_config = TRAINING_CONFIG.copy()
training_config['epochs'] = epochs
training_config['learning_rate'] = learning_rate
training_config['batch_size'] = batch_size
if load_model_path:
    training_config['load_model'] = True
    training_config['model_path'] = load_model_path

print("Training configuration:")
print(f"  Epochs: {epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Optimizer: {training_config['optimizer']}")
print(f"  Resume from checkpoint: {'Yes' if load_model_path else 'No'}")

# Create trainer
trainer = create_trainer(model, training_config, DEVICE)
print("Trainer initialized successfully")

# Get measurement indices for evaluation
meas_indices = get_measurement_indices(case)
print(f"Measurement indices loaded for case: {case}")

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
print("Evaluation every 10 epochs")
print(f"Models will be saved to: {save_dir}")

# Training variables
best_mae_v = float('inf')
training_history = {
    'train_losses': [],
    'val_metrics': []
}

# Main training loop
for epoch in progressBar(range(epochs), prefix='Training Progress:', suffix='Complete', length=50):

    # Train for one epoch
    train_loss = trainer.train_epoch(train_loader, LOSS_COEFFICIENTS)
    training_history['train_losses'].append(train_loss)

    # Evaluate every 10 epochs or on the last epoch
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        metrics = trainer.evaluate(test_loader, meas_indices, EVALUATION_CONFIG['thresh'])
        training_history['val_metrics'].append(metrics)

        print(f"\nEpoch {epoch + 1}/{epochs} Results:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Validation Metrics:")
        print(f"     Voltage RMSE: {metrics['rmse_v']:.6f}")
        print(f"     Voltage MAE: {metrics['mae_v']:.6f}")
        print(f"     Angle RMSE: {metrics['rmse_th']:.6f}")
        print(f"     Angle MAE: {metrics['mae_th']:.6f}")
        print(f"     Voltage Prop Std: {metrics['prop_std_v']:.6f}")
        print(f"     Angle Prop Std: {metrics['prop_std_th']:.6f}")

        # Save best model
        if metrics['mae_v'] < best_mae_v:
            best_mae_v = metrics['mae_v']
            best_model_path = os.path.join(save_dir, f'best_deepgat_{case}.pt')
            trainer.save_checkpoint(best_model_path, epoch + 1, metrics)
            print(f"  New best model saved! MAE_V: {best_mae_v:.6f}")

# ============================================================================
# SECTION 8: FINAL EVALUATION AND SAVING
# ============================================================================

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)

# Final evaluation
final_metrics = trainer.evaluate(test_loader, meas_indices, EVALUATION_CONFIG['thresh'])

# Save final model
final_model_path = os.path.join(save_dir, f'final_deepgat_{case}.pt')
trainer.save_checkpoint(final_model_path, epochs, final_metrics)

# Print final results
print("Training completed successfully")
print("\nFinal Results:")
print(f"  Best Validation MAE_V: {best_mae_v:.6f}")
print(f"  Final Validation MAE_V: {final_metrics['mae_v']:.6f}")
print(f"  Final Validation RMSE_V: {final_metrics['rmse_v']:.6f}")
print(f"  Final Validation MAE_θ: {final_metrics['mae_th']:.6f}")
print(f"  Final Validation RMSE_θ: {final_metrics['rmse_th']:.6f}")

print("\nSaved Models:")
print(f"  Best model: {best_model_path}")
print(f"  Final model: {final_model_path}")

# Performance assessment
if final_metrics['mae_v'] < 0.05:
    status = "EXCELLENT - Production Ready"
elif final_metrics['mae_v'] < 0.07:
    status = "GOOD - Ready for deployment"
else:
    status = "ACCEPTABLE - Consider further tuning"

print(f"\nPerformance Status: {status}")

print("\n" + "="*60)
print("TRAINING SCRIPT COMPLETED")
print("="*60)
