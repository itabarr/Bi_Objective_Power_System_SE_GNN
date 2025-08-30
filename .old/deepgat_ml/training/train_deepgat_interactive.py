"""
DeepGAT Interactive Training Script

This script is designed for interactive execution in Jupyter notebooks or IPython.
You can run sections individually by copying and pasting them into your notebook cells.
"""

# ============================================================================
# CELL 1: IMPORTS AND SETUP
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
# CELL 2: CONFIGURATION
# ============================================================================

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

# Training parameters - MODIFY THESE AS NEEDED
case = 'cigre14'
epochs = 100  # Reduced for interactive testing
batch_size = 64
learning_rate = 3e-3
save_dir = './trained_models'
load_model_path = None

print(f"Case Study: {case}")
print(f"Training Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")

# ============================================================================
# CELL 3: MODEL SETUP
# ============================================================================

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)

# Model configuration
model_config = {
    'dim_nodes': 8,         # Node features
    'dim_lines': 6,         # Edge features  
    'dim_hid': 64,          # Hidden dimension
    'dim_out': 4,           # Output dimension
    'heads': 4,             # Attention heads
    'gnn_layers': 3,        # GNN layers
    'dropout_rate': 0.3,    # Dropout rate
    'norm': 'lipschitznorm' # Normalization
}

# Initialize model
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

total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params:,} parameters")

# ============================================================================
# CELL 4: DATA LOADING
# ============================================================================

print("\n" + "="*60)
print("DATA LOADING")
print("="*60)

# Setup data configuration
data_config = DATA_CONFIG.copy()
data_config['case'] = case
data_config['folder'] = f'data/{case}/'
data_config['batch_size'] = batch_size

# Load data
train_loader, test_loader, data_info = create_data_loaders(data_config)

print("Data loaded successfully")
if data_info:
    print(f"  Training samples: {data_info['train_samples']:,}")
    print(f"  Test samples: {data_info['test_samples']:,}")

    # Show example data structure
    sample_batch = next(iter(train_loader))
    print("\nData structure:")
    print(f"  Node features shape: {sample_batch.x.shape}")
    print(f"  Edge features shape: {sample_batch.edge_attr.shape}")
    print(f"  Targets shape: {sample_batch.y.shape}")

# ============================================================================
# CELL 5: TRAINING SETUP
# ============================================================================

print("\n" + "="*60)
print("TRAINING SETUP")
print("="*60)

# Setup training configuration
training_config = TRAINING_CONFIG.copy()
training_config['epochs'] = epochs
training_config['learning_rate'] = learning_rate
training_config['batch_size'] = batch_size

# Create trainer
trainer = create_trainer(model, training_config, DEVICE)
meas_indices = get_measurement_indices(case)

# Create save directory
Path(save_dir).mkdir(parents=True, exist_ok=True)

print("✅ Training setup complete!")

# ============================================================================
# CELL 6: TRAINING LOOP (INTERACTIVE)
# ============================================================================

print("\n" + "="*60)
print("🚀 INTERACTIVE TRAINING")
print("="*60)

# Training variables
best_mae_v = float('inf')
training_history = {'train_losses': [], 'val_metrics': []}

print("💡 You can run this training loop in smaller chunks:")
print("   - Run 10 epochs at a time")
print("   - Check results after each chunk")
print("   - Modify parameters if needed")

# Example: Train for 10 epochs at a time
chunk_size = 10
total_chunks = epochs // chunk_size

for chunk in range(total_chunks):
    start_epoch = chunk * chunk_size
    end_epoch = min((chunk + 1) * chunk_size, epochs)
    
    print(f"\n🔄 Training chunk {chunk + 1}/{total_chunks} (epochs {start_epoch + 1}-{end_epoch})")
    
    for epoch in range(start_epoch, end_epoch):
        # Train one epoch
        train_loss = trainer.train_epoch(train_loader, LOSS_COEFFICIENTS)
        training_history['train_losses'].append(train_loss)
        
        # Show progress
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {train_loss:.6f}")
    
    # Evaluate after each chunk
    metrics = trainer.evaluate(test_loader, meas_indices, EVALUATION_CONFIG['thresh'])
    training_history['val_metrics'].append(metrics)
    
    print(f"\n📊 Chunk {chunk + 1} Results:")
    print(f"  • Voltage MAE: {metrics['mae_v']:.6f}")
    print(f"  • Voltage RMSE: {metrics['rmse_v']:.6f}")
    print(f"  • Angle MAE: {metrics['mae_th']:.6f}")
    print(f"  • Angle RMSE: {metrics['rmse_th']:.6f}")
    
    # Save if best
    if metrics['mae_v'] < best_mae_v:
        best_mae_v = metrics['mae_v']
        best_model_path = os.path.join(save_dir, f'best_deepgat_{case}_interactive.pt')
        trainer.save_checkpoint(best_model_path, end_epoch, metrics)
        print(f"  🏆 New best model saved! MAE_V: {best_mae_v:.6f}")
    
    # Interactive pause - you can stop here and analyze results
    print(f"  ⏸️  Chunk completed. You can analyze results or continue...")

# ============================================================================
# CELL 7: FINAL RESULTS
# ============================================================================

print("\n" + "="*60)
print("🏁 TRAINING RESULTS")
print("="*60)

# Final evaluation
final_metrics = trainer.evaluate(test_loader, meas_indices, EVALUATION_CONFIG['thresh'])

print("🎉 Interactive training completed!")
print(f"\n📊 Final Performance:")
print(f"  🏆 Best MAE_V: {best_mae_v:.6f}")
print(f"  📈 Final MAE_V: {final_metrics['mae_v']:.6f}")
print(f"  📈 Final RMSE_V: {final_metrics['rmse_v']:.6f}")
print(f"  📈 Final MAE_θ: {final_metrics['mae_th']:.6f}")
print(f"  📈 Final RMSE_θ: {final_metrics['rmse_th']:.6f}")

# Performance status
if final_metrics['mae_v'] < 0.05:
    status = "🟢 EXCELLENT"
elif final_metrics['mae_v'] < 0.07:
    status = "🟡 GOOD"
else:
    status = "🟠 ACCEPTABLE"

print(f"\n🎯 Status: {status}")

# ============================================================================
# CELL 8: ANALYSIS AND VISUALIZATION (OPTIONAL)
# ============================================================================

print("\n" + "="*60)
print("📊 TRAINING ANALYSIS")
print("="*60)

# Plot training history if matplotlib is available
try:
    import matplotlib.pyplot as plt
    
    # Plot training loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot validation metrics
    plt.subplot(1, 2, 2)
    val_epochs = [i * chunk_size for i in range(len(training_history['val_metrics']))]
    mae_values = [m['mae_v'] for m in training_history['val_metrics']]
    plt.plot(val_epochs, mae_values, 'o-')
    plt.title('Validation MAE_V')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (p.u.)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("📈 Training plots displayed!")
    
except ImportError:
    print("📊 Matplotlib not available - install it for training plots")
    print("   pip install matplotlib")

print("\n✅ Interactive training script completed!")

# ============================================================================
# USAGE NOTES
# ============================================================================

print("""
💡 USAGE TIPS FOR INTERACTIVE TRAINING:

1. 📝 JUPYTER NOTEBOOK:
   - Copy each CELL section into separate notebook cells
   - Run cells individually to see results step by step
   - Modify parameters between cells as needed

2. 🔄 IPYTHON:
   - Run sections one at a time
   - Use %run magic command for full execution
   - Use variables interactively

3. 🛠️ CUSTOMIZATION:
   - Modify 'epochs', 'chunk_size' for different training lengths
   - Change 'learning_rate', 'batch_size' for experimentation
   - Add your own analysis cells

4. 📊 DEBUGGING:
   - Print statements show data dimensions (as requested)
   - Training progress is visible in real-time
   - Easy to stop and inspect intermediate results

5. 💾 SAVING:
   - Models are saved automatically during training
   - You can save intermediate results manually
   - Training history is stored for analysis
""")
