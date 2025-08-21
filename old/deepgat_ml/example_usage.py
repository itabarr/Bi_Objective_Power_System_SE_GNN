"""
Example Usage of DeepGAT ML Package

This script demonstrates how to use the DeepGAT ML package for training
and evaluating models on power system state estimation tasks.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import DeepGAT_DSSE
from configs.deepgat_config import (
    DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, 
    LOSS_COEFFICIENTS, DEVICE, get_measurement_indices
)
from utils.data_loader import create_data_loaders
from utils.training_utils import create_trainer


def example_training():
    """Example of how to train a DeepGAT model."""
    print("=== DeepGAT Training Example ===")
    
    # 1. Setup configuration
    print("1. Setting up configuration...")
    data_config = DATA_CONFIG.copy()
    data_config['case'] = 'cigre14'
    data_config['folder'] = 'data/cigre14/'
    
    # 2. Load data from pickle files
    print("2. Loading data from pickle files...")
    try:
        train_loader, test_loader, data_info = create_data_loaders(data_config)
        print(f"   ✓ Loaded {data_info['train_samples']} training samples")
        print(f"   ✓ Loaded {data_info['test_samples']} test samples")
        print(f"   ✓ Node features: {data_info['num_node_features']}")
        print(f"   ✓ Edge features: {data_info['num_edge_features']}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        print("   Note: Make sure pickle files exist in data/cigre14/")
        return
    
    # 3. Initialize model
    print("3. Initializing DeepGAT model...")
    model = DeepGAT_DSSE(
        dim_feat=MODEL_CONFIG['dim_nodes'],
        dim_dense=MODEL_CONFIG['dim_hid'],
        dim_out=MODEL_CONFIG['dim_out'],
        heads=MODEL_CONFIG['heads'],
        num_layers=MODEL_CONFIG['gnn_layers'],
        edge_dim=MODEL_CONFIG['dim_lines'],
        norm=MODEL_CONFIG['norm'],
        dropout=MODEL_CONFIG['dropout_rate']
    )
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Create trainer
    print("4. Setting up trainer...")
    trainer = create_trainer(model, TRAINING_CONFIG, DEVICE)
    print(f"   ✓ Trainer initialized on device: {DEVICE}")
    
    # 5. Short training example (just 5 epochs)
    print("5. Running short training example (5 epochs)...")
    meas_indices = get_measurement_indices(data_config['case'])
    
    for epoch in range(5):
        # Train one epoch
        train_loss = trainer.train_epoch(train_loader, LOSS_COEFFICIENTS)
        
        # Evaluate
        if epoch % 2 == 0:  # Evaluate every 2 epochs
            metrics = trainer.evaluate(test_loader, meas_indices)
            print(f"   Epoch {epoch+1}: Loss={train_loss:.6f}, MAE_V={metrics['mae_v']:.6f}")
    
    print("   ✓ Training example completed!")
    
    # 6. Save model example
    print("6. Saving model checkpoint...")
    os.makedirs('example_checkpoints', exist_ok=True)
    trainer.save_checkpoint('example_checkpoints/example_model.pt', 5)
    print("   ✓ Model saved to example_checkpoints/example_model.pt")
    
    return trainer


def example_evaluation():
    """Example of how to evaluate a trained model."""
    print("\n=== DeepGAT Evaluation Example ===")
    
    # Check if example model exists
    model_path = 'example_checkpoints/example_model.pt'
    if not os.path.exists(model_path):
        print(f"   ✗ Example model not found at {model_path}")
        print("   Run the training example first!")
        return
    
    print("1. Loading trained model...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Initialize model with same config
    model = DeepGAT_DSSE(
        dim_feat=MODEL_CONFIG['dim_nodes'],
        dim_dense=MODEL_CONFIG['dim_hid'],
        dim_out=MODEL_CONFIG['dim_out'],
        heads=MODEL_CONFIG['heads'],
        num_layers=MODEL_CONFIG['gnn_layers'],
        edge_dim=MODEL_CONFIG['dim_lines'],
        norm=MODEL_CONFIG['norm'],
        dropout=MODEL_CONFIG['dropout_rate']
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Load test data
    print("2. Loading test data...")
    data_config = DATA_CONFIG.copy()
    data_config['case'] = 'cigre14'
    data_config['folder'] = 'data/cigre14/'
    
    try:
        _, test_loader, data_info = create_data_loaders(data_config)
        print(f"   ✓ Loaded {data_info['test_samples']} test samples")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Evaluate model
    print("3. Evaluating model...")
    dummy_config = {'optimizer': 'Adam', 'learning_rate': 0.001, 'load_model': False, 'model_path': ''}
    trainer = create_trainer(model, dummy_config, DEVICE)
    
    meas_indices = get_measurement_indices('cigre14')
    metrics = trainer.evaluate(test_loader, meas_indices)
    
    print("   ✓ Evaluation completed!")
    print(f"   Results:")
    print(f"     - Voltage RMSE: {metrics['rmse_v']:.6f}")
    print(f"     - Voltage MAE:  {metrics['mae_v']:.6f}")
    print(f"     - Angle RMSE:   {metrics['rmse_th']:.6f}")
    print(f"     - Angle MAE:    {metrics['mae_th']:.6f}")


def main():
    """Run the complete example."""
    print("DeepGAT ML Package - Usage Example")
    print("=" * 50)
    
    # Run training example
    trainer = example_training()
    
    if trainer is not None:
        # Run evaluation example
        example_evaluation()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nTo run full training, use:")
    print("python deepgat_ml/training/train_deepgat.py --case cigre14 --epochs 600")
    print("\nTo evaluate a model, use:")
    print("python deepgat_ml/training/evaluate_deepgat.py --model_path path/to/model.pt")


if __name__ == "__main__":
    main()
