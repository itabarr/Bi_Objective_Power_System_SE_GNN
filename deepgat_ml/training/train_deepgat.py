"""
DeepGAT Training Script

Main training script for the DeepGAT model for power system state estimation.
This script handles the complete training pipeline including data loading,
model initialization, training loop, and evaluation.

Usage:
    python train_deepgat.py [--config CONFIG_PATH] [--case CASE_NAME]
"""

import sys
import os
import argparse
import torch

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DeepGAT model')
    parser.add_argument('--case', type=str, default='cigre14',
                       help='Case name for training (default: cigre14)')
    parser.add_argument('--epochs', type=int, default=600,
                       help='Number of training epochs (default: 600)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=3e-3,
                       help='Learning rate (default: 3e-3)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load existing model checkpoint')
    
    return parser.parse_args()


def setup_model(model_config, device):
    """Initialize the DeepGAT model."""
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
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model architecture:")
    print(f"  - Node features: {model_config['dim_nodes']}")
    print(f"  - Edge features: {model_config['dim_lines']}")
    print(f"  - Hidden dimension: {model_config['dim_hid']}")
    print(f"  - Output dimension: {model_config['dim_out']}")
    print(f"  - Number of heads: {model_config['heads']}")
    print(f"  - Number of layers: {model_config['gnn_layers']}")
    print(f"  - Normalization: {model_config['norm']}")
    print(f"  - Dropout rate: {model_config['dropout_rate']}")
    
    return model


def train_model(args):
    """Main training function."""
    print(f"Starting DeepGAT training on device: {DEVICE}")
    print(f"Training case: {args.case}")
    
    # Update configurations with command line arguments
    data_config = DATA_CONFIG.copy()
    data_config['case'] = args.case
    data_config['folder'] = f'data/{args.case}/'
    data_config['batch_size'] = args.batch_size
    
    training_config = TRAINING_CONFIG.copy()
    training_config['epochs'] = args.epochs
    training_config['learning_rate'] = args.lr
    training_config['batch_size'] = args.batch_size
    if args.load_model:
        training_config['load_model'] = True
        training_config['model_path'] = args.load_model
    
    # Create data loaders
    print("Loading data from pickle files...")
    train_loader, test_loader, data_info = create_data_loaders(data_config)
    
    if data_info:
        print(f"Data loaded successfully:")
        print(f"  - Training samples: {data_info['train_samples']}")
        print(f"  - Test samples: {data_info['test_samples']}")
        print(f"  - Node features: {data_info['num_node_features']}")
        print(f"  - Edge features: {data_info['num_edge_features']}")
        print(f"  - Target features: {data_info['num_target_features']}")
    
    # Initialize model
    model = setup_model(MODEL_CONFIG, DEVICE)
    
    # Create trainer
    trainer = create_trainer(model, training_config, DEVICE)
    
    # Get measurement indices
    meas_indices = get_measurement_indices(args.case)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_mae_v = float('inf')
    
    for epoch in progressBar(range(args.epochs), prefix='Training Progress:', suffix='Complete', length=50):
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader, LOSS_COEFFICIENTS)
        
        # Evaluate every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            metrics = trainer.evaluate(test_loader, meas_indices, EVALUATION_CONFIG['thresh'])
            
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Validation Metrics:")
            print(f"  - Voltage RMSE: {metrics['rmse_v']:.6f}")
            print(f"  - Voltage MAE: {metrics['mae_v']:.6f}")
            print(f"  - Angle RMSE: {metrics['rmse_th']:.6f}")
            print(f"  - Angle MAE: {metrics['mae_th']:.6f}")
            print(f"  - Voltage Prop Std: {metrics['prop_std_v']:.6f}")
            print(f"  - Angle Prop Std: {metrics['prop_std_th']:.6f}")
            
            # Save best model
            if metrics['mae_v'] < best_mae_v:
                best_mae_v = metrics['mae_v']
                best_model_path = os.path.join(args.save_dir, f'best_deepgat_{args.case}.pt')
                trainer.save_checkpoint(best_model_path, epoch + 1, metrics)
                print(f"New best model saved! MAE_V: {best_mae_v:.6f}")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f'final_deepgat_{args.case}.pt')
    final_metrics = trainer.evaluate(test_loader, meas_indices, EVALUATION_CONFIG['thresh'])
    trainer.save_checkpoint(final_model_path, args.epochs, final_metrics)
    
    print(f"\nTraining completed!")
    print(f"Best validation MAE_V: {best_mae_v:.6f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    
    return trainer, final_metrics


if __name__ == "__main__":
    args = parse_arguments()
    trainer, metrics = train_model(args)
