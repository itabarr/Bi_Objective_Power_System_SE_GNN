"""
DeepGAT Evaluation Script

Script for evaluating trained DeepGAT models on test data.

Usage:
    python evaluate_deepgat.py --model_path MODEL_PATH [--case CASE_NAME]
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
    DATA_CONFIG, MODEL_CONFIG, DEVICE,
    get_measurement_indices
)
from utils.data_loader import create_data_loaders
from utils.training_utils import create_trainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate DeepGAT model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--case', type=str, default='cigre14',
                       help='Case name for evaluation (default: cigre14)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation (default: 64)')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config from checkpoint or use default
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        # Extract model parameters from training config
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
    else:
        # Use default config if not available in checkpoint
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
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"Model was trained for {checkpoint['epoch']} epochs")
    
    return model, checkpoint


def evaluate_model(args):
    """Main evaluation function."""
    print(f"Starting DeepGAT evaluation on device: {DEVICE}")
    print(f"Evaluation case: {args.case}")
    
    # Setup data configuration
    data_config = DATA_CONFIG.copy()
    data_config['case'] = args.case
    data_config['folder'] = f'data/{args.case}/'
    data_config['batch_size'] = args.batch_size
    
    # Load data
    print("Loading test data from pickle files...")
    _, test_loader, data_info = create_data_loaders(data_config)
    
    if data_info:
        print(f"Test data loaded successfully:")
        print(f"  - Test samples: {data_info['test_samples']}")
        print(f"  - Node features: {data_info['num_node_features']}")
        print(f"  - Edge features: {data_info['num_edge_features']}")
        print(f"  - Target features: {data_info['num_target_features']}")
    
    # Load trained model
    model, checkpoint = load_model(args.model_path, DEVICE)
    
    # Create trainer for evaluation utilities
    dummy_config = {'optimizer': 'Adam', 'learning_rate': 0.001, 'load_model': False, 'model_path': ''}
    trainer = create_trainer(model, dummy_config, DEVICE)
    
    # Get measurement indices
    meas_indices = get_measurement_indices(args.case)
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    metrics = trainer.evaluate(test_loader, meas_indices, thresh=0)
    
    # Print detailed results
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Case: {args.case}")
    print(f"Test samples: {data_info['test_samples'] if data_info else 'Unknown'}")
    print(f"\nVoltage Metrics:")
    print(f"  - RMSE: {metrics['rmse_v']:.6f}")
    print(f"  - MAE:  {metrics['mae_v']:.6f}")
    print(f"  - Proportional Std: {metrics['prop_std_v']:.6f}")
    print(f"\nAngle Metrics:")
    print(f"  - RMSE: {metrics['rmse_th']:.6f}")
    print(f"  - MAE:  {metrics['mae_th']:.6f}")
    print(f"  - Proportional Std: {metrics['prop_std_th']:.6f}")
    print(f"\nLoading Metrics:")
    print(f"  - Lines RMSE: {metrics['rmse_loading']:.6f}")
    print(f"  - Lines MAE:  {metrics['mae_loading']:.6f}")
    print(f"  - Trafos RMSE: {metrics['rmse_loading_trafos']:.6f}")
    print(f"  - Trafos MAE:  {metrics['mae_loading_trafos']:.6f}")
    print(f"{'='*50}")
    
    # Show training history if available
    if 'validation_metrics' in checkpoint:
        val_metrics = checkpoint['validation_metrics']
        if val_metrics['mae_v']:
            print(f"\nTraining History:")
            print(f"  - Best validation MAE_V: {min(val_metrics['mae_v']):.6f}")
            print(f"  - Final validation MAE_V: {val_metrics['mae_v'][-1]:.6f}")
            print(f"  - Training epochs: {len(val_metrics['mae_v'])}")
    
    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    metrics = evaluate_model(args)
