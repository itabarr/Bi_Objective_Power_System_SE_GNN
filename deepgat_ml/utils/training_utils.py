"""
Training Utilities for DeepGAT

This module provides utility functions for training the DeepGAT model.
"""

import sys
import os
import torch
import torch.optim as optim
import numpy as np
from torchmetrics.regression import MeanAbsoluteError

# Add parent directory to path to import from main project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dsml_data import gsp_wls_edge
from dsml_loadsampling import progressBar


class DeepGATTrainer:
    """Trainer class for DeepGAT model."""
    
    def __init__(self, model, config, device):
        """
        Initialize the trainer.
        
        Args:
            model: DeepGAT model instance
            config: Training configuration dictionary
            device: PyTorch device (cuda/cpu)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer
        optimizer_class = getattr(optim, config['optimizer'])
        self.optimizer = optimizer_class(
            self.model.parameters(), 
            lr=config['learning_rate']
        )
        
        # Load model if specified
        if config['load_model'] and config['model_path']:
            self.load_checkpoint(config['model_path'])
        
        # Initialize metrics tracking
        self.train_losses = []
        self.validation_metrics = {
            'rmse_v': [], 'mae_v': [],
            'rmse_th': [], 'mae_th': [],
            'rmse_loading': [], 'mae_loading': [],
            'rmse_loading_trafos': [], 'mae_loading_trafos': [],
            'prop_std_v': [], 'prop_std_th': []
        }
    
    def train_epoch(self, train_loader, loss_coefficients):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(self.device)
            num_samples = data.batch[-1] + 1
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data.x, data.edge_index, data.edge_attr)
            
            # Calculate loss using GSP-WLS
            loss = gsp_wls_edge(
                out, data.y, data.edge_index, data.edge_attr,
                data.batch, num_samples, loss_coefficients
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, test_loader, meas_indices, thresh=0):
        """Evaluate the model on test data."""
        self.model.eval()
        
        # Initialize metrics
        mae_v = MeanAbsoluteError()
        mae_th = MeanAbsoluteError()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr)
                
                all_predictions.append(out.cpu())
                all_targets.append(data.y.cpu())
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self._calculate_detailed_metrics(
            predictions, targets, meas_indices, thresh
        )
        
        # Update validation metrics
        for key, value in metrics.items():
            if key in self.validation_metrics:
                self.validation_metrics[key].append(value)
        
        return metrics
    
    def _calculate_detailed_metrics(self, predictions, targets, meas_indices, thresh):
        """Calculate detailed evaluation metrics."""
        # Extract voltage magnitudes and angles
        pred_v = predictions[:, :len(meas_indices['meas_v'])]
        target_v = targets[:, :len(meas_indices['meas_v'])]
        
        pred_th = predictions[:, len(meas_indices['meas_v']):len(meas_indices['meas_v'])*2]
        target_th = targets[:, len(meas_indices['meas_v']):len(meas_indices['meas_v'])*2]
        
        # Calculate RMSE and MAE for voltages
        rmse_v = torch.sqrt(torch.mean((pred_v - target_v) ** 2)).item()
        mae_v = torch.mean(torch.abs(pred_v - target_v)).item()
        
        # Calculate RMSE and MAE for angles
        rmse_th = torch.sqrt(torch.mean((pred_th - target_th) ** 2)).item()
        mae_th = torch.mean(torch.abs(pred_th - target_th)).item()
        
        # Calculate proportional standard deviations
        prop_std_v = torch.std(pred_v - target_v).item() / torch.mean(target_v).item()
        prop_std_th = torch.std(pred_th - target_th).item() / torch.mean(torch.abs(target_th)).item()
        
        # Calculate loading metrics if we have power flow data
        # For now, using placeholders - would need additional power flow calculations
        rmse_loading = 0.0
        mae_loading = 0.0
        rmse_loading_trafos = 0.0
        mae_loading_trafos = 0.0
        
        return {
            'rmse_v': rmse_v,
            'mae_v': mae_v,
            'rmse_th': rmse_th,
            'mae_th': mae_th,
            'rmse_loading': rmse_loading,
            'mae_loading': mae_loading,
            'rmse_loading_trafos': rmse_loading_trafos,
            'mae_loading_trafos': mae_loading_trafos,
            'prop_std_v': prop_std_v,
            'prop_std_th': prop_std_th
        }
    
    def save_checkpoint(self, filepath, epoch, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'validation_metrics': self.validation_metrics,
            'config': self.config
        }
        if metrics:
            checkpoint['latest_metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'validation_metrics' in checkpoint:
            self.validation_metrics = checkpoint['validation_metrics']
        
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint.get('epoch', 0)


def create_trainer(model, training_config, device):
    """
    Create a trainer instance.
    
    Args:
        model: DeepGAT model instance
        training_config: Training configuration dictionary
        device: PyTorch device
        
    Returns:
        DeepGATTrainer: Configured trainer instance
    """
    return DeepGATTrainer(model, training_config, device)
