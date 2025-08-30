"""
Training Utilities for Power System State Estimation Models

This module provides training utilities, optimizers, and evaluation functions
for power system state estimation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from torchmetrics.regression import MeanAbsoluteError
import time

from data._dsml_data import angular_distance, get_pflow
from models.power_system_models import (
    PowerSystemGNNBase, MultiObjectivePowerSystemModel, 
    AngularAwareLoss, PowerFlowConstraintLoss
)


class PowerSystemTrainer:
    """
    Trainer class for power system state estimation models.
    Supports both single-objective and multi-objective training.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5,
                 scheduler_type: str = "exponential", scheduler_params: Dict = None):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizers
        if isinstance(model, MultiObjectivePowerSystemModel):
            self.optimizer_accuracy = optim.Adam(
                model.get_accuracy_parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            self.optimizer_constraint = optim.Adam(
                model.get_constraint_parameters(), 
                lr=learning_rate * 0.1,  # Slower learning for constraints
                weight_decay=weight_decay
            )
            self.multi_objective = True
        else:
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            self.multi_objective = False
        
        # Initialize schedulers
        scheduler_params = scheduler_params or {}
        if scheduler_type == "exponential":
            gamma = scheduler_params.get("gamma", 0.99)
            if self.multi_objective:
                self.scheduler_accuracy = ExponentialLR(self.optimizer_accuracy, gamma=gamma)
                self.scheduler_constraint = ExponentialLR(self.optimizer_constraint, gamma=gamma)
            else:
                self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type == "plateau":
            patience = scheduler_params.get("patience", 10)
            factor = scheduler_params.get("factor", 0.5)
            if self.multi_objective:
                self.scheduler_accuracy = ReduceLROnPlateau(
                    self.optimizer_accuracy, patience=patience, factor=factor
                )
                self.scheduler_constraint = ReduceLROnPlateau(
                    self.optimizer_constraint, patience=patience, factor=factor
                )
            else:
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, patience=patience, factor=factor
                )
        
        # Initialize loss functions
        self.angular_loss = AngularAwareLoss(device=device)
        self.constraint_loss = PowerFlowConstraintLoss(device=device)
        self.mse_loss = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'mae_v': [],
            'mae_th': [],
            'rmse_v': [],
            'rmse_th': []
        }
    
    def train_epoch(self, train_loader, loss_coefficients: Dict[str, float], 
                   num_nfeat: int, num_efeat: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            
            if self.multi_objective:
                loss = self._train_multi_objective_step(
                    data, loss_coefficients, num_nfeat, num_efeat
                )
            else:
                loss = self._train_single_objective_step(
                    data, loss_coefficients, num_nfeat, num_efeat
                )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def _train_single_objective_step(self, data, loss_coefficients: Dict[str, float], 
                                   num_nfeat: int, num_efeat: int) -> torch.Tensor:
        """Single training step for single-objective models."""
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(
            data.x[:, :num_nfeat],
            data.edge_index,
            data.edge_attr[:, :num_efeat]
        )
        
        # Compute loss using existing GSP-WLS loss
        from data._dsml_data import gsp_wls_edge
        loss = gsp_wls_edge(
            input=data.x[:, :num_nfeat],
            edge_input=data.edge_attr[:, :num_efeat],
            output=out,
            x_mean=torch.zeros(num_nfeat),  # These should be passed as parameters
            x_std=torch.ones(num_nfeat),
            edge_mean=torch.zeros(num_efeat),
            edge_std=torch.ones(num_efeat),
            edge_index=data.edge_index,
            reg_coefs=loss_coefficients,
            num_samples=data.batch[-1] + 1 if hasattr(data, 'batch') else 1,
            node_param=data.x[:, num_nfeat:],
            edge_param=data.edge_attr[:, num_efeat:]
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def _train_multi_objective_step(self, data, loss_coefficients: Dict[str, float], 
                                  num_nfeat: int, num_efeat: int) -> torch.Tensor:
        """Single training step for multi-objective models."""
        # Train constraint network
        self.optimizer_constraint.zero_grad()
        
        state_est, constraint_violation = self.model(
            data.x[:, :num_nfeat],
            data.edge_index,
            data.edge_attr[:, :num_efeat]
        )
        
        # Constraint loss (minimize violations)
        constraint_loss = torch.mean(constraint_violation ** 2)
        constraint_loss.backward()
        self.optimizer_constraint.step()
        
        # Train accuracy network
        self.optimizer_accuracy.zero_grad()
        
        state_est, constraint_violation = self.model(
            data.x[:, :num_nfeat],
            data.edge_index,
            data.edge_attr[:, :num_efeat]
        )
        
        # Accuracy loss with angular awareness
        accuracy_loss = self.angular_loss(state_est, data.y)
        
        # Combined loss
        total_loss = accuracy_loss - loss_coefficients.get('constraint_weight', 0.1) * torch.mean(constraint_violation)
        total_loss.backward()
        self.optimizer_accuracy.step()
        
        return total_loss
    
    def evaluate(self, test_loader, num_nfeat: int, num_efeat: int, 
                normalization_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate the model on test data."""
        self.model.eval()
        
        total_rmse_v = 0.0
        total_mae_v = 0.0
        total_rmse_th = 0.0
        total_mae_th = 0.0
        num_batches = len(test_loader)
        
        mae_fn = MeanAbsoluteError().to(self.device)
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                
                # Forward pass
                if self.multi_objective:
                    out, _ = self.model(
                        data.x[:, :num_nfeat],
                        data.edge_index,
                        data.edge_attr[:, :num_efeat]
                    )
                else:
                    out = self.model(
                        data.x[:, :num_nfeat],
                        data.edge_index,
                        data.edge_attr[:, :num_efeat]
                    )
                
                # Denormalize output
                x_mean = normalization_params.get('x_mean', torch.zeros(2))
                x_std = normalization_params.get('x_std', torch.ones(2))
                
                out = torch.cat([
                    out[:, 0:1] * x_std[:1] + x_mean[:1], 
                    out[:, 1:]
                ], dim=1)
                
                # Voltage magnitude metrics
                total_rmse_v += torch.sqrt(nn.functional.mse_loss(out[:, :1], data.y[:, :1])).item()
                total_mae_v += mae_fn(out[:, :1], data.y[:, :1]).item()
                
                # Voltage angle metrics (using angular distance)
                angle_diff = angular_distance(out[:, 1:], data.y[:, 1:])
                total_rmse_th += torch.sqrt(torch.mean(angle_diff ** 2)).item()
                total_mae_th += torch.mean(torch.abs(angle_diff)).item()
        
        metrics = {
            'rmse_v': total_rmse_v / num_batches,
            'mae_v': total_mae_v / num_batches,
            'rmse_th': total_rmse_th / num_batches,
            'mae_th': total_mae_th / num_batches
        }
        
        return metrics
    
    def train(self, train_loader, test_loader, epochs: int, 
              loss_coefficients: Dict[str, float], num_nfeat: int, num_efeat: int,
              normalization_params: Dict[str, torch.Tensor],
              validation_freq: int = 1, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            loss_coefficients: Loss function coefficients
            num_nfeat: Number of node features
            num_efeat: Number of edge features
            normalization_params: Normalization parameters
            validation_freq: Frequency of validation evaluation
            verbose: Whether to print progress
        
        Returns:
            Dictionary containing training history
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(
                train_loader, loss_coefficients, num_nfeat, num_efeat
            )
            
            # Validation
            if epoch % validation_freq == 0:
                val_metrics = self.evaluate(
                    test_loader, num_nfeat, num_efeat, normalization_params
                )
                
                # Update history
                self.training_history['train_loss'].append(train_metrics['train_loss'])
                self.training_history['val_loss'].append(val_metrics.get('val_loss', 0.0))
                self.training_history['mae_v'].append(val_metrics['mae_v'])
                self.training_history['mae_th'].append(val_metrics['mae_th'])
                self.training_history['rmse_v'].append(val_metrics['rmse_v'])
                self.training_history['rmse_th'].append(val_metrics['rmse_th'])
                
                # Learning rate scheduling
                if hasattr(self, 'scheduler'):
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['mae_v'])
                    else:
                        self.scheduler.step()
                elif self.multi_objective:
                    if isinstance(self.scheduler_accuracy, ReduceLROnPlateau):
                        self.scheduler_accuracy.step(val_metrics['mae_v'])
                        self.scheduler_constraint.step(val_metrics['mae_v'])
                    else:
                        self.scheduler_accuracy.step()
                        self.scheduler_constraint.step()
                
                # Print progress
                if verbose:
                    epoch_time = time.time() - start_time
                    print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s) - "
                          f"Train Loss: {train_metrics['train_loss']:.6f} | "
                          f"MAE V: {val_metrics['mae_v']:.6f} | "
                          f"MAE Theta: {val_metrics['mae_th']:.6f}")
        
        return self.training_history
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'metrics': metrics
        }
        
        if self.multi_objective:
            checkpoint['optimizer_accuracy_state_dict'] = self.optimizer_accuracy.state_dict()
            checkpoint['optimizer_constraint_state_dict'] = self.optimizer_constraint.state_dict()
        else:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        if self.multi_objective:
            self.optimizer_accuracy.load_state_dict(checkpoint['optimizer_accuracy_state_dict'])
            self.optimizer_constraint.load_state_dict(checkpoint['optimizer_constraint_state_dict'])
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
