"""
Power System State Estimation Models

This module provides various neural network architectures for power system state estimation,
including Graph Neural Networks (GNNs) and traditional neural networks.
Inspired by FairBiNN architecture but adapted for power system applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from typing import Dict, List, Optional, Tuple
import itertools

from data._dsml_data import angular_distance


class PowerSystemLoss(nn.Module):
    """Base class for power system specific loss functions."""
    
    def __init__(self, alpha: float = 1.0, device: Optional[torch.device] = None):
        super(PowerSystemLoss, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class AngularAwareLoss(PowerSystemLoss):
    """Loss function that properly handles angular differences in voltage angles."""
    
    def __init__(self, voltage_weight: float = 1.0, angle_weight: float = 1.0, 
                 alpha: float = 1.0, device: Optional[torch.device] = None):
        super(AngularAwareLoss, self).__init__(alpha, device)
        self.voltage_weight = voltage_weight
        self.angle_weight = angle_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute loss with proper angular distance for voltage angles.
        
        Args:
            predictions: Model predictions [batch_size, 2] (voltage_mag, voltage_angle)
            targets: Ground truth [batch_size, 2] (voltage_mag, voltage_angle)
        """
        # Voltage magnitude loss (standard MSE)
        voltage_loss = F.mse_loss(predictions[:, 0:1], targets[:, 0:1])
        
        # Voltage angle loss (using angular distance)
        angle_diff = angular_distance(predictions[:, 1:2], targets[:, 1:2])
        angle_loss = torch.mean(angle_diff ** 2)
        
        total_loss = (self.voltage_weight * voltage_loss + 
                     self.angle_weight * angle_loss) * self.alpha
        
        return total_loss


class PowerFlowConstraintLoss(PowerSystemLoss):
    """Loss function that enforces power flow physics constraints."""
    
    def __init__(self, constraint_weight: float = 1.0, alpha: float = 1.0, 
                 device: Optional[torch.device] = None):
        super(PowerFlowConstraintLoss, self).__init__(alpha, device)
        self.constraint_weight = constraint_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                power_flow_violations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute loss with power flow constraint violations.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            power_flow_violations: Power flow constraint violations
        """
        base_loss = F.mse_loss(predictions, targets)
        constraint_loss = torch.mean(power_flow_violations ** 2)
        
        total_loss = base_loss + self.constraint_weight * constraint_loss * self.alpha
        return total_loss


class PowerSystemGNNBase(nn.Module):
    """Base class for power system Graph Neural Networks."""
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, 
                 output_dim: int, dropout: float = 0.1):
        super(PowerSystemGNNBase, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Common layers
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PowerSystemGAT(PowerSystemGNNBase):
    """Graph Attention Network for Power System State Estimation."""
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, 
                 output_dim: int, num_heads: int = 4, num_layers: int = 3, 
                 dropout: float = 0.1, use_lipschitz_norm: bool = True):
        super(PowerSystemGAT, self).__init__(node_features, edge_features, hidden_dim, 
                                           output_dim, dropout)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_lipschitz_norm = use_lipschitz_norm
        
        # Import your existing models
        from models import DeepGAT_DSSE
        from ref_models import GAT_DSSE
        
        if use_lipschitz_norm:
            self.gnn_backbone = DeepGAT_DSSE(
                dim_feat=node_features,
                dim_dense=hidden_dim,
                dim_out=output_dim,
                heads=num_heads,
                num_layers=num_layers,
                edge_dim=edge_features,
                norm='lipschitznorm',
                dropout=dropout
            )
        else:
            self.gnn_backbone = GAT_DSSE(
                dim_feat=node_features,
                dim_dense=hidden_dim,
                dim_out=output_dim,
                heads=num_heads,
                num_layers=num_layers,
                edge_dim=edge_features
            )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor) -> torch.Tensor:
        return self.gnn_backbone(node_features, edge_index, edge_features)


class PowerSystemGCN(PowerSystemGNNBase):
    """Graph Convolutional Network for Power System State Estimation."""
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int, 
                 output_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super(PowerSystemGCN, self).__init__(node_features, edge_features, hidden_dim, 
                                           output_dim, dropout)
        self.num_layers = num_layers
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        
        # Input layer
        self.gcn_layers.append(nn.Linear(node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Edge feature processing
        self.edge_processor = nn.Linear(edge_features, hidden_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor) -> torch.Tensor:
        x = node_features
        
        # Process through GCN layers
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Output layer
        x = self.output_layer(x)
        return x


class MultiObjectivePowerSystemModel(nn.Module):
    """
    Multi-objective model for power system state estimation.
    Handles both accuracy and constraint satisfaction objectives.
    """
    
    def __init__(self, backbone_model: PowerSystemGNNBase, 
                 constraint_layers: List[int] = [64, 32, 16],
                 constraint_mode: str = "linear"):
        super(MultiObjectivePowerSystemModel, self).__init__()
        self.backbone = backbone_model
        self.constraint_mode = constraint_mode
        
        # Constraint satisfaction layers
        self.constraint_layers = nn.ModuleList()
        input_dim = backbone_model.hidden_dim
        
        for hidden_dim in constraint_layers:
            self.constraint_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Final constraint output
        self.constraint_output = nn.Linear(input_dim, 1)  # Constraint violation score
    
    def get_accuracy_parameters(self):
        """Get parameters for accuracy optimization."""
        return self.backbone.parameters()
    
    def get_constraint_parameters(self):
        """Get parameters for constraint optimization."""
        return itertools.chain(self.constraint_layers.parameters(), 
                              self.constraint_output.parameters())
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both state estimation and constraint violation.
        
        Returns:
            state_estimation: Voltage magnitudes and angles
            constraint_violation: Constraint violation score
        """
        # Get state estimation from backbone
        state_estimation = self.backbone(node_features, edge_index, edge_features)
        
        # Get intermediate representation for constraint evaluation
        # This would need to be extracted from the backbone model
        # For now, using the state estimation as input to constraint layers
        x = state_estimation
        
        # Process through constraint layers
        for layer in self.constraint_layers:
            x = F.relu(layer(x))
        
        constraint_violation = torch.sigmoid(self.constraint_output(x))
        
        return state_estimation, constraint_violation


def get_power_system_model(model_type: str, node_features: int, edge_features: int, 
                          hidden_dim: int, output_dim: int, **kwargs) -> PowerSystemGNNBase:
    """
    Factory function to create power system models.
    
    Args:
        model_type: Type of model ('gat', 'gcn', 'deepgat')
        node_features: Number of node features
        edge_features: Number of edge features
        hidden_dim: Hidden dimension size
        output_dim: Output dimension size
        **kwargs: Additional model-specific parameters
    
    Returns:
        PowerSystemGNNBase: Instantiated model
    """
    model_type = model_type.lower()
    
    if model_type == 'gat':
        return PowerSystemGAT(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_lipschitz_norm=False,
            **kwargs
        )
    elif model_type == 'deepgat':
        return PowerSystemGAT(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_lipschitz_norm=True,
            **kwargs
        )
    elif model_type == 'gcn':
        return PowerSystemGCN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
