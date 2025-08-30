"""
Power System State Estimation Loss Functions

This module contains refactored loss functions for power system state estimation,
including weighted least squares (WLS) loss with regularization terms.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from torch_geometric.utils import scatter, get_laplacian
from data._dsml_data import get_pflow, angular_distance


class PowerSystemLoss(nn.Module):
    """
    Power System State Estimation Loss with WLS and regularization.
    
    This class implements a comprehensive loss function for power system state estimation
    that includes weighted least squares for measurements and regularization terms for
    physical constraints (voltage limits, angle differences, loading limits).
    """
    
    def __init__(self, reg_coefs: Dict[str, float]):
        """
        Initialize the power system loss function.
        
        Args:
            reg_coefs: Dictionary containing regularization coefficients:
                - 'lam_v': Weight for voltage measurements
                - 'lam_p': Weight for active power measurements  
                - 'lam_pf': Weight for power flow measurements
                - 'lam_reg': Weight for regularization terms
        """
        super(PowerSystemLoss, self).__init__()
        self.reg_coefs = reg_coefs
        
        # Pre-compute measurement weights as tensors for efficiency
        self.register_buffer('node_weights', torch.tensor([
            reg_coefs['lam_v'], reg_coefs['lam_v'], 
            reg_coefs['lam_p'], reg_coefs['lam_p']
        ]))
        
        self.register_buffer('edge_weights', torch.tensor([
            reg_coefs['lam_pf'], reg_coefs['lam_pf']
        ]))
    
    def _denormalize_measurements(self, measurements: torch.Tensor, mean: torch.Tensor, 
                                std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Denormalize measurements using mean and std."""
        return (measurements * std + mean) * mask
    
    def _extract_measurements_and_covariances(self, input_data: torch.Tensor, 
                                            edge_input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract measurements and covariance matrices from input data.
        
        Args:
            input_data: Node input features [batch*nodes, features]
            edge_input: Edge input features [batch*edges, features]
            
        Returns:
            Tuple of extracted measurements and covariance matrices
        """
        # Node measurements: V, theta, P, Q (every 2nd column starting from 0)
        node_measurements = input_data[:, ::2]  # [batch*nodes, 4]
        node_covariances = input_data[:, 1::2]  # [batch*nodes, 4]
        
        # Edge measurements: Pflow, Qflow (first 4 columns, every 2nd starting from 0)
        edge_measurements = edge_input[:, :4:2]  # [batch*edges, 2]
        edge_covariances = edge_input[:, 1:4:2]  # [batch*edges, 2]
        
        # Create masks for non-zero measurements
        node_mask = node_measurements != 0.0
        edge_mask = edge_measurements != 0.0
        cov_mask = node_covariances != 0.0
        edge_cov_mask = edge_covariances != 0.0
        
        return (node_measurements, node_covariances, edge_measurements, edge_covariances,
                node_mask, edge_mask, cov_mask, edge_cov_mask)
    
    def _compute_power_balance(self, p_from: torch.Tensor, q_from: torch.Tensor,
                             p_to: torch.Tensor, q_to: torch.Tensor,
                             edge_index: torch.Tensor, total_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute power balance at each node using power flows.
        
        Args:
            p_from, q_from: Active/reactive power flowing from source nodes
            p_to, q_to: Active/reactive power flowing to target nodes
            edge_index: Edge connectivity [2, num_edges]
            total_nodes: Total number of nodes
            
        Returns:
            Tuple of (active_power_balance, reactive_power_balance)
        """
        indices_from = edge_index[0]
        indices_to = edge_index[1]
        
        # Sum flows at each node (negative signs follow PandaPower conventions)
        p_balance = (-scatter(p_to, indices_to, dim_size=total_nodes) - 
                    scatter(p_from, indices_from, dim_size=total_nodes))
        q_balance = (-scatter(q_to, indices_to, dim_size=total_nodes) - 
                    scatter(q_from, indices_from, dim_size=total_nodes))
        
        return p_balance, q_balance
    
    def _compute_regularization_terms(self, v_denorm: torch.Tensor, theta_angles: torch.Tensor,
                                    loading: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regularization terms for physical constraints.
        
        Args:
            v_denorm: Denormalized voltage magnitudes
            theta_angles: Voltage angles
            loading: Line/transformer loading
            edge_index: Edge connectivity
            
        Returns:
            Dictionary of regularization terms
        """
        # Voltage limits: penalize voltages outside [0.9, 1.1] p.u.
        voltage_violation = torch.relu(v_denorm - 1.1) + torch.relu(0.9 - v_denorm)
        J_voltage = self.reg_coefs['lam_reg'] * torch.mean(voltage_violation) ** 2
        
        # Angle differences: penalize large angle differences across lines
        indices_from = edge_index[0]
        indices_to = edge_index[1]
        angle_diff = torch.abs(
            torch.gather(theta_angles[:, 0], 0, indices_from) - 
            torch.gather(theta_angles[:, 0], 0, indices_to)
        )
        J_angle = self.reg_coefs['lam_reg'] * torch.mean(torch.relu(angle_diff - 0.5)) ** 2
        
        # Loading limits: penalize loading above 150%
        J_loading = self.reg_coefs['lam_reg'] * torch.mean(torch.relu(loading - 1.5)) ** 2
        
        return {
            'voltage': J_voltage,
            'angle': J_angle, 
            'loading': J_loading
        }
    
    def forward(self, input_data: torch.Tensor, edge_input: torch.Tensor, 
                output: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor,
                edge_mean: torch.Tensor, edge_std: torch.Tensor, 
                edge_index: torch.Tensor, node_param: torch.Tensor, 
                edge_param: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        """
        Compute the complete power system state estimation loss.
        
        Args:
            input_data: Node input features (measurements and covariances)
            edge_input: Edge input features (measurements and covariances)
            output: Model predictions for node states
            x_mean, x_std: Normalization parameters for node features
            edge_mean, edge_std: Normalization parameters for edge features
            edge_index: Edge connectivity tensor
            node_param: Node parameters (voltage levels, slack bus indicators)
            edge_param: Edge parameters (admittances, limits, etc.)
            num_samples: Number of samples in batch (optional, for compatibility)
            
        Returns:
            Total regularized loss
        """
        total_nodes = input_data.shape[0]
        
        # Extract measurements and covariances
        (node_meas, node_cov, edge_meas, edge_cov, 
         node_mask, edge_mask, cov_mask, edge_cov_mask) = self._extract_measurements_and_covariances(
            input_data, edge_input)
        
        # Denormalize measurements
        Z_node = self._denormalize_measurements(node_meas, x_mean[::2], x_std[::2], node_mask)
        Z_edge = self._denormalize_measurements(edge_meas, edge_mean[:4:2], edge_std[:4:2], edge_mask)
        
        # Denormalize covariance matrices (inverse weights)
        R_inv_node = self._denormalize_measurements(node_cov, x_mean[1::2], x_std[1::2], cov_mask)
        R_inv_edge = self._denormalize_measurements(edge_cov, edge_mean[1:4:2], edge_std[1:4:2], edge_cov_mask)
        
        # Denormalize model outputs
        v_denorm = output[:, 0:1] * x_std[:1] + x_mean[:1]  # Voltage magnitude
        theta_angles = output[:, 1:]  # Voltage angles
        theta_angles *= (1.0 - node_param[:, 1:2])  # Enforce slack bus angle = 0
        
        # Compute power flows using physics model
        (loading_lines, loading_trafos, p_from, q_from, 
         p_to, q_to, i_from, i_to) = get_pflow(
            torch.cat([v_denorm, theta_angles], dim=1), 
            edge_index, node_param, edge_param
        )
        
        total_loading = loading_lines + loading_trafos
        
        # Compute power balance at nodes
        p_balance, q_balance = self._compute_power_balance(
            p_from, q_from, p_to, q_to, edge_index, total_nodes)
        
        # Construct predicted measurements
        h_node = torch.cat([v_denorm, theta_angles, 
                           p_balance.unsqueeze(1), q_balance.unsqueeze(1)], dim=1)
        h_edge = torch.cat([p_from.unsqueeze(1), q_from.unsqueeze(1)], dim=1)
        
        # Compute measurement residuals
        delta_node = Z_node - h_node
        delta_edge = Z_edge - h_edge
        
        # Handle angular measurements properly (wrap to [-π, π])
        delta_node[:, 1:2] = angular_distance(Z_node[:, 1:2], h_node[:, 1:2])
        
        # Weighted least squares terms
        wls_node = torch.sum(delta_node**2 * R_inv_node * self.node_weights, dim=1)
        wls_edge = torch.sum(delta_edge**2 * R_inv_edge * self.edge_weights, dim=1)
        
        J_measurements = torch.mean(wls_node) + torch.mean(wls_edge)
        
        # Regularization terms
        reg_terms = self._compute_regularization_terms(
            v_denorm, theta_angles, total_loading, edge_index)
        
        J_regularization = sum(reg_terms.values())
        
        # Total loss
        total_loss = J_measurements + J_regularization
        
        return total_loss


def gsp_wls_edge_refactored(input_data: torch.Tensor, edge_input: torch.Tensor, 
                           output: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor,
                           edge_mean: torch.Tensor, edge_std: torch.Tensor, 
                           edge_index: torch.Tensor, reg_coefs: Dict[str, float],
                           num_samples: int, node_param: torch.Tensor, 
                           edge_param: torch.Tensor) -> torch.Tensor:
    """
    Refactored version of gsp_wls_edge function with improved structure and documentation.
    
    This is a functional interface to the PowerSystemLoss class for backward compatibility.
    
    Args:
        input_data: Node input features (measurements and covariances)
        edge_input: Edge input features (measurements and covariances)  
        output: Model predictions for node states
        x_mean, x_std: Normalization parameters for node features
        edge_mean, edge_std: Normalization parameters for edge features
        edge_index: Edge connectivity tensor
        reg_coefs: Dictionary of regularization coefficients
        num_samples: Number of samples in batch
        node_param: Node parameters
        edge_param: Edge parameters
        
    Returns:
        Total regularized WLS loss
    """
    loss_fn = PowerSystemLoss(reg_coefs)
    return loss_fn(input_data, edge_input, output, x_mean, x_std, 
                   edge_mean, edge_std, edge_index, node_param, edge_param, num_samples)
