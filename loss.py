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


class WLSLoss(nn.Module):
    """
    Weighted least squares loss for power system state estimation.
    """

    def __init__(self, lambda_voltage: float = 1e-4,
                 lambda_phase: float = 1e-8,
                 lambda_power_flow: float = 1e-6):
        """Initialize the physical-aware loss function, given the weights for each term."""
        
        super().__init__()
        self.measurement_weights = {
            "lam_voltage": lambda_voltage,
            "lam_phase": lambda_phase,
            "lam_power_flow": lambda_power_flow
        }
        self.node_weights = torch.tensor([lambda_voltage, lambda_phase, lambda_power_flow, lambda_power_flow])
        self.edge_weights = torch.tensor([lambda_power_flow, lambda_power_flow])

    def _denormalize_measurements(self, measurements: torch.Tensor, mean: torch.Tensor,
                            std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Denormalize measurements using mean and std."""
        return (measurements * std + mean) * mask

    def _extract_measurements_and_covariances(self, input_data: torch.Tensor,
                                            edge_input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract measurements and covariance matrices from input data.
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

    def forward(self, input_data: torch.Tensor, edge_input: torch.Tensor,
                output: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor,
                edge_mean: torch.Tensor, edge_std: torch.Tensor,
                edge_index: torch.Tensor, node_param: torch.Tensor,
                edge_param: torch.Tensor) -> torch.Tensor:
    
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
        (_, _, p_from, q_from,
        p_to, q_to, _, _) = get_pflow(
            torch.cat([v_denorm, theta_angles], dim=1),
            edge_index, node_param, edge_param
        )

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

        return J_measurements

    
class PhysicalLoss(nn.Module):
    """
    Physical-aware loss for power system state estimation using weighted least squares.

    This class implements the measurement-based loss function that compares model predictions
    with actual measurements using weighted least squares, incorporating power flow physics.
    """

    def __init__(self, reg_weight: float = 1e2):
        """
        Initialize the regularization loss function.

        Args:
            reg_weight: Weight for all regularization terms
        """
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, output: torch.Tensor, edge_index: torch.Tensor,
                node_param: torch.Tensor, edge_param: torch.Tensor,
                x_mean: torch.Tensor, x_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regularization terms for physical constraints.

        Args:
            output: Model predictions for node states
            edge_index: Edge connectivity tensor
            node_param: Node parameters
            edge_param: Edge parameters
            x_mean, x_std: Normalization parameters for denormalizing voltage

        Returns:
            Dictionary containing individual regularization terms and total loss
        """
        # Denormalize model outputs
        v_denorm = output[:, 0:1] * x_std[:1] + x_mean[:1]  # Voltage magnitude
        theta_angles = output[:, 1:]  # Voltage angles
        theta_angles *= (1.0 - node_param[:, 1:2])  # Enforce slack bus angle = 0

        # Compute power flows to get loading
        (loading_lines, loading_trafos, _, _, _, _, _, _) = get_pflow(
            torch.cat([v_denorm, theta_angles], dim=1),
            edge_index, node_param, edge_param
        )

        total_loading = loading_lines + loading_trafos

        # Voltage limits: penalize voltages outside [0.9, 1.1] p.u.
        voltage_violation = torch.relu(v_denorm - 1.1) + torch.relu(0.9 - v_denorm)
        J_voltage = self.reg_weight * torch.mean(voltage_violation) ** 2

        # Angle differences: penalize large angle differences across lines
        indices_from = edge_index[0]
        indices_to = edge_index[1]
        angle_diff = torch.abs(
            torch.gather(theta_angles[:, 0], 0, indices_from) -
            torch.gather(theta_angles[:, 0], 0, indices_to)
        )
        J_angle = self.reg_weight * torch.mean(torch.relu(angle_diff - 0.5)) ** 2

        # Loading limits: penalize loading above 150%
        J_loading = self.reg_weight * torch.mean(torch.relu(total_loading - 1.5)) ** 2

        # Total regularization loss
        J_total = J_voltage + J_angle + J_loading

        return {
            'voltage': J_voltage,
            'angle': J_angle,
            'loading': J_loading,
            'total': J_total
        }
    

class CombinedWLSPhysicalLoss(nn.Module):
    def __init__(self, wls_loss: WLSLoss, physical_loss: PhysicalLoss , lambda_wls: float = 1.0, lambda_physical: float = 1.0):
        super().__init__()
        self.wls_loss = wls_loss
        self.physical_loss = physical_loss
        self.lambda_wls = lambda_wls
        self.lambda_physical = lambda_physical
    
    def forward(self, input_data: torch.Tensor, edge_input: torch.Tensor,
                output: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor,
                edge_mean: torch.Tensor, edge_std: torch.Tensor,
                edge_index: torch.Tensor, node_param: torch.Tensor,
                edge_param: torch.Tensor) -> torch.Tensor:
        
        wls_loss = self.wls_loss(input_data, edge_input, output, x_mean, x_std,
                                edge_mean, edge_std, edge_index, node_param, edge_param)
        
        physical_loss = self.physical_loss(output, edge_index, node_param, edge_param, x_mean, x_std)
        
        return self.lambda_wls * wls_loss + self.lambda_physical * physical_loss['total']
