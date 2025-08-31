import pandas as pd
import pandapower as pp
import numpy as np
from data._dsml_data import data_from_pickles
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
import random
from typing import Tuple, List, Dict, Any, Optional
import os


class PowerSystemDataset(Dataset):
    """
    PyTorch Geometric Dataset for power system data loaded from pickle files.

    This class provides a standard PyTorch Dataset interface for power system data,
    allowing direct initialization from pickle files and integration with PyTorch
    data loading utilities.
    """

    def __init__(
            self,
            case: str = 'cigre14',
            num_node_features: int = 8,
            num_edge_features: int = 6,
            num_node_measurements: int = 4,
            num_edge_measurements: int = 2,
            root: Optional[str] = None,
            ):
        
        self.case = case
        self._num_node_features = num_node_features
        self._num_edge_features = num_edge_features
        self._num_node_measurements = num_node_measurements
        self._num_edge_measurements = num_edge_measurements

        # Set root directory
        if root is None: self.root = f'data/{case}/'

        # Validate case
        options = ['cigre14', 'cigre14_reswitched', 'ober_sub']
        if case not in options: raise ValueError(f"Unsupported case: {case}. Only {options} are supported.")

        # Get measurement indices
        self.meas_v, self.meas_pflow = self._get_measurement_indices()

        # Initialize data containers
        self._data_list = None
        self._x_mean = None
        self._x_std = None
        self._pflow_mean = None
        self._pflow_std = None

        super().__init__(self.root)

        # Load data immediately
        self._load_data()

    def _get_measurement_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get measurement indices based on the case type."""
        
        if self.case == 'cigre14' or self.case == 'cigre14_reswitched':
            meas_v = np.array([0, 1, 12, 7, 11, 14])
            meas_pflow = np.array([0, 10])
        
        elif self.case == 'ober_sub':
            meas_v = np.array([35, 16, 52, 47, 6, 48, 59, 27, 37, 56])
            meas_pflow = np.array([40, 43, 11, 21, 54, 57])
        
        else: raise ValueError(f"Unsupported case: {self.case}")

        return meas_v, meas_pflow

    def _load_data(self):
        """Load data from pickle files."""
        try:

            dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
                self.root + "/",
                self._num_node_features,
                self._num_edge_features,
                self._num_node_measurements,
                self._num_edge_measurements,
                self.meas_v,
                self.meas_pflow
            )
            
            # keep only nfeat and ne features in x

            self._all_data = dataset
            
            for data in dataset:
                data.x = data.x[:, :self._num_node_features]
                data.edge_attr = data.edge_attr[:, :self._num_edge_features]
                

            self._data = dataset
            self._x_features_names = [
                "voltage_magnitude (pu)",
                "voltage_magnitude_inverse_covariance (pu^{-2})",
                "voltage_angle (rad)",
                "voltage_angle_inverse_covariance (rad^{-2})",
                "active_power_injection (MW)",
                "active_power_injection_inverse_covariance (MW^{-2})",
                "reactive_power_injection (MVar)",
                "reactive_power_injection_inverse_covariance (MVar^{-2})",
                "nominal_voltage (kV)",
                "slack_bus_flag (bool)",
                "zero_injection_bus_flag (bool)",
            ]
            
            self._y_features_names = [
                "voltage_magnitude (pu)",
                "voltage_angle (rad)"
            ]
            self._edge_features_names = [
                "active_power_flow_from (MW)",
                "active_power_flow_from_inverse_covariance (MW^{-2})",
                "reactive_power_flow_from (MVar)",
                "reactive_power_flow_from_inverse_covariance (MVar^{-2})",
                "series_conductance (S)",
                "series_susceptance (S)",
                "series_conductance (S)",
                "series_susceptance (S)",
                "shunt_conductance (S)",
                "shunt_susceptance (S)",
                "closed_line (bool)",
                "phase_shift (rad)",
                "max_current_or_rated_power (kA or MVA)",
            ]

            self._x_mean = x_mean
            self._x_std = x_std
            self._edge_features_mean = pflow_mean
            self._edge_features_std = pflow_std

        except Exception as e: raise RuntimeError(f"Failed to load data from {self.root}: {str(e)}")

    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get normalization statistics for the dataset."""
        return {
            "x_features_mean": self._x_mean,
            "x_features_std": self._x_std,
            "edge_features_mean": self._edge_features_mean,
            "edge_features_std": self._edge_features_std
        }

    def len(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self._data)

    def get(self, idx: int) -> Data:
        """Get a single data sample by index."""
        if idx >= len(self._data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._data)}")
        return self._data[idx]

    def split_dataset(self, split_coef: float = 0.9) -> Tuple['PowerSystemDataset', 'PowerSystemDataset']:
        """
        Split the dataset into test and train datasets.

        Args:
            split_coef: Coefficient for train/test split (train ratio)

        Returns:
            Tuple of (test_dataset, train_dataset)
        """
        # Split the dataset
        data_list = self._data.copy()
        random.shuffle(data_list)

        split_idx = int(split_coef * len(data_list))
        train_dataset = data_list[:split_idx]
        test_dataset = data_list[split_idx:]
        
        return test_dataset, train_dataset
    
    def __repr__(self) -> str:
        """String representation of the dataset."""

        _str = f"\nPower System Dataset\n\n"
        _str = _str + f"case = {self.case}"

        number_of_graphs = len(self._data)
        _str = _str + f"\nNumber of graphs: {number_of_graphs}"

        number_of_nodes_per_graph = self._data[0].x.shape[0]
        _str = _str + f"\nNumber of nodes per graph: {number_of_nodes_per_graph}"

        number_of_edges_per_graph = self._data[0].edge_index.shape[1]
        _str = _str + f"\nNumber of edges per graph: {number_of_edges_per_graph}"

        # Node features
        nodes_features_names = self._x_features_names[:self._num_node_features]
        nodes_features_len = self._data[0].x.shape[1]
        _str = _str + f"\n\nNumber of node features: {nodes_features_len}"
        df = pd.DataFrame(columns=['Feature', 'Unit'])
        df['Feature'] = [name.split(' (')[0] for name in nodes_features_names]
        df['Unit'] = [name.split(' (')[1].split(')')[0] for name in nodes_features_names]
        df['Mean'] = list(self._x_mean) + [np.nan] * (len(nodes_features_names) - len(list(self._x_mean)))
        df['Std'] = list(self._x_std) + [np.nan] * (len(nodes_features_names) - len(list(self._x_std)))
        df.index = [f"({i})" for i in range(nodes_features_len)]
        df.index.name = 'Index'
        _str = _str + f"\n{df.to_markdown()}"
        
        # Edge features
        edge_features_names = self._edge_features_names[:self._num_edge_features]
        edge_features_len = self._data[0].edge_attr.shape[1]
        _str = _str + f"\n\nNumber of edge features: {edge_features_len}\n"
        df = pd.DataFrame(columns=['Feature', 'Unit'])
        df['Feature'] = [name.split(' (')[0] for name in edge_features_names]
        df['Unit'] = [name.split(' (')[1].split(')')[0] for name in edge_features_names]
        df['Mean'] =list(self._edge_features_mean) + [np.nan] * (len(edge_features_names) - len(list(self._edge_features_mean)))
        df['Std'] = list(self._edge_features_std) + [np.nan] * (len(edge_features_names) - len(list(self._edge_features_std)))
        df.index = [f"({i})" for i in range(edge_features_len)]
        df.index.name = 'Index'
        _str = _str + f"\n{df.to_markdown()}"

        # Example graph
        example_index = 5
        example_graph = self._data[example_index]
        _str = _str + f"\n\nExample graph (index {example_index}):"
        _str = _str + f"\nNumber of nodes: {example_graph.x.shape[0]}"
        _str = _str + f"\nNumber of edges: {example_graph.edge_index.shape[1]}"
        _str = _str + f"\nNode features shape: {example_graph.x.shape}"
        _str = _str + f"\nEdge features shape: {example_graph.edge_attr.shape}"
        _str = _str + f"\nTarget shape: {example_graph.y.shape}"

        # First 5 nodes 
        example_nodes_len = 5

        node_data = example_graph.x[:example_nodes_len].detach().numpy()
        df = pd.DataFrame(
            node_data.T,  # transpose so features are rows
            index=nodes_features_names[:node_data.shape[1]],
            columns=[f"Node {i}" for i in range(example_nodes_len)]
        )
        _str += "\n\nFirst 5 node features:"
        _str += f"\n{df.to_markdown()}"

        # First 5 edges 
        example_edges_len = 5
        edge_index = example_graph.edge_index[:, :example_edges_len].detach().numpy()
        edge_index = np.array(["{} -> {}".format(edge_index[0, i], edge_index[1, i]) for i in range(example_edges_len)])
        edge_index = edge_index.reshape(-1, 1)
        edge_data = example_graph.edge_attr[:example_edges_len].detach().numpy()
        
        
        edge_combo = np.concatenate([edge_index, edge_data], axis=1)

        df = pd.DataFrame(
            edge_combo.T,  # transpose so features are rows
            index=["Source -> Target"] + edge_features_names[:edge_data.shape[1]],
            columns=[f"Edge {i}" for i in range(example_edges_len)]
        )
        _str += "\n\nFirst 5 edge features:"
        _str += f"\n{df.to_markdown()}"
        
        return _str


if __name__ == "__main__":
    case = 'cigre14'
    # case = 'cigre14_reswitched'
    # case = 'ober_sub'
    
    dataset = PowerSystemDataset(case = case)


    print(dataset)
   
