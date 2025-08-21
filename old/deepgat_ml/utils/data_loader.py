"""
Data Loading Utilities for DeepGAT

This module provides utilities for loading data from pickle files for DeepGAT training.
Note: This module is not responsible for dataset creation, only loading existing pickle files.
"""

import sys
import os
import torch
from torch_geometric.loader import DataLoader

# Add parent directory to path to import from main project
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dsml_data import data_from_pickles


class DeepGATDataLoader:
    """
    Data loader class for DeepGAT that loads data from pickle files.
    
    Note: This class assumes that pickle files have already been created
    by the data generation pipeline. It is not responsible for dataset creation.
    """
    
    def __init__(self, data_folder, case_name, meas_indices, batch_size=64):
        """
        Initialize the data loader.
        
        Args:
            data_folder (str): Path to the folder containing pickle files
            case_name (str): Name of the case (e.g., 'cigre14')
            meas_indices (dict): Dictionary containing measurement indices
            batch_size (int): Batch size for data loading
        """
        self.data_folder = data_folder
        self.case_name = case_name
        self.meas_indices = meas_indices
        self.batch_size = batch_size
        
        # Load data from pickle files
        self._load_data()
    
    def _load_data(self):
        """Load training and testing data from pickle files."""
        print(f"Loading data from pickle files in {self.data_folder}")

        # Load data using the existing data_from_pickles function
        # Note: This assumes the pickle files exist and were created by the data pipeline
        # The function returns: data_list, x_mean, x_std, pflow_mean, pflow_std
        data_list, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
            folder=self.data_folder,
            num_nfeat=11,  # Updated to match actual data
            num_efeat=13,  # Updated to match actual data
            num_nmeas=4,   # From DATA_CONFIG
            num_emeas=2,   # From DATA_CONFIG
            meas_v=self.meas_indices['meas_v'],
            meas_pflow=self.meas_indices['meas_p']  # Use meas_p as meas_pflow
        )

        # Split data into train and test (90% train, 10% test)
        import random
        random.shuffle(data_list)
        split_idx = int(0.9 * len(data_list))
        self.train_data = data_list[:split_idx]
        self.test_data = data_list[split_idx:]

        # Store normalization parameters for potential use
        self.x_mean = x_mean
        self.x_std = x_std
        self.pflow_mean = pflow_mean
        self.pflow_std = pflow_std
        
        print(f"Loaded {len(self.train_data)} training samples")
        print(f"Loaded {len(self.test_data)} test samples")
        
        # Print example data structure for debugging (as per user preference)
        if len(self.train_data) > 0:
            sample = self.train_data[0]
            print(f"Sample data structure:")
            print(f"  - Node features shape: {sample.x.shape}")
            print(f"  - Edge features shape: {sample.edge_attr.shape}")
            print(f"  - Edge index shape: {sample.edge_index.shape}")
            print(f"  - Target shape: {sample.y.shape}")
            print(f"  - Node features example values: {sample.x[:3]}")  # First 3 nodes
            print(f"  - Edge features example values: {sample.edge_attr[:3]}")  # First 3 edges
    
    def get_train_loader(self):
        """Get training data loader."""
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def get_test_loader(self):
        """Get test data loader."""
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    
    def get_data_info(self):
        """Get information about the loaded data."""
        if len(self.train_data) > 0:
            sample = self.train_data[0]
            return {
                'num_node_features': sample.x.shape[1],
                'num_edge_features': sample.edge_attr.shape[1],
                'num_target_features': sample.y.shape[1],
                'train_samples': len(self.train_data),
                'test_samples': len(self.test_data)
            }
        return None


def create_data_loaders(config):
    """
    Create data loaders from configuration.

    Args:
        config: Configuration object with data settings

    Returns:
        tuple: (train_loader, test_loader, data_info)
    """
    # Import here to avoid circular imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'configs'))
    from deepgat_config import get_measurement_indices, get_data_folder
    
    # Get measurement indices for the case
    meas_indices = get_measurement_indices(config['case'])
    
    # Create data loader
    data_loader = DeepGATDataLoader(
        data_folder=config['folder'],
        case_name=config['case'],
        meas_indices=meas_indices,
        batch_size=config.get('batch_size', 64)
    )
    
    # Get data loaders
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    data_info = data_loader.get_data_info()
    
    return train_loader, test_loader, data_info
