"""
DeepGAT Configuration File

Contains hyperparameters and configuration settings for DeepGAT training.
"""

import torch
import numpy as np

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data configuration
DATA_CONFIG = {
    'case': 'cigre14',
    'folder': 'data/cigre14/',
    'phase_shift': True,
    'num_nfeat': 8,
    'num_efeat': 6,
    'num_nmeas': 4,
    'num_emeas': 2,
}

# Measurement indices for different grids
MEASUREMENT_INDICES = {
    'cigre14': {
        'meas_v': np.array([0, 1, 12, 7, 11, 14]),
        'meas_p': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        'meas_q': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    }
}

# Model hyperparameters
MODEL_CONFIG = {
    'dim_nodes': 8,
    'dim_lines': 6,
    'dim_hid': 64,
    'dim_out': 4,
    'heads': 4,
    'gnn_layers': 3,
    'dropout_rate': 0.3,
    'L': 5,
    'norm': 'lipschitznorm'
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 600,
    'learning_rate': 3e-3,
    'optimizer': 'Adamax',
    'load_model': False,
    'model_path': '',
}

# Loss function coefficients
LOSS_COEFFICIENTS = {
    'mu_v': 1e-1,
    'mu_theta': 1e-1,
    'lam_v': 1e-4,
    'lam_p': 1e-8,
    'lam_pf': 1e-6,
    'lam_reg': 1e2
}

# Evaluation thresholds
EVALUATION_CONFIG = {
    'thresh': 0,
}

def get_measurement_indices(case_name):
    """Get measurement indices for a specific case."""
    return MEASUREMENT_INDICES.get(case_name, MEASUREMENT_INDICES['cigre14'])

def get_data_folder(case_name):
    """Get data folder path for a specific case."""
    return f'data/{case_name}/'
