# DeepGAT Machine Learning Package

This package contains the DeepGAT model implementation and training utilities for power system state estimation using Graph Neural Networks with Lipschitz normalization.

## Package Structure

```
deepgat_ml/
├── models/
│   ├── __init__.py
│   └── deepgat_dsse.py          # DeepGAT model definition
├── training/
│   ├── __init__.py
│   ├── train_deepgat.py         # Main training script
│   └── evaluate_deepgat.py      # Model evaluation script
├── configs/
│   ├── __init__.py
│   └── deepgat_config.py        # Configuration and hyperparameters
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   └── training_utils.py        # Training utilities and trainer class
└── README.md                    # This file
```

## Features

- **DeepGAT Model**: Graph Attention Network with Lipschitz normalization for stable training
- **Pickle Data Loading**: Loads pre-processed data from pickle files (not responsible for dataset creation)
- **Configurable Training**: Easy-to-modify configuration files for hyperparameters
- **Comprehensive Evaluation**: Detailed metrics for voltage, angle, and loading predictions
- **Checkpoint Support**: Save and load model checkpoints during training

## Quick Start

### Training a Model

```bash
# Basic training with default settings
python deepgat_ml/training/train_deepgat.py

# Training with custom parameters
python deepgat_ml/training/train_deepgat.py --case cigre14 --epochs 600 --batch_size 64 --lr 3e-3

# Resume training from checkpoint
python deepgat_ml/training/train_deepgat.py --load_model path/to/checkpoint.pt
```

### Evaluating a Model

```bash
# Evaluate a trained model
python deepgat_ml/training/evaluate_deepgat.py --model_path path/to/model.pt --case cigre14
```

## Configuration

The main configuration is in `configs/deepgat_config.py`:

- **MODEL_CONFIG**: Model architecture parameters (layers, heads, dimensions)
- **TRAINING_CONFIG**: Training parameters (epochs, batch size, learning rate)
- **DATA_CONFIG**: Data loading parameters (case name, features)
- **LOSS_COEFFICIENTS**: Loss function weighting coefficients

## Data Requirements

This package expects data to be pre-processed and saved as pickle files in the following structure:
```
data/
└── {case_name}/
    ├── train_data.pkl
    └── test_data.pkl
```

**Note**: This package is NOT responsible for dataset creation. It only loads existing pickle files created by the data generation pipeline.

## Model Architecture

The DeepGAT model includes:
- **DeepGATConv**: Custom GAT convolution layer with edge features
- **StableLipschitzNorm**: Lipschitz normalization for training stability
- **Multi-head Attention**: Configurable number of attention heads
- **Dropout**: Configurable dropout for regularization

## Training Features

- **Progress Tracking**: Visual progress bars during training
- **Automatic Checkpointing**: Saves best and final models
- **Comprehensive Metrics**: RMSE, MAE, and proportional standard deviation
- **GPU Support**: Automatic CUDA detection and usage
- **Configurable Loss**: Multi-component loss function with tunable coefficients

## Example Usage in Code

```python
from deepgat_ml.models import DeepGAT_DSSE
from deepgat_ml.configs.deepgat_config import MODEL_CONFIG
from deepgat_ml.utils.data_loader import create_data_loaders
from deepgat_ml.utils.training_utils import create_trainer

# Initialize model
model = DeepGAT_DSSE(**MODEL_CONFIG)

# Load data
train_loader, test_loader, data_info = create_data_loaders(data_config)

# Create trainer
trainer = create_trainer(model, training_config, device)

# Train model
for epoch in range(epochs):
    loss = trainer.train_epoch(train_loader, loss_coefficients)
    metrics = trainer.evaluate(test_loader, meas_indices)
```

## Dependencies

- PyTorch
- PyTorch Geometric
- NumPy
- TorchMetrics
- torch_scatter
- torch_sparse

## Notes for ML Engineers

- The model uses Lipschitz normalization for stable training of deep GAT networks
- Hyperparameters are easily configurable through the config files
- The package includes debugging utilities that print data structure information
- All training metrics are tracked and can be saved with checkpoints
- The evaluation script provides comprehensive performance analysis
