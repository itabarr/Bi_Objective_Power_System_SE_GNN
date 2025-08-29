import pandas as pd
import pandapower as pp
import numpy as np
from _data._dsml_data import data_from_pickles
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random
from typing import Tuple, List, Dict, Any


class PowerSystemDataLoader:
    """
    A class to handle power system data loading and preprocessing.

    This class manages the loading of power system data from pickle files,
    handles train/test splitting, and creates PyTorch Geometric DataLoaders.
    """

    def __init__(self, case='cigre14', batch_size=64, split_coef=0.9,
                 num_nfeat=8, num_efeat=6, num_nmeas=4, num_emeas=2):
        """
        Initialize the PowerSystemDataLoader.

        Args:
            case: Case name for the power system ('cigre' or 'ober')
            batch_size: Batch size for data loaders
            split_coef: Split coefficient for train/test split
            num_nfeat: Number of node features
            num_efeat: Number of edge features
            num_nmeas: Number of node measurements
            num_emeas: Number of edge measurements
        """

        options = ['cigre14', 'cigr14-reswitched', 'ober_sub']
        if case not in options:
            raise ValueError(f"Unsupported case: {case}. Only {options} are supported.")
        
        self.case = case
        self.batch_size = batch_size
        self.split_coef = split_coef
        self.num_nfeat = num_nfeat
        self.num_efeat = num_efeat
        self.num_nmeas = num_nmeas
        self.num_emeas = num_emeas

        
        self.folder = f'data/{self.case}/'

        # Initialize measurement indices based on case
        self.meas_v, self.meas_pflow = self._get_measurement_indices()

        # Data containers
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.train_loader = None
        self.test_loader = None

        # Statistics
        self.x_mean = None
        self.x_std = None
        self.pflow_mean = None
        self.pflow_std = None

    def _get_measurement_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get measurement indices based on the case type."""
        
        if self.case == 'cigre14' or self.case == 'cigr14-reswitched':
            meas_v = np.array([0, 1, 12, 7, 11, 14])
            meas_pflow = np.array([0, 10])

        elif self.case == 'ober_sub':        
            meas_v = np.array([35, 16, 52, 47, 6, 48, 59, 27, 37, 56])
            meas_pflow = np.array([40, 43, 11, 21, 54, 57])
        
        else:
            raise ValueError(f"Unsupported case: {self.case}. Only 'cigre14', 'cigr14-reswitched', and 'ober_sub' are supported.")

        return meas_v, meas_pflow

    def load_data(self) -> Tuple[List, float, float, float, float]:
        """Load data from pickle files using the data_from_pickles function."""
        # Keep data_from_pickles as it is - unchanged
        dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(
            self.folder,
            self.num_nfeat,
            self.num_efeat,
            self.num_nmeas,
            self.num_emeas,
            self.meas_v,
            self.meas_pflow
        )

        # Store the data and statistics
        self.dataset = dataset
        self.x_mean = x_mean
        self.x_std = x_std
        self.pflow_mean = pflow_mean
        self.pflow_std = pflow_std

        # Shuffle dataset
        random.shuffle(self.dataset)

        return dataset, x_mean, x_std, pflow_mean, pflow_std

    def split_dataset(self) -> Tuple[List, List]:
        """Split the dataset into training and testing sets."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        split_idx = int(self.split_coef * len(self.dataset))
        self.X_train = self.dataset[:split_idx]
        self.X_test = self.dataset[split_idx:]

        return self.X_train, self.X_test

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch Geometric DataLoaders for training and testing."""
        if self.X_train is None or self.X_test is None:
            raise ValueError("Datasets not split. Call split_dataset() first.")

        self.train_loader = DataLoader(
            self.X_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.X_test,
            batch_size=self.batch_size,
            shuffle=False
        )

        return self.train_loader, self.test_loader

    def setup_complete_pipeline(self) -> Tuple[DataLoader, DataLoader]:
        """Run the complete data loading pipeline."""
        self.load_data()
        self.split_dataset()
        return self.create_data_loaders()

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        if self.dataset is None:
            return {"status": "No data loaded"}

        return {
            "case": self.case,
            "total_samples": len(self.dataset),
            "train_samples": len(self.X_train) if self.X_train else 0,
            "test_samples": len(self.X_test) if self.X_test else 0,
            "batch_size": self.batch_size,
            "split_coefficient": self.split_coef,
            "voltage_measurements": self.meas_v.tolist(),
            "power_flow_measurements": self.meas_pflow.tolist(),
            "statistics": {
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "pflow_mean": self.pflow_mean,
                "pflow_std": self.pflow_std
            }
        }

    def print_sample_data(self, num_nodes=5, sample_idx=0):
        """
        Print detailed information about the first sample including node features table.

        Args:
            num_nodes: Number of nodes to display in the table (default: 5)
            sample_idx: Index of the sample to analyze (default: 0)
        """
        if self.dataset is None:
            print("No data loaded. Call load_data() first.")
            return

        if sample_idx >= len(self.dataset):
            print(f"Sample index {sample_idx} out of range. Dataset has {len(self.dataset)} samples.")
            return

        # Get the first sample
        sample = self.dataset[sample_idx]

        print(f"\n{'='*60}")
        print(f"DETAILED DATA ANALYSIS - Sample {sample_idx}")
        print(f"{'='*60}")

        # Basic sample information
        print(f"\n📊 Sample Overview:")
        print(f"   • Number of nodes: {sample.x.shape[0]}")
        print(f"   • Node features per node: {sample.x.shape[1]}")
        print(f"   • Number of edges: {sample.edge_index.shape[1]}")
        print(f"   • Edge features per edge: {sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 'N/A'}")

        # Node features table
        print(f"\n🔍 First {min(num_nodes, sample.x.shape[0])} Node Features:")

        # Create feature column names
        feature_names = [f"Feature_{i}" for i in range(sample.x.shape[1])]

        # Extract first num_nodes
        nodes_to_show = min(num_nodes, sample.x.shape[0])
        node_features = sample.x[:nodes_to_show].detach().numpy()

        # Create DataFrame
        df = pd.DataFrame(node_features, columns=feature_names)
        df.index = [f"Node_{i}" for i in range(nodes_to_show)]

        # Print as markdown table
        print(df.to_markdown(floatfmt=".4f"))

        # Edge information if available
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            print(f"\n🔗 Edge Features (first 5 edges):")
            edge_features = sample.edge_attr[:5].detach().numpy()
            edge_names = [f"Edge_Feature_{i}" for i in range(sample.edge_attr.shape[1])]

            df_edges = pd.DataFrame(edge_features, columns=edge_names)
            df_edges.index = [f"Edge_{i}" for i in range(min(5, sample.edge_attr.shape[0]))]
            print(df_edges.to_markdown(floatfmt=".4f"))

        # Target information if available
        if hasattr(sample, 'y') and sample.y is not None:
            print(f"\n🎯 Target Information:")
            print(f"   • Target shape: {sample.y.shape}")
            target_values = sample.y[:5].detach().numpy()
            # Handle multi-dimensional targets
            if len(target_values.shape) > 1:
                print(f"   • Target values (first 5 nodes):")
                for i, target_row in enumerate(target_values):
                    formatted_vals = [f'{val:.4f}' for val in target_row]
                    print(f"     Node_{i}: {formatted_vals}")
            else:
                formatted_vals = [f'{val:.4f}' for val in target_values]
                print(f"   • Target values (first 5): {formatted_vals}")

        # Graph connectivity sample
        print(f"\n🌐 Graph Connectivity (first 10 edges):")
        edges_to_show = min(10, sample.edge_index.shape[1])
        edge_list = []
        for i in range(edges_to_show):
            src = sample.edge_index[0, i].item()
            dst = sample.edge_index[1, i].item()
            edge_list.append(f"Node_{src} → Node_{dst}")

        print("   " + " | ".join(edge_list))

        print(f"\n{'='*60}")

    def print_detailed_statistics(self):
        """Print detailed statistics about the loaded data."""
        if self.dataset is None:
            print("No data loaded. Call load_data() first.")
            return

        print(f"\n{'='*60}")
        print(f"DATASET STATISTICS")
        print(f"{'='*60}")

        # Basic info
        info = self.get_data_info()
        print(f"\n📋 Dataset Configuration:")
        print(f"   • Case: {info['case']}")
        print(f"   • Total samples: {info['total_samples']}")
        print(f"   • Training samples: {info['train_samples']}")
        print(f"   • Test samples: {info['test_samples']}")
        print(f"   • Batch size: {info['batch_size']}")
        print(f"   • Split coefficient: {info['split_coefficient']}")

        print(f"\n📏 Measurement Indices:")
        print(f"   • Voltage measurements: {info['voltage_measurements']}")
        print(f"   • Power flow measurements: {info['power_flow_measurements']}")

        # Statistics
        if self.x_mean is not None:
            print(f"\n📊 Normalization Statistics:")

            # Handle tensor statistics properly
            if hasattr(self.x_mean, 'shape') and len(self.x_mean.shape) > 0:
                x_mean_vals = self.x_mean.detach().numpy() if hasattr(self.x_mean, 'detach') else self.x_mean
                x_std_vals = self.x_std.detach().numpy() if hasattr(self.x_std, 'detach') else self.x_std
                print(f"   • Node features mean (shape {x_mean_vals.shape}): {x_mean_vals}")
                print(f"   • Node features std (shape {x_std_vals.shape}): {x_std_vals}")
            else:
                print(f"   • Node features mean: {float(self.x_mean):.6f}")
                print(f"   • Node features std: {float(self.x_std):.6f}")

            if hasattr(self.pflow_mean, 'shape') and len(self.pflow_mean.shape) > 0:
                pflow_mean_vals = self.pflow_mean.detach().numpy() if hasattr(self.pflow_mean, 'detach') else self.pflow_mean
                pflow_std_vals = self.pflow_std.detach().numpy() if hasattr(self.pflow_std, 'detach') else self.pflow_std
                print(f"   • Power flow mean (shape {pflow_mean_vals.shape}): {pflow_mean_vals}")
                print(f"   • Power flow std (shape {pflow_std_vals.shape}): {pflow_std_vals}")
            else:
                print(f"   • Power flow mean: {float(self.pflow_mean):.6f}")
                print(f"   • Power flow std: {float(self.pflow_std):.6f}")

        # Sample analysis
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            print(f"\n🔬 Sample Analysis:")
            print(f"   • Nodes per sample: {sample.x.shape[0]}")
            print(f"   • Node features: {sample.x.shape[1]}")
            print(f"   • Edges per sample: {sample.edge_index.shape[1]}")

            if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
                print(f"   • Edge features: {sample.edge_attr.shape[1]}")

            if hasattr(sample, 'y') and sample.y is not None:
                print(f"   • Target dimension: {sample.y.shape}")

        print(f"\n{'='*60}")


# Example usage
if __name__ == "__main__":
    # Create data loader with default configuration
    data_loader = PowerSystemDataLoader()

    # Run complete pipeline
    train_loader, test_loader = data_loader.setup_complete_pipeline()

    # Print sample data with node features table
    # data_loader.print_sample_data(num_nodes=5, sample_idx=0)

    # Example of accessing the data variables
    dataset = data_loader.dataset
    x_mean = data_loader.x_mean
    x_std = data_loader.x_std
    pflow_mean = data_loader.pflow_mean
    pflow_std = data_loader.pflow_std









