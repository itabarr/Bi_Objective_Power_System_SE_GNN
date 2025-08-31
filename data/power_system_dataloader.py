
from torch_geometric.loader import DataLoader
from typing import List
import torch_geometric


class PowerSystemDataLoader(DataLoader):
    """
    DataLoader wrapper that accepts a list of torch_geometric.data.Data objects.

    This class provides a convenient interface for loading power system data in batches.
    """

    def __init__(
            self,
            dataset: List[torch_geometric.data.Data],
            batch_size: int = 64,
            shuffle: bool = False,
            **kwargs
            ):
        """
        Initialize the PowerSystemDataLoader.

        Args:
            dataset: List of torch_geometric.data.Data objects
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            **kwargs: Additional arguments passed to DataLoader
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

   





