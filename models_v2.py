from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, LeakyReLU
from torch_scatter import scatter

import torch_geometric.nn as nn_geo
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax



class LipschitzNorm(nn.Module):
    """
    Scales pre-softmax logits e_ij per head to control sensitivity.
    Shapes:
      e_ij: [E, H]
      x_i, x_j: [E, H, C]
      index: [E]  – destination node index for softmax groups
    """
    def __init__(self, att_norm: float = 4.0, eps: float = 1e-12):
        super().__init__()
        self.att_norm = float(att_norm)
        self.eps = float(eps)

    def forward(self, e_ij: Tensor, x_i: Tensor, x_j: Tensor, index: Tensor) -> Tensor:
        ni = torch.norm(x_i, dim=-1)         # [E, H]
        nj = torch.norm(x_j, dim=-1)         # [E, H]
        max_nj_per_node = scatter(nj, index, dim=0, reduce='max')  # [N_t, H]
        denom = self.att_norm * (ni + max_nj_per_node[index]) + self.eps
        return e_ij / denom


class GATv2ConvNorm(GATv2Conv):
    """
    GATv2 with optional Lipschitz normalization on attention logits.
    Set `enable_lip=True` to activate.
    """
    def __init__(self,
                 *args,
                 enable_lip: bool = True,
                 lipschitz_norm: Optional[nn.Module] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_lip = enable_lip
        self.lipschitz_norm = lipschitz_norm if lipschitz_norm is not None else LipschitzNorm()

    # Keep the same signature to stay TorchScript-friendly
    def edge_update(self,
                    x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)

        # Pre-softmax logits per edge per head
        e_ij = (x * self.att).sum(dim=-1)  # [E, H]

        # Lipschitz scaling before softmax
        if self.enable_lip and self.lipschitz_norm is not None:
            # x_i, x_j are [E, H, C] already from MessagePassing expansion
            e_ij = self.lipschitz_norm(e_ij, x_i, x_j, index)

        alpha = softmax(e_ij, index, ptr, dim_size)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


import torch
import torch.nn as nn
from torch.nn import Linear, LeakyReLU
import torch_geometric.nn as nn_geo
from typing import Optional

# assumes GATv2ConvNorm and LipschitzNorm are already defined/imported

class GAT_DSSE_NORM(nn.Module):
    def __init__(
        self,
        dim_feat: int,
        dim_dense: int,
        dim_out: int,
        num_layers: int,
        edge_dim: Optional[int],
        heads: int = 1,
        concat: bool = True,
        slope: float = 0.2,
        self_loops: bool = True,
        dropout: float = 0.0,
        nonlin: str = 'leaky_relu',
        lipschitz_norm: Optional[nn.Module] = None,   # allow override
    ):
        super().__init__()
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_feat = dim_feat
        self.dim_dense = dim_dense
        self.edge_dim = edge_dim

        self.channels = dim_feat
        self.heads = heads
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.loop = self_loops

        # hidden width after one GAT layer
        self.dim_hidden = self.channels * self.heads if self.concat else self.channels

        # activation
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = LeakyReLU()
        else:
            raise ValueError('invalid activation type')

        nn_layer = []

        # Lipschitz normalization: default if not provided
        if lipschitz_norm is None:
            lipschitz_norm = LipschitzNorm(att_norm=4.0, eps=1e-12)

        # stack GATv2ConvNorm blocks
        for _ in range(self.num_layers - 1):
            hyper = dict(
                in_channels=self.channels,
                out_channels=self.channels,
                heads=self.heads,
                concat=self.concat,
                negative_slope=self.slope,
                dropout=self.dropout,
                add_self_loops=self.loop,
                edge_dim=self.edge_dim,
                enable_lip=True,
                lipschitz_norm=lipschitz_norm,
            )
            nn_layer.extend([
                (GATv2ConvNorm(**hyper), 'x, edge_index, edge_attr -> x'),
                self.nonlin,
            ])

        # projection head
        nn_layer.extend([
            Linear(in_features=self.dim_hidden, out_features=self.dim_dense),
            self.nonlin,
            Linear(in_features=self.dim_dense, out_features=self.dim_out),
        ])

        self.model = nn_geo.Sequential('x, edge_index, edge_attr', nn_layer)

    def forward(self, x, edge_index, edge_attr):
        return self.model(x, edge_index, edge_attr)

