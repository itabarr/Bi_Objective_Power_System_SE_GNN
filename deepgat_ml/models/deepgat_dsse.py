from typing import Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, Linear, LeakyReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter
from torch_geometric.nn import Sequential as GeoSequential


class StableLipschitzNorm(nn.Module):
    def __init__(self, att_norm=4, recenter=False, scale_individually=True, eps=1e-8):
        super(StableLipschitzNorm, self).__init__()
        self.att_norm = att_norm
        self.eps = eps
        self.recenter = recenter
        self.scale_individually = scale_individually

    def forward(self, x, att, alpha, index):
        att_l, att_r, att_e = att
        
        if self.recenter:
            mean = scatter(src=x, index=index, dim=0, reduce='mean')
            x = x - mean[index]
        
        norm_x_squared = torch.sum(x * x, dim=-1, keepdim=False)
        max_norm_squared = scatter(src=norm_x_squared, index=index, dim=0, reduce='max')
        max_norm_squared = max_norm_squared.view(-1, 1)
        
        combined_norm_squared = max_norm_squared[index] + norm_x_squared
        combined_norm_squared = torch.clamp(combined_norm_squared, min=self.eps * self.eps)
        max_norm = torch.sqrt(combined_norm_squared)

        att_params = [att_l, att_r]
        if att_e is not None:
            att_params.append(att_e)

        concat_att = torch.cat(att_params, dim=-1)
        
        # IMPROVEMENT: Add bounds on norm_att
        if not self.scale_individually:
            norm_att = torch.norm(concat_att, p='fro')
            norm_att = torch.clamp(norm_att, min=self.eps, max=100.0)  # Prevent extreme values
            norm_att = self.att_norm * norm_att
        else:
            norm_att = torch.norm(concat_att, p=2, dim=-1)
            norm_att = torch.clamp(norm_att, min=self.eps, max=100.0)
            norm_att = self.att_norm * norm_att

        denominator = norm_att * max_norm + self.eps
        alpha_normalized = alpha / denominator
        
        # IMPROVEMENT: Better fallback strategy
        if torch.isnan(alpha_normalized).any() or torch.isinf(alpha_normalized).any():
            print("Warning: Numerical instability in LipschitzNorm, using fallback")
            # Return original alpha with mild scaling to be safe
            return alpha * 0.1
        
        # IMPROVEMENT: Reasonable bounds for softmax stability
        # Values beyond ±10 cause softmax to saturate anyway
        alpha_normalized = torch.clamp(alpha_normalized, min=-8.0, max=8.0)
        
        return alpha_normalized


class DeepGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 norm=None, edge_dim: Optional[int] = None, **kwargs):
        super(DeepGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        if norm == "lipschitznorm":
            self.norm = StableLipschitzNorm(scale_individually=False)
        elif norm == 'lipschitznorm-si':
            self.norm = StableLipschitzNorm(scale_individually=True)
        else:
            self.norm = None
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
            self.att_e = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_e = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        if self.edge_dim is not None:
            glorot(self.lin_edge.weight)
            glorot(self.att_e)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]], edge_index: Tensor,
                edge_attr: OptTensor = None, size: Optional[Tuple[int, int]] = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x_l = x_r = self.lin_l(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x_l.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value='mean',
                    num_nodes=num_nodes)

        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        alpha_e = None
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr).view(-1, H, C)
            alpha_e = (edge_attr * self.att_e).sum(dim=-1)

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r),
                             alpha_e=alpha_e, edge_attr=edge_attr, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert self._alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, self._alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(self._alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j, alpha_i, alpha_e,
                index, ptr, size_i) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if alpha_e is not None:
            alpha = alpha + alpha_e

        if self.att_e is not None:
            att = (self.att_l, self.att_r, self.att_e) # added the att_e parameter to the norm function to account for edge features
        else:
            att = (self.att_l, self.att_r)
        if self.norm is not None:
            alpha = self.norm(x_j, att=att, alpha=alpha, index=index) 

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


# TODO: make sure this network works with the new DeepGATConv layer, need to run and debug
class DeepGAT_DSSE(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim,
                 heads=1, concat=True, slope=0.2, self_loops=True, dropout=0.,
                 nonlin='leaky_relu', norm=None):
        super().__init__()
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_feat = dim_feat
        self.dim_dense = dim_dense
        self.edge_dim = edge_dim
        self.dim_hidden = dim_feat

        self.channels = dim_feat
        self.heads = heads
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.self_loops = self_loops

        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = LeakyReLU(slope)
        else:
            raise Exception('Invalid activation type')

        nn_layer = []
        # Build DeepGAT layers
        for _ in range(self.num_layers-1):
            hyperparameters = {
                "in_channels": self.channels,
                "out_channels": self.channels,
                "heads": self.heads,
                "concat": self.concat,
                "negative_slope": self.slope,
                "dropout": self.dropout,
                "add_self_loops": self.self_loops,
                "edge_dim": self.edge_dim,
                "norm": norm
            }
            nn_layer.extend([
                (DeepGATConv(**hyperparameters), 'x, edge_index, edge_attr -> x'),
                self.nonlin
            ])

        # Final dense layers
        nn_layer.extend([
            Linear(in_features=self.dim_hidden, out_features=self.dim_dense),
            nn.ReLU(),
            Linear(in_features=self.dim_dense, out_features=self.dim_out)
        ])

        self.model = GeoSequential('x, edge_index, edge_attr', nn_layer)

    def forward(self, x, edge_index, edge_attr):
        return self.model(x, edge_index, edge_attr)
