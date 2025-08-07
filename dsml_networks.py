from typing import Tuple, Union, Optional
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch_geometric.nn as nn_geo
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCN2Conv, FAConv, TAGConv, GINEConv, MessagePassing, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (Adj, Size, OptTensor)
from torch_geometric.utils import add_self_loops, softmax
from torch.nn import Linear, LeakyReLU
from torch_scatter import scatter
from torch_geometric.nn import Sequential as GeoSequential


class gnn_dsse(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, nonlin = 'leaky_relu', main_param = 0.1, K = 3, bias = True, dropout = 0., theta = None, shared_weights = True, cached = True, add_self_loops = True, normalize = True, model = 'gcn2'):
        super().__init__()
        self.channels = dim_feat
        self.main_param = main_param
        self.dim_out = dim_out
        self.K = K
        self.dropout = dropout
        self.bias = bias
        self.theta = theta
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = nn.LeakyReLU()
        else:
            raise Exception('invalid activation type')
        
        nn_layer = []
        if model == 'gcn2':
            for l in range(self.num_layers-1):
                hyperparameters = {"channels":self.channels,"alpha":self.main_param,"theta":self.theta,
                "shared_weights":self.shared_weights,
                "cached":self.cached,
                "normalize":self.normalize,
                "add_self_loops":self.add_self_loops}
                nn_layer.extend([(GCN2Conv(**hyperparameters), 'x, x_0, edge_index -> x'), self.nonlin])
        elif model == 'fagcn':
            for l in range(self.num_layers-1):
                hyperparameters = {"channels":self.channels,"eps":self.main_param,"dropout":self.dropout,
                "cached":self.cached,
                "normalize":self.normalize,
                "add_self_loops":self.add_self_loops}
                nn_layer.extend([(FAConv(**hyperparameters), 'x, x_0,  edge_index -> x'), self.nonlin])
        elif model == 'tagcn':
            for l in range(self.num_layers-1):
                hyperparameters = {"in_channels":self.channels, "out_channels":self.channels,
                "K":self.K,
                "bias":self.bias,
                "normalize":self.normalize}
                nn_layer.extend([(TAGConv(**hyperparameters), 'x, edge_index -> x'), self.nonlin])
        else:
            raise Exception('invalid model type')
        
        nn_layer.extend([Linear(in_features=dim_feat, out_features=dim_dense)])
        nn_layer.extend([Linear(in_features=dim_dense, out_features=dim_out)])
        
        self.model = nn_geo.Sequential('x, x_0, edge_index', nn_layer)

    def forward(self,x, edge_index):
        x_0 = x
        return self.model(x, x_0, edge_index)

class GINE_DSSE(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim, nn = 'mlp', nonlin = 'leaky_relu', eps =0., train_eps = False, model = 'gine'):
        super().__init__()
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_feat=dim_feat
        self.dim_dense = dim_dense
        self.eps = eps
        self.train_eps = train_eps
        self.edge_dim = edge_dim
        self.dim_hidden = dim_feat
        
        if nn == 'mlp':
            self.nn = Linear(in_features = self.dim_feat, out_features = self.dim_hidden)
        else:
            raise Exception('invalid nn type')
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = LeakyReLU()
        else:
            raise Exception('invalid activation type')
        
        nn_layer = []
        if model == 'gine':
            for l in range(self.num_layers-1):
                hyperparameters = {"nn":self.nn, "eps":self.eps, "train_eps": self.train_eps, "edge_dim":self.edge_dim}
                nn_layer.extend([(GINEConv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])
        else:
            raise Exception('invalid model type')
        
        nn_layer.extend([Linear(in_features=self.dim_hidden, out_features=self.dim_dense)])
        nn_layer.extend([Linear(in_features=self.dim_dense, out_features=self.dim_out)])
        
        self.model = nn_geo.Sequential('x, edge_index,edge_attr', nn_layer)

    def forward(self,x, edge_index, edge_attr):
        return self.model(x, edge_index,edge_attr)
    
class GAT_DSSE(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim, heads = 1, concat = True, slope = 0.2, self_loops = True, dropout = 0., nonlin = 'leaky_relu', model = 'gat'):
        super().__init__()
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_feat = dim_feat
        self.dim_dense = dim_dense
        self.edge_dim = edge_dim
        self.dim_hidden = dim_feat
        
        self.channels = dim_feat
        self.heads = heads
        self.dim_out = dim_out
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.loop = self_loops
        self.num_layers = num_layers
        
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = LeakyReLU()
        else:
            raise Exception('invalid activation type')
        
        nn_layer = []
        if model == 'gat':
            for l in range(self.num_layers-1):
                hyperparameters = {"in_channels":self.channels,"out_channels":self.channels,
                "heads": self.heads, "concat": self.concat, "negative_slope": self.slope, "dropout": self.dropout, "add_self_loops": self.loop, "edge_dim":self.edge_dim}
                nn_layer.extend([(GATv2Conv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])
        else:
            raise Exception('invalid model type')
        
        nn_layer.extend([Linear(in_features=self.dim_hidden, out_features=self.dim_dense)])
        nn_layer.extend([Linear(in_features=self.dim_dense, out_features=self.dim_out)])
        
        self.model = nn_geo.Sequential('x, edge_index,edge_attr', nn_layer)

    def forward(self,x, edge_index, edge_attr):
        return self.model(x, edge_index,edge_attr)
    
# LipschitzNorm is the layer from the LipschitzNorm paper
# it calculates a normalizated attention coefficient such that the attention will be lipschitz continuous for self attention that is calculated from a linear projection (no edge features).
# I added att_e to account for the edge features
class LipschitzNorm(nn.Module):
    def __init__(self, att_norm=4, recenter=False, scale_individually=True, eps=1e-6):
        super(LipschitzNorm, self).__init__()
        self.att_norm = att_norm
        self.eps = eps
        self.recenter = recenter
        self.scale_individually = scale_individually

    def forward(self, x, att, alpha, index):
        """
        x: Node features (N, d)
        att: (att_l, att_r)
        alpha: raw attention scores
        index: target node indices
        """

        # NaN detection
        if torch.isnan(x).any():
            print("NaN detected in x input")
            return alpha
        if torch.isnan(alpha).any():
            print("NaN detected in alpha input")
            return alpha

        att_l, att_r, att_e = att
        if self.recenter:
            mean = scatter(src = x, index = index, dim=0, reduce='mean')
            x = x - mean
        norm_x = torch.norm(x, dim=-1) ** 2
        max_norm = scatter(src = norm_x, index = index, dim=0, reduce = 'max').view(-1,1)
        max_norm = torch.sqrt(max_norm[index] + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2
        max_norm = torch.clamp(max_norm, min=self.eps)
        print(f"min x_norm = {torch.min(max_norm)}, max x_norm = {torch.max(max_norm)}")

        att_params = [att_l, att_r]
        if att_e is not None:
            att_params.append(att_e)

        # Compute Frobenius norm of concatenated attentions
        concat_att = torch.cat(att_params, dim=-1)
        if not self.scale_individually:
            norm_att = self.att_norm * torch.norm(concat_att)
        else:
            norm_att = self.att_norm * torch.norm(concat_att, dim=-1)
        print(f"min att_norm = {torch.min(norm_att)}, max att_norm = {torch.max(norm_att)}")

        # Normalize alpha
        # Debug: Check intermediate values
        norm_att = torch.clamp(norm_att, min=1.0)
        denominator = norm_att * max_norm + self.eps
        print(f"min denominator = {torch.min(denominator)}, max denominator = {torch.max(denominator)}")
        if torch.isnan(denominator).any() or (denominator == 0).any():
            print(f"Issue with denominator: norm_att range [{norm_att.min():.6f}, {norm_att.max():.6f}]")
            print(f"max_norm range [{max_norm.min():.6f}, {max_norm.max():.6f}]")
            # Fallback: return original alpha
            return alpha
            
        # Normalize alpha
        alpha = alpha / denominator
        
        # Final NaN check
        if torch.isnan(alpha).any():
            print("NaN detected in alpha output - returning input alpha")
            return att  # Return something safe
            
        return alpha

# This is another Lipschitz Norm implementation, with balanced stabillity and theoretical guarantees.
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

# This is the lipschitznorm DeepGATConv layer and network
# ----- IMPORTANT: This uses the original GATconv attention and not GATv2Conv attention, which might cause a drop in performance -----
# The difference is the order of multplication and activation function in the attention calculation.
# I think developing a lipschitz continuous GATv2Conv is too much work for the project, but maybe if we will have time later on

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
            self.register_parameter('att_e', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    # def reset_parameters(self):
    #     glorot(self.lin_l.weight)
    #     glorot(self.lin_r.weight)
    #     glorot(self.att_l)
    #     glorot(self.att_r)
    #     zeros(self.bias)
    #     if self.lin_edge is not None:
    #         glorot(self.lin_edge.weight)
    #         glorot(self.att_e)
    def reset_parameters(self):
        # Use Xavier/Glorot initialization but with smaller scale
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        
        # Initialize attention parameters more conservatively
        nn.init.xavier_uniform_(self.att_l, gain=0.1)
        nn.init.xavier_uniform_(self.att_r, gain=0.1)
        
        if self.bias is not None:
            zeros(self.bias)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_e, gain=0.1)


    def forward(self, x, edge_index: Adj, edge_attr: Optional[Tensor] = None,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        self.edge_index = edge_index

        if isinstance(x, Tensor):
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)
            else:
                alpha_r = None

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes=num_nodes)

        edge_feat = None
        alpha_e = None
        if self.lin_edge is not None and edge_attr is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, H, C)
            alpha_e = (edge_feat * self.att_e).sum(dim=-1)

        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            alpha=(alpha_l, alpha_r),
            edge_feat=edge_feat,
            alpha_e=alpha_e,
            size=size
        )

        alpha = self._alpha
        self._alpha = None
        self.edge_index = None

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
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
        for l in range(self.num_layers-1):
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
