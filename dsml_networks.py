from typing import Tuple, Union, Optional
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch_geometric.nn as nn_geo
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCN2Conv, FAConv, TAGConv, GINEConv, MessagePassing, GCNConv, ChebConv, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_sparse import SparseTensor, set_diag
from torch.nn import Linear, LeakyReLU
from torch_scatter import scatter


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
    def __init__(self, att_norm=4, recenter=False, scale_individually=True, eps=1e-12):
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
        att_l, att_r, att_e = att
        if self.recenter:
            mean = scatter(src = x, index = index, dim=0, reduce='mean')
            x = x - mean
        norm_x = torch.norm(x, dim=-1) ** 2
        max_norm = scatter(src = norm_x, index = index, dim=0, reduce = 'max').view(-1,1)
        max_norm = torch.sqrt(max_norm[index] + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2

        att_params = [att_l, att_r]
        if att_e is not None:
            att_params.append(att_e)

        # Compute Frobenius norm of concatenated attentions
        concat_att = torch.cat(att_params, dim=-1)
        if not self.scale_individually:
            norm_att = self.att_norm * torch.norm(concat_att)
        else:
            norm_att = self.att_norm * torch.norm(concat_att, dim=-1)

        # Normalize alpha
        alpha = alpha / (norm_att * max_norm + self.eps)
        return alpha

    
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
        self.norm = norm
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

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
            glorot(self.att_e)

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

        edge_feat = None
        alpha_e = None
        if self.lin_edge is not None and edge_attr is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, H, C)
            alpha_e = (edge_feat * self.att_e).sum(dim=-1)

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes=num_nodes)

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

# TODO: edit this network to fit the dsmlGAT
class DeepGAT(nn.Module):
    def __init__(self, idim, hdim, odim, heads, num_layers, norm = None , ogb=False):

        super(DeepGAT, self).__init__()
        self.num_layers = num_layers
        self.norm_name = norm
        self.ogb = ogb
        # Normalization methods
        if self.norm_name == "lipschitznorm":
            self.norm = LipschitzNorm(scale_individually=False)
        elif self.norm_name == "lipschitznorm-si":
            self.norm = LipschitzNorm(scale_individually=True)
        else:
            self.norm = None

        self.layers = nn.ModuleList([DeepGATConv(hdim, hdim, heads=heads, dropout=0.3, norm=self.norm)])
        for _ in range(1,num_layers):
            self.layers.append(DeepGATConv(heads * hdim, hdim, heads=heads, dropout=0.3, norm=self.norm))
        
        self.lin = nn.Linear(idim, hdim)
        self.fc1 = nn.Linear(heads * hdim, odim)
        
    def forward(self, batched_data):
        # x, edge_index = batched_data.x, batched_data.edge_index
        if self.ogb:
            x, edge_index = batched_data.x, batched_data.adj_t
        else:
            x, edge_index = batched_data.x, batched_data.edge_index

        h = self.lin(x)
        # h = x
        for i, layer in enumerate(self.layers):
            h = F.dropout(h, p=0.6, training = self.training)
            h = layer(h, edge_index)
            if i < self.num_layers - 1:
                h = F.elu(h)

        # return F.log_softmax(h, dim=1)  
        return F.log_softmax(self.fc1(h), dim=1)