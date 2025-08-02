import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.nn as nn_geo
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCN2Conv, FAConv, TAGConv, GINEConv, MessagePassing, GCNConv, ChebConv, GATv2Conv
from torch.nn import Linear, LeakyReLU
from torch_geometric.utils import degree
from torch_scatter import scatter


class gnn_dsse(nn.Module):
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, nonlin = 'leaky_relu', main_param =0.1, K = 3, bias = True, dropout = 0., theta = None, shared_weights = True, cached = True, add_self_loops = True, normalize = True, model = 'gcn2'):
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
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim, heads =1, concat = True, slope = 0.2, self_loops = True, dropout = 0., nonlin = 'leaky_relu', model = 'gat'):
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
    
# LipschitzNorm is a custom normalization layer that normalizes the features of each node based on its neighbors, based on the lipschitz norm paper
class LipschitzNorm(nn.Module):
    def __init__(self, att_norm = 4, recenter = False, scale_individually = True, eps = 1e-12):
        super(LipschitzNorm, self).__init__()
        self.att_norm = att_norm
        self.eps = eps
        self.recenter = recenter
        self.scale_individually = scale_individually

    def forward(self, x, att, alpha, index):
        att_l, att_r = att
        
        if self.recenter:
            mean = scatter(src = x, index = index, dim=0, reduce='mean')
            x = x - mean


        norm_x = torch.norm(x, dim=-1) ** 2
        max_norm = scatter(src = norm_x, index = index, dim=0, reduce = 'max').view(-1,1)
        max_norm = torch.sqrt(max_norm[index] + norm_x)  # simulation of max_j ||x_j||^2 + ||x_i||^2

        
        # scaling_factor =  4 * norm_att , where att = [ att_l | att_r ]         
        if self.scale_individually == False:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim = -1))
        else:
            norm_att = self.att_norm * torch.norm(torch.cat((att_l, att_r), dim=-1), dim = -1)

        alpha = alpha / ( norm_att * max_norm + self.eps )
        return alpha