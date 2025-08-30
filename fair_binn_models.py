from typing import Tuple, Union, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATv2Conv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (Adj, Size, OptTensor)
from torch_geometric.utils import add_self_loops, softmax
from torch.nn import Linear, LeakyReLU
from torch_scatter import scatter
from torch_geometric.nn import Sequential as GeoSequential
from torch.optim.lr_scheduler import ExponentialLR


class ConstraintLoss(nn.Module):
    """Base class for fairness constraint losses."""
    def __init__(self, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constraints)

    def forward(self, X, out, sensitive, y=None):
        # Handle shape mismatch: sensitive is per-node, out might be multi-dimensional
        if out.dim() > 1 and out.size(-1) > 1:
            # For multi-output case, use the first output dimension or mean
            out_for_fairness = out.mean(dim=-1) if out.size(-1) > 1 else out.squeeze(-1)
        else:
            out_for_fairness = out.squeeze() if out.dim() > 1 else out
            
        # Ensure sensitive has the same shape as the output used for fairness
        if sensitive.dim() != out_for_fairness.dim():
            sensitive = sensitive.view_as(out_for_fairness)
            
        if isinstance(y, torch.Tensor):
            if y.dim() > 1 and y.size(-1) > 1:
                y = y.mean(dim=-1) if y.size(-1) > 1 else y.squeeze(-1)
            y = y.view_as(out_for_fairness)
            
        out_for_fairness = torch.sigmoid(out_for_fairness)
        mu = self.mu_f(X=X, out=out_for_fairness, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons


class DemographicParityLoss(ConstraintLoss):
    """Demographic parity fairness constraint loss."""
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(DemographicParityLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y=None):
        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v
            expected_values_list.append(out[idx_true].mean())
        expected_values_list.append(out.mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y=None):
        return super(DemographicParityLoss, self).forward(X, out, sensitive)


class FairGCN_Body(nn.Module):
    """GCN body adapted for PyTorch Geometric from FairBiNN."""
    def __init__(self, nfeat, nhid, dropout):
        super(FairGCN_Body, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.gc1(x, edge_index))
        x = self.dropout(x)
        x = self.gc2(x, edge_index)
        return x


class FairGAT_Body(nn.Module):
    """GAT body adapted for PyTorch Geometric from FairBiNN."""
    def __init__(self, num_layers, in_dim, num_hidden, heads, feat_drop, attn_drop, 
                 negative_slope, residual, edge_dim=None):
        super(FairGAT_Body, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        
        # Ensure heads list has correct length
        if len(heads) != num_layers + 1:
            if isinstance(heads, int):
                heads = [heads] * num_layers + [1]
            else:
                heads = heads + [1] * (num_layers + 1 - len(heads))
        
        # Input projection
        self.gat_layers.append(GATv2Conv(
            in_dim, num_hidden, heads=heads[0],
            dropout=feat_drop, edge_dim=edge_dim,
            add_self_loops=True, bias=True
        ))
        
        # Hidden layers
        for l in range(1, num_layers):
            self.gat_layers.append(GATv2Conv(
                num_hidden * heads[l-1], num_hidden, heads=heads[l],
                dropout=feat_drop, edge_dim=edge_dim,
                add_self_loops=True, bias=True
            ))
        
        # Output projection
        if num_layers > 1:
            self.gat_layers.append(GATv2Conv(
                num_hidden * heads[-2], num_hidden, heads=heads[-1],
                dropout=feat_drop, edge_dim=edge_dim,
                add_self_loops=True, bias=True
            ))

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        for l in range(len(self.gat_layers)):
            h = self.gat_layers[l](h, edge_index, edge_attr)
            if l < len(self.gat_layers) - 1:
                h = self.activation(h)
                h = h.flatten(1) if h.dim() > 2 else h
        
        if h.dim() > 2:
            h = h.mean(1)
        
        return h


def get_fair_model(nfeat, model_type, **kwargs):
    """Factory function to create fair GNN models."""
    if model_type == "GCN":
        return FairGCN_Body(nfeat, kwargs.get('num_hidden', 64), kwargs.get('dropout', 0.5))
    elif model_type == "GAT":
        num_layers = kwargs.get('num_layers', 2)
        heads = kwargs.get('heads', [4, 4, 1])
        
        if isinstance(heads, int):
            heads = [heads] * num_layers + [1]
        elif len(heads) != num_layers + 1:
            heads = heads + [1] * (num_layers + 1 - len(heads))
            
        return FairGAT_Body(
            num_layers, nfeat, kwargs.get('num_hidden', 64), 
            heads, kwargs.get('dropout', 0.5), kwargs.get('attn_drop', 0.5),
            kwargs.get('negative_slope', 0.2), kwargs.get('residual', False),
            kwargs.get('edge_dim', None)
        )
    else:
        raise ValueError(f"Model {model_type} not implemented")


class FairGNN_DSSE(nn.Module):
    """Fair Graph Neural Network for Distribution System State Estimation."""
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim,
                 model_type="GAT", heads=None, dropout=0.5, attn_drop=0.5,
                 negative_slope=0.2, residual=False, fairness_alpha=100,
                 sensitive_classes=[0, 1]):
        super(FairGNN_DSSE, self).__init__()
        
        self.dim_feat = dim_feat
        self.dim_dense = dim_dense
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.model_type = model_type
        
        if heads is None:
            heads = [4] * (num_layers - 1) + [1]
        
        model_kwargs = {
            'num_hidden': dim_dense,
            'dropout': dropout,
            'num_layers': num_layers,
            'heads': heads,
            'attn_drop': attn_drop,
            'negative_slope': negative_slope,
            'residual': residual,
            'edge_dim': edge_dim
        }
        
        self.GNN = get_fair_model(dim_feat, model_type, **model_kwargs)
        self.fairness_layer = nn.Sequential(nn.Linear(dim_dense, dim_dense), nn.ReLU())
        self.classifier = nn.Linear(dim_dense, dim_out)
        
        self.criterion = nn.MSELoss()
        self.criterion_fairness = DemographicParityLoss(
            sensitive_classes=sensitive_classes, 
            alpha=fairness_alpha, 
            p_norm=2
        )
        
        self.G_loss = 0
        self.F_loss = 0

    def forward(self, x, edge_index, edge_attr=None):
        z = self.GNN(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)
        return y, z

    def setup_optimizers(self, lr_g=1e-3, lr_f=1e-9, weight_decay=1e-5):
        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        F_params = list(self.fairness_layer.parameters())
        
        self.optimizer_G = torch.optim.Adam(G_params, lr=lr_g, weight_decay=weight_decay)
        self.optimizer_F = torch.optim.Adam(F_params, lr=lr_f, weight_decay=weight_decay)
        
        self.scheduler_G = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_F = ExponentialLR(self.optimizer_F, gamma=0.99)

    def optimize_step(self, x, edge_index, edge_attr, labels, sensitive_attr, idx_train=None):
        if idx_train is None:
            idx_train = torch.arange(x.size(0))
            
        self.train()
        
        # Step 1: Optimize fairness layer
        self.zero_grad()
        z = self.GNN(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)
        
        if y.dim() > 1:
            y = y.squeeze(-1)
            
        self.F_loss = self.criterion_fairness(
            x[idx_train], torch.sigmoid(y[idx_train]), sensitive_attr[idx_train]
        )
        self.F_loss.backward()
        self.optimizer_F.step()
        
        # Step 2: Optimize GNN and classifier
        z = self.GNN(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)
        
        if y.dim() > 1:
            y = y.squeeze(-1)
            
        self.G_loss = self.criterion(y[idx_train], labels[idx_train])
        fairness_loss = self.criterion_fairness(x[idx_train], y[idx_train], sensitive_attr[idx_train])
        
        total_loss = self.G_loss - fairness_loss
        total_loss.backward()
        self.optimizer_G.step()
        
        return {
            'main_loss': self.G_loss.item(),
            'fairness_loss': self.F_loss.item(),
            'total_loss': total_loss.item()
        }


def create_fair_gnn_for_dsse(dim_nodes=8, dim_lines=6, dim_out=2, dim_hidden=64, 
                            num_layers=3, model_type="GAT", fairness_alpha=100):
    model = FairGNN_DSSE(
        dim_feat=dim_nodes,
        dim_dense=dim_hidden,
        dim_out=dim_out,
        num_layers=num_layers,
        edge_dim=dim_lines,
        model_type=model_type,
        heads=[4, 4, 1] if model_type == "GAT" else None,
        dropout=0.3,
        fairness_alpha=fairness_alpha,
        sensitive_classes=[0, 1]
    )
    
    model.setup_optimizers(lr_g=1e-3, lr_f=1e-9, weight_decay=1e-5)
    return model


def evaluate_fairness_metrics(model, data_loader, sensitive_attr_key='sensitive', device='cpu'):
    model.eval()
    all_predictions = []
    all_sensitive = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            num_nfeat = 8
            num_efeat = 6
            y_pred, _ = model(data.x[:, :num_nfeat], data.edge_index, data.edge_attr[:, :num_efeat])
            
            all_predictions.append(y_pred.cpu())
            if hasattr(data, sensitive_attr_key):
                all_sensitive.append(getattr(data, sensitive_attr_key).cpu())
    
    if not all_sensitive:
        return {}
    
    predictions = torch.cat(all_predictions, dim=0)
    sensitive = torch.cat(all_sensitive, dim=0)
    
    group_0_pred = predictions[sensitive == 0].mean()
    group_1_pred = predictions[sensitive == 1].mean()
    dp_difference = abs(group_0_pred - group_1_pred)
    
    return {
        'demographic_parity_difference': dp_difference.item(),
        'group_0_mean_prediction': group_0_pred.item(),
        'group_1_mean_prediction': group_1_pred.item()
    }
