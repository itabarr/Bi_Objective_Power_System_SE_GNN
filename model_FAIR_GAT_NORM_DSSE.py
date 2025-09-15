import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from models_GAT_NORM_DSSE import LipschitzNorm
from loss import PhysicalLoss , WLSLoss
from models_GAT_NORM_DSSE import GAT_NORM_DSSE

class FAIR_GAT_NORM_DSSE(nn.Module):
    """
    Fair Graph Neural Network for Distribution System State Estimation.

    This class integrates fairness constraints with power system state estimation
    using the GAT_NORM_DSSE backbone model with Lipschitz normalization.
    """

    def __init__(self, dim_feat: int, dim_dense: int, dim_out: int, num_layers: int,
                 edge_dim: int, heads: int = 1, concat: bool = True, slope: float = 0.2,
                 self_loops: bool = True, dropout: float = 0.0, nonlin: str = 'leaky_relu',
                 lipschitz_norm=None, fairness_alpha: float = 100.0,
                 sensitive_classes=[0, 1], lr_g: float = 1e-3, lr_f: float = 1e-9,
                 weight_decay: float = 1e-5):

        super(FAIR_GAT_NORM_DSSE, self).__init__()

        # Main GNN backbone
        self.model = GAT_NORM_DSSE(
            dim_feat=dim_feat,
            dim_dense=dim_dense,
            dim_out=dim_dense,  # Output hidden features for fairness processing
            num_layers=num_layers,
            edge_dim=edge_dim,
            heads=heads,
            concat=concat,
            slope=slope,
            self_loops=self_loops,
            dropout=dropout,
            nonlin=nonlin,
            lipschitz_norm=lipschitz_norm
        )

        # The fairness layer from the original paper is unnecessary in our case, replaced with Identity for compatibility
        self.fairness_layer = nn.Identity()

        # Final output layer for state estimation (V, Theta)
        # TODO: make sure this is unnecessary and remove
        self.classifier = nn.Linear(dim_dense, dim_out)

        # Loss functions
        self.criterion = WLSLoss()
        self.criterion_fairness = PhysicalLoss()

        # Optimizers for adversarial training
        G_params = list(self.model.parameters()) + list(self.classifier.parameters())
        F_params = list(self.fairness_layer.parameters())

        self.optimizer_G = torch.optim.Adam(G_params, lr=lr_g, weight_decay=weight_decay)
        self.optimizer_F = torch.optim.Adam(F_params, lr=lr_f, weight_decay=weight_decay)

        # TODO: this scheduler probably wont fit our network - integrate the original DSSE one perhaps
        self.scheduler_G = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_F = ExponentialLR(self.optimizer_F, gamma=0.99)

        # Loss tracking
        self.G_loss = 0
        self.F_loss = 0

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the network.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features

        Returns:
            Tuple of (predictions, hidden_features)
        """
        z = self.model(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)
        return y, z

    def optimize_step(self, x, edge_index, edge_attr, labels, sensitive_attr,
                     idx_train=None):
        """
        Perform one optimization step with to accuracy loss and constraint loss.

        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            labels: Target labels (V, Theta)
            sensitive_attr: Sensitive attributes for fairness
            idx_train: Training indices (if None, use all)

        Returns:
            Dictionary with loss values
        """
        if idx_train is None:
            idx_train = torch.arange(x.size(0))

        self.train()

        # Step 1: Optimize constraint layer 
        self.optimizer_F.zero_grad()
        z = self.model(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)

        # Fairness loss (maximize fairness violation for adversarial training)
        self.F_loss = self.criterion_fairness(x[idx_train], torch.sigmoid(y[idx_train].mean(dim=-1)), sensitive_attr[idx_train]) # TODO: change this
        self.F_loss.backward()
        self.optimizer_F.step()

        # Step 2: Optimize GNN and classifier (main task + fairness constraint)
        self.optimizer_G.zero_grad()
        z = self.model(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)

        # Main task loss (state estimation)
        self.G_loss = self.criterion(y[idx_train], labels[idx_train]) # TODO: Change this

        # Fairness constraint (minimize fairness violation)
        fairness_loss = self.criterion_fairness(
            x[idx_train], y[idx_train].mean(dim=-1), sensitive_attr[idx_train]
        )

        # Combined loss (main task - fairness for adversarial training)
        total_loss = self.G_loss - fairness_loss
        total_loss.backward()
        self.optimizer_G.step()

        return {
            'main_loss': self.G_loss.item(),
            'fairness_loss': self.F_loss.item(),
            'total_loss': total_loss.item()
        }

class PhysicalFollower(nn.Module):
    """
    Follower loss module with learnable multipliers for powerflow constraints.
    """
    def __init__(self):
        super().__init__()
        # multipliers for voltage, angle, loading constraints
        self.reg_coefs = nn.Parameter(torch.ones(1))
        # wrapped PhysicalLoss
        self.phys_loss = PhysicalLoss()

    def forward(self, output, edge_index, node_param, edge_param, x_mean, x_std):
        loss_dict = self.phys_loss(output, edge_index, node_param, edge_param, x_mean, x_std)
        total = loss_dict['total']
        return total, loss_dict

class FAIR_GAT_BILEVEL(nn.Module):
    """
    Bi-level GAT for DSSE.
    Leader: odd layers + final projection -> minimize WLSLoss
    Follower: even layers -> minimize PhysicalLoss
    """
    def __init__(self,
                 dim_feat, dim_dense, dim_out, num_layers, edge_dim,
                 heads=1, concat=True, slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', lipschitz_norm=None,
                 fairness_alpha=100.0, lr_g=1e-3, lr_f=1e-2, weight_decay=1e-5):

        super().__init__()

        # GAT backbone (split layers)
        self.model = GAT_NORM_DSSE(
            dim_feat=dim_feat, dim_dense=dim_dense, dim_out=dim_out,
            num_layers=num_layers, edge_dim=edge_dim, heads=heads,
            concat=concat, slope=slope, self_loops=self_loops,
            dropout=dropout, nonlin=nonlin, lipschitz_norm=lipschitz_norm
        )

        # Losses
        self.criterion = WLSLoss()
        self.physical_loss = PhysicalLoss()
        self.fairness_alpha = fairness_alpha

        # Parameter groups: odd (leader) vs even (follower)
        leader_params = [p for i, l in enumerate(self.model.layers) if i % 2 == 0 for p in l.parameters()]
        leader_params += list(self.model.projection.parameters())
        follower_params = [p for i, l in enumerate(self.model.layers) if i % 2 == 1 for p in l.parameters()]

        # Optimizers
        self.optimizer_G = torch.optim.Adam(leader_params, lr=lr_g, weight_decay=weight_decay)
        self.optimizer_F = torch.optim.Adam(follower_params, lr=lr_f, weight_decay=weight_decay)

        self.scheduler_G = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_F = ExponentialLR(self.optimizer_F, gamma=0.99)

    def forward(self, x, edge_index, edge_attr):
        return self.model(x, edge_index, edge_attr)

    def optimize_step(self, x, edge_index, edge_attr,
                      input_data, edge_input, x_mean, x_std, edge_mean, edge_std,
                      node_param, edge_param, idx_train=None, k_follower=1):

        if idx_train is None:
            idx_train = torch.arange(x.size(0), device=x.device)

        self.train()

        # ---------------------
        # 1) Follower (even layers)
        # ---------------------
        for _ in range(k_follower):
            self.optimizer_F.zero_grad()
            y_pred = self.forward(x, edge_index, edge_attr)
            phys_dict = self.physical_loss(
                y_pred, edge_index, node_param, edge_param, x_mean, x_std
            )
            follower_loss = phys_dict['total']
            follower_loss.backward()
            self.optimizer_F.step()

        # Recompute follower loss without grad
        with torch.no_grad():
            y_pred = self.forward(x, edge_index, edge_attr)
            follower_loss = self.physical_loss(
                y_pred, edge_index, node_param, edge_param, x_mean, x_std
            )['total'] * self.fairness_alpha

        # ---------------------
        # 2) Leader (odd layers + projection)
        # ---------------------
        self.optimizer_G.zero_grad()
        y_pred = self.forward(x, edge_index, edge_attr)

        wls_loss = self.criterion(
            input_data=input_data, edge_input=edge_input, output=y_pred,
            x_mean=x_mean, x_std=x_std, edge_mean=edge_mean, edge_std=edge_std,
            edge_index=edge_index, node_param=node_param, edge_param=edge_param
        )

        total_loss = wls_loss + follower_loss
        total_loss.backward()
        self.optimizer_G.step()

        return {
            'wls_loss': wls_loss.item(),
            'follower_loss': follower_loss.item(),
            'total_loss': total_loss.item()
        }