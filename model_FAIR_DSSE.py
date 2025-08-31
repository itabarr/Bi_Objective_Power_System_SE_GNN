import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from loss import PhysicalLoss , WLSLoss
from models_GAT_V2_CONV_NORM_DSSE import GAT_NORM_DSSE

class ConstraintLoss(nn.Module):
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
        sensitive = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons


class DemographicParityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity
        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
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
            idx_true = sensitive == v  # torch.bool
            expected_values_list.append(out[idx_true].mean())
        expected_values_list.append(out.mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y=None):
        return super(DemographicParityLoss, self).forward(X, out, sensitive)


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

        # Fairness layer for adversarial training
        self.fairness_layer = nn.Sequential(
            nn.Linear(dim_dense, dim_dense),
            nn.ReLU()
        )

        # Final output layer for state estimation (V, Theta)
        self.classifier = nn.Linear(dim_dense, dim_out)

        # Loss functions
        self.criterion = WLSLoss()
        self.criterion_fairness = PhysicalLoss()

        # Optimizers for adversarial training
        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        F_params = list(self.fairness_layer.parameters())

        self.optimizer_G = torch.optim.Adam(G_params, lr=lr_g, weight_decay=weight_decay)
        self.optimizer_F = torch.optim.Adam(F_params, lr=lr_f, weight_decay=weight_decay)

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
        Perform one optimization step with fairness constraints.

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

        # Step 1: Optimize fairness layer (adversarial)
        self.optimizer_F.zero_grad()
        z = self.model(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)

        # Fairness loss (maximize fairness violation for adversarial training)
        self.F_loss = self.criterion_fairness(
            x[idx_train], torch.sigmoid(y[idx_train].mean(dim=-1)), sensitive_attr[idx_train]
        )
        self.F_loss.backward()
        self.optimizer_F.step()

        # Step 2: Optimize GNN and classifier (main task + fairness constraint)
        self.optimizer_G.zero_grad()
        z = self.model(x, edge_index, edge_attr)
        z = self.fairness_layer(z)
        y = self.classifier(z)

        # Main task loss (state estimation)
        self.G_loss = self.criterion(y[idx_train], labels[idx_train])

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

