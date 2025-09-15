import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError
from data._dsml_data import get_pflow
from data_processing import PowerSystemDataLoader
from model_FAIR_GAT_NORM_DSSE import FAIR_GAT_BILEVEL
from utils import get_device, angular_mae, angular_rmse
import live_plot

# -----------------------------
# Global CONFIG
# -----------------------------
LIVE_PLOT = True

# -----------------------------
# GENERAL PARAMETERS
# -----------------------------
phase_shift = True
num_nfeat = 8
num_efeat = 6
num_nmeas = 4 # TODO: Understand why this is not in use
num_emeas = 2 # TODO: Understand why this is not in use 
device = get_device()

# -----------------------------
# DATA LOADING
# -----------------------------
case = 'ober_sub'
batch_size = 64
split_coef = 0.9

data_loader = PowerSystemDataLoader(case=case, batch_size=batch_size, split_coef=split_coef)
train_loader, test_loader = data_loader.setup_complete_pipeline()

X_MEAN = data_loader.x_mean
X_STD = data_loader.x_std
PFLOW_MEAN = data_loader.pflow_mean
PFLOW_STD = data_loader.pflow_std

# -----------------------------
# MODEL SETUP
# -----------------------------
hyperparameters = {
    'dim_nodes': 8, 
    'dim_lines': 6, 
    'dim_out': 2, 
    'dim_hid': 32, 
    'gnn_layers': 8,
    'heads': 1,
    'dropout_rate': 0.3,
    'norm': None # This will lipschitznorm for some reason
}

model = FAIR_GAT_BILEVEL(
    dim_feat=hyperparameters['dim_nodes'],
    dim_dense=hyperparameters['dim_hid'],
    dim_out=hyperparameters['dim_out'],
    heads=hyperparameters['heads'],
    num_layers=hyperparameters['gnn_layers'],
    edge_dim=hyperparameters['dim_lines'],
    fairness_alpha=1.0,
    lr_g=1e-3,
    lr_f=1e-2,
    weight_decay=1e-5,
    lipschitz_norm=hyperparameters['norm'],
    dropout=hyperparameters['dropout_rate']
)
model = model.to(device)

# -----------------------------
# METRICS TRACKING
# -----------------------------
rmse_v_list, mae_v_list = [], []
rmse_th_list, mae_th_list = [], []
rmse_loading_list, mae_loading_list = [], []
rmse_loading_trafos_list, mae_loading_trafos_list = [], []
prop_std_v_list, prop_std_th_list = [], []

metrics_list = [
    {'Train Loss': -1},
    {'RMSE V': -1},
    {'RMSE Theta': -1}, 
]

if LIVE_PLOT:
    live_plot.init_live_plot(metrics_list)

# -----------------------------
# TRAINING LOOP
# -----------------------------
epochs = 15
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    num_train_batches = len(train_loader)

    for data in train_loader:
        data = data.to(device)
        out = model.optimize_step(
            x=data.x[:, :num_nfeat],
            edge_index=data.edge_index,
            edge_attr=data.edge_attr[:, :num_efeat],
            input_data=data.x[:, :num_nfeat],
            edge_input=data.edge_attr[:, :num_efeat], # Why is this the same as edge input and edge attr?
            x_mean=X_MEAN.to(device),
            x_std=X_STD.to(device),
            edge_mean=PFLOW_MEAN.to(device),
            edge_std=PFLOW_STD.to(device),
            node_param=data.x[:, num_nfeat:],
            edge_param=data.edge_attr[:, num_efeat:] 
        )
        total_train_loss += out['total_loss']
        # _loss_cmp = gsp_wls_edge(input=data.x[:,:num_nfeat], edge_input=data.edge_attr[:,:num_efeat],
        #  output= out, x_mean=X_MEAN, x_std=X_STD, edge_mean = PFLOW_MEAN, edge_std = PFLOW_STD,
        #  edge_index=data.edge_index, reg_coefs = reg_coefs,num_samples=data.batch[-1]+1,
        #  node_param=data.x[:,num_nfeat:], edge_param = data.edge_attr[:,num_efeat:])

    avg_train_loss = total_train_loss / num_train_batches

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        total_rmse_v = total_mae_v = 0.0
        total_rmse_th = total_mae_th = 0.0

        num_test_batches = len(test_loader)
        mae_metric = MeanAbsoluteError().to(device)

        for data in test_loader:
            data = data.to(device)
            y_pred = model(data.x[:, :num_nfeat], data.edge_index, data.edge_attr[:, :num_efeat])
            y_pred_denorm = torch.cat([y_pred[:, 0:1] * X_STD[:1] + X_MEAN[:1], y_pred[:, 1:]], dim=1)
            y_pred_denorm[:, 1:] *= (1. - data.x[:, 9:10])

            # Voltage
            total_rmse_v += torch.sqrt(F.mse_loss(y_pred_denorm[:, :1], data.y[:, :1])).item()
            total_mae_v += mae_metric(y_pred_denorm[:, :1], data.y[:, :1]).item()

            # Theta
            total_rmse_th += angular_rmse(y_pred_denorm[:, 1:], data.y[:, 1:]).item()
            total_mae_th += angular_mae(y_pred_denorm[:, 1:], data.y[:, 1:]).item()

        # Average metrics
        rmse_v = total_rmse_v / num_test_batches
        mae_v = total_mae_v / num_test_batches
        rmse_th = total_rmse_th / num_test_batches
        mae_th = total_mae_th / num_test_batches

        # Append for logging
        rmse_v_list.append(rmse_v)
        mae_v_list.append(mae_v)
        rmse_th_list.append(rmse_th)
        mae_th_list.append(mae_th)

    # Print
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} | RMSE V: {rmse_v:.6f} | RMSE Theta: {rmse_th:.6f}")

    metrics_list = [
        {'Train Loss': avg_train_loss},
        {'RMSE V': rmse_v},
        {'RMSE Theta': rmse_th}, 
    ]
    if LIVE_PLOT:
        live_plot.update_live_plot(epoch=epoch, metrics_list=metrics_list)

if LIVE_PLOT:
    live_plot.finalize_live_plot()
