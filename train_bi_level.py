import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError
from data._dsml_data import get_pflow
from data_processing import PowerSystemDataLoader
from model_FAIR_GAT_NORM_DSSE import FAIR_GAT_BILEVEL
from utils import get_device, angular_mae, angular_rmse
import live_plot
import matplotlib.pyplot as plt

# -----------------------------
# Global CONFIG
# -----------------------------
LIVE_PLOT = False

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
case = 'ober_sub' # options = ['cigre14', 'cigre14_reswitched', 'ober_sub']
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
    lr_g=1e-4,
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
epochs = 10
# METRICS TRACKING
train_list = []
thresh = 0
train_losses = []
rmse_v_list = []
mae_v_list = []
rmse_th_list = []
mae_th_list = []
rmse_loading_list = []
mae_loading_list = []
rmse_loading_trafos_list = []
mae_loading_trafos_list = []
prop_std_v_list = []
prop_std_th_list = []

for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    num_train_batches = len(train_loader)

    for idx, data in enumerate(train_loader):
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
            edge_param=data.edge_attr[:, num_efeat:],
            print_loss=True if idx%10==1 else False
        )
        # a = out['total_loss']
        # print(f'training loss is {a}')
        total_train_loss += out['total_loss']
        # _loss_cmp = gsp_wls_edge(input=data.x[:,:num_nfeat], edge_input=data.edge_attr[:,:num_efeat],
        #  output= out, x_mean=X_MEAN, x_std=X_STD, edge_mean = PFLOW_MEAN, edge_std = PFLOW_STD,
        #  edge_index=data.edge_index, reg_coefs = reg_coefs,num_samples=data.batch[-1]+1,
        #  node_param=data.x[:,num_nfeat:], edge_param = data.edge_attr[:,num_efeat:])

    avg_train_loss = total_train_loss / num_train_batches

    # Evaluation phase

    model.eval()
    mae = MeanAbsoluteError().to(device)
    with torch.no_grad():
        total_rmse_v = total_mae_v = 0.0
        total_rmse_th = total_mae_th = 0.0
        total_rmse_loading = 0.0
        total_mae_loading = 0.0
        total_rmse_loading_trafos = 0.0
        total_mae_loading_trafos = 0.0
        total_prop_std_v = 0.0
        total_prop_std_th = 0.0

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
        
            # Power flow calculations
            true_loading_lines, true_loading_trafos = get_pflow(
                data.y,
                data.edge_index,
                node_param=data.x[:, num_nfeat:],
                edge_param=data.edge_attr[:, num_efeat:]
            )[:2]
            
            out_loading_lines, out_loading_trafos = get_pflow(
                y_pred_denorm,
                data.edge_index,
                node_param=data.x[:, num_nfeat:],
                edge_param=data.edge_attr[:, num_efeat:]
            )[:2]
            
            # Filter non-zero for lines
            true_loading_lines_nonzero = true_loading_lines[true_loading_lines.nonzero()]
            out_loading_lines = out_loading_lines[true_loading_lines.nonzero()]
            
            total_rmse_loading += torch.sqrt(F.mse_loss(out_loading_lines, true_loading_lines_nonzero)).item()
            total_mae_loading += mae(out_loading_lines, true_loading_lines_nonzero).item()
            
            # Filter non-zero for trafos
            true_loading_trafos_nonzero = true_loading_trafos[true_loading_trafos.nonzero()]
            out_loading_trafos = out_loading_trafos[true_loading_trafos.nonzero()]
            
            total_rmse_loading_trafos += torch.sqrt(F.mse_loss(out_loading_trafos, true_loading_trafos_nonzero)).item()
            total_mae_loading_trafos += mae(out_loading_trafos, true_loading_trafos_nonzero).item()

            std_ratios = (y_pred_denorm.std(dim=0) / data.y.std(dim=0)) * 100
            total_prop_std_v += std_ratios[0].item()
            total_prop_std_th += std_ratios[1].item()

        # Average metrics per epoch
        rmse_v = total_rmse_v / num_test_batches
        mae_v = total_mae_v / num_test_batches
        rmse_th = total_rmse_th / num_test_batches
        mae_th = total_mae_th / num_test_batches
        rmse_loading = total_rmse_loading / num_test_batches
        mae_loading = total_mae_loading / num_test_batches
        rmse_loading_trafos = total_rmse_loading_trafos / num_test_batches
        mae_loading_trafos = total_mae_loading_trafos / num_test_batches
        prop_std_v = total_prop_std_v / num_test_batches
        prop_std_th = total_prop_std_th / num_test_batches

        # Append for logging
        rmse_v_list.append(rmse_v)
        mae_v_list.append(mae_v)
        rmse_th_list.append(rmse_th)
        mae_th_list.append(mae_th)
        rmse_loading_list.append(rmse_loading)
        mae_loading_list.append(mae_loading)
        rmse_loading_trafos_list.append(rmse_loading_trafos)
        mae_loading_trafos_list.append(mae_loading_trafos)
        prop_std_v_list.append(prop_std_v)
        prop_std_th_list.append(prop_std_th)

    # Print
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} | RMSE V: {rmse_v:.6f} | RMSE Theta: {rmse_th:.6f} | RMSE Loading Lines: {rmse_loading:.6f}| RMSE Loading Trafos: {rmse_loading_trafos:.6f} | prop STD Voltage: {prop_std_v} | prop STD Theta: {prop_std_th}")

    metrics_list = [
        {'Train Loss': avg_train_loss},
        {'RMSE V': rmse_v},
        {'RMSE Theta': rmse_th}, 
    ]
    if LIVE_PLOT:
        live_plot.update_live_plot(epoch=epoch, metrics_list=metrics_list)

if LIVE_PLOT:
    live_plot.finalize_live_plot()

# plot final results and save:
metrics = [
    ("RMSE Voltage", rmse_v_list),
    ("MAE Voltage", mae_v_list),
    ("RMSE Theta", rmse_th_list),
    ("MAE Theta", mae_th_list),
    ("RMSE Loading", rmse_loading_list),
    ("MAE Loading", mae_loading_list),
    ("RMSE Loading (Trafos)", rmse_loading_trafos_list),
    ("MAE Loading (Trafos)", mae_loading_trafos_list),
    ("Prop STD Voltage", prop_std_v_list),
    ("Prop STD Theta", prop_std_th_list),
]

# Number of metrics
n_metrics = len(metrics)
n_cols = 2
n_rows = (n_metrics + n_cols - 1) // n_cols

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
axes = axes.flatten()

# Plot each metric
for i, (title, data) in enumerate(metrics):
    axes[i].plot(data, marker='o')
    axes[i].set_title(title)
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Value')
    axes[i].grid(True)

# Hide any unused subplots if n_metrics is odd
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(f"metrics_plots_case_{case}_epochs_{epochs}.png", dpi=300)
plt.show()
plt.close()

print("Saved figure as metrics_plots.png")

