import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError
import torch.optim as optim

from data._dsml_data import gsp_wls_edge, get_pflow, angular_distance

from data_processing import PowerSystemDataLoader
from ref_models import GAT_DSSE
from models import DeepGAT_DSSE
from utils import get_device
import live_plot



def angular_mae(pred_angles, true_angles):
    """Calculate Mean Absolute Error for angles with proper wrapping."""
    diff = angular_distance(pred_angles, true_angles)
    return torch.mean(torch.abs(diff))

def angular_rmse(pred_angles, true_angles):
    """Calculate Root Mean Square Error for angles with proper wrapping."""
    diff = angular_distance(pred_angles, true_angles)
    return torch.sqrt(torch.mean(diff**2))

# GENERAL PARAMETERS
phase_shift = True
num_nfeat = 8
num_efeat = 6
num_nmeas = 4
num_emeas = 2
device = get_device()

# DATA LOADING

case = 'cigre14'
batch_size = 64
split_coef = 0.9

data_loader = PowerSystemDataLoader(case='cigre14', batch_size=64, split_coef=0.9)
train_loader, test_loader = data_loader.setup_complete_pipeline()

X_MEAN = data_loader.x_mean
X_STD = data_loader.x_std
PFLOW_MEAN = data_loader.pflow_mean
PFLOW_STD = data_loader.pflow_std

# MODEL SETUP

# dim_hid, gnn_layers, heads, K, dropout and L to tune as wanted
hyperparameters = {
    'dim_nodes': 8, # V, Theta, P, Q and Covs
    'dim_lines': 6, # P, Q and their Cov,B,G
    'dim_out': 2, # V, Theta
    'dim_hid': 32, 
    'gnn_layers': 8,
    'heads': 1,
    'K': 2,
    'dropout_rate': 0.3,
    'L': 5,
    'norm': 'lipschitznorm'
}

model_name = 'gat'
model = GAT_DSSE(dim_feat= hyperparameters['dim_nodes'],
                dim_dense=hyperparameters['dim_hid'],
                dim_out=hyperparameters['dim_out'],
                heads=hyperparameters['heads'],
                num_layers=hyperparameters['gnn_layers'],
                edge_dim=hyperparameters['dim_lines'])


# model_name = 'lipchitz_gat'
# model = DeepGAT_DSSE(dim_feat=hyperparameters['dim_nodes'],
#                     dim_dense=hyperparameters['dim_hid'],
#                     dim_out=hyperparameters['dim_out'],
#                     heads=hyperparameters['heads'],
#                     num_layers=hyperparameters['gnn_layers'],
#                     edge_dim=hyperparameters['dim_lines'],
#                     norm=hyperparameters['norm'],
#                     dropout=hyperparameters['dropout_rate'])

# Generate the optimizers.
lr = 3e-3
optimizer = getattr(optim, 'Adamax')(model.parameters(), lr=lr)


# coefficients for training loss, to tune to find good training conditions/balance
mu_v = 1e-1
reg_coefs = {
'mu_v': mu_v,
'mu_theta': mu_v,
'lam_v': 1e-4,
'lam_p': 1e-8,
'lam_pf': 1e-6,
'lam_reg': 1e2
}

train_list = []
thresh = 0

# Initialize lists for training and testing metrics
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


# TRAINING LOOP
model = model.to(device) 

metrics_list = [
            {'Train Loss': -1},
            {'MAE V': -1},
            {'MAE Theta': -1}, 
        ]

live_plot.init_live_plot(metrics_list)

epochs = 1000
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    num_batches = len(train_loader)
    
    for data in train_loader:
        data = data.to(device)  # Move data to device
        
        optimizer.zero_grad()
        
        out = model(
            data.x[:, :num_nfeat],
            data.edge_index,
            data.edge_attr[:, :num_efeat]
        )
        
        loss = gsp_wls_edge(
            input=data.x[:, :num_nfeat],
            edge_input=data.edge_attr[:, :num_efeat],
            output=out,
            x_mean=X_MEAN,
            x_std=X_STD,
            edge_mean=PFLOW_MEAN,
            edge_std=PFLOW_STD,
            edge_index=data.edge_index,
            reg_coefs=reg_coefs,
            num_samples=data.batch[-1] + 1,
            node_param=data.x[:, num_nfeat:],
            edge_param=data.edge_attr[:, num_efeat:]
        )
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        total_rmse_v = 0.0
        total_mae_v = 0.0
        total_rmse_th = 0.0
        total_mae_th = 0.0
        total_rmse_loading = 0.0
        total_mae_loading = 0.0
        total_rmse_loading_trafos = 0.0
        total_mae_loading_trafos = 0.0
        total_prop_std_v = 0.0
        total_prop_std_th = 0.0
        
        num_test_batches = len(test_loader)
        
        mae = MeanAbsoluteError().to(device)
        
        for data in test_loader:
            data = data.to(device)  # Move data to device
            
            out = model(
                data.x[:, :num_nfeat],
                data.edge_index,
                data.edge_attr[:, :num_efeat]
            )
            out = torch.cat([out[:, 0:1] * X_STD[:1] + X_MEAN[:1], out[:, 1:]], dim=1)
            out[:, 1:] *= (1. - data.x[:, 9:10])
            
            # Voltage magnitude
            total_rmse_v += torch.sqrt(F.mse_loss(out[:, :1], data.y[:, :1])).item()
            total_mae_v += mae(out[:, :1], data.y[:, :1]).item()
            
            # Voltage angle - using proper angular distance
            total_rmse_th += angular_rmse(out[:, 1:], data.y[:, 1:]).item()
            total_mae_th += angular_mae(out[:, 1:], data.y[:, 1:]).item()
            
            # Power flow calculations
            true_loading_lines, true_loading_trafos = get_pflow(
                data.y,
                data.edge_index,
                node_param=data.x[:, num_nfeat:],
                edge_param=data.edge_attr[:, num_efeat:]
            )[:2]
            
            out_loading_lines, out_loading_trafos = get_pflow(
                out,
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
            
            # Proportion of std
            std_ratios = (out.std(dim=0) / data.y.std(dim=0)) * 100
            total_prop_std_v += std_ratios[0].item()
            total_prop_std_th += std_ratios[1].item()
        
        # Average metrics
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
        
        # Append to lists
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
        

        # Print training progress every 10 epochs
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} | MAE V: {mae_v:.6f} | MAE Theta: {mae_th:.6f}")
        
        metrics_list = [
            {'Train Loss': avg_train_loss},
            {'MAE V': mae_v},
            {'MAE Theta': mae_th}, 
        ]
        live_plot.update_live_plot(epoch = epoch, metrics_list = metrics_list)
        
live_plot.finalize_live_plot()


    
