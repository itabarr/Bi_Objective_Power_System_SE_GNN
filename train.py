import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError
import torch.optim as optim

from data._dsml_data import gsp_wls_edge, get_pflow

from data.power_system_dataset import PowerSystemDataset
from data.power_system_dataloader import PowerSystemDataLoader

from models.models_ref import GAT_DSSE
from models.models_GAT_CONV_NORM_DSSE import DeepGAT_DSSE
from models.models_GAT_V2_CONV_NORM_DSSE import GAT_NORM_DSSE

from loss import WLSLoss, PhysicalLoss, CombinedWLSPhysicalLoss

from utils import get_device 
from utils import angular_mae, angular_rmse
import live_plot

# GLOBAL CONFIG
LIVE_PLOT = True
DEVICE = get_device()

# PROBLEM DIMENSIONS
NUM_NODE_FEATURES = 8       # V, V_cov, Theta, Theta_cov, P, P_cov, Q, Q_cov
NUM_EDGE_FEATURES = 6       # PF, PF_cov, QF, QF_cov, B, G
NUM_OUTPUT_FEATURES = 2     # V, Theta

# MEASUREMENTS PENETRATION
NUM_NODE_MEASUREMENTS = 4   
NUM_EDGE_MEASUREMENTS = 2

# DATA PREPARATION
CASE = 'cigre14'
BATCH_SIZE = 64
SPLIT_COEF = 0.9

power_system_dataset = PowerSystemDataset(case = CASE , num_node_features = NUM_NODE_FEATURES, num_edge_features = NUM_EDGE_FEATURES,
                                         num_node_measurements = NUM_NODE_MEASUREMENTS, num_edge_measurements = NUM_EDGE_MEASUREMENTS)
print(power_system_dataset)

test_dataset , train_dataset = power_system_dataset.split_dataset(split_coef = SPLIT_COEF)

X_MEAN = power_system_dataset._x_mean
X_STD = power_system_dataset._x_std
PFLOW_MEAN = power_system_dataset._edge_features_mean
PFLOW_STD = power_system_dataset._edge_features_std

test_loader = PowerSystemDataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)
train_loader = PowerSystemDataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False)

# MODEL SETUP

# dim_hid, gnn_layers, heads, K, dropout and L to tune as wanted
hyperparameters = {
    'dim_nodes': NUM_NODE_FEATURES, 
    'dim_lines': NUM_EDGE_FEATURES, 
    'dim_out':   NUM_OUTPUT_FEATURES,                   
    'dim_hid': 32, 
    'gnn_layers': 8,
    'heads': 1,
}

# model_name = 'gat'
# model = GAT_DSSE(dim_feat= hyperparameters['dim_nodes'],
#                 dim_dense=hyperparameters['dim_hid'],
#                 dim_out=hyperparameters['dim_out'],
#                 heads=hyperparameters['heads'],
#                 num_layers=hyperparameters['gnn_layers'],
#                 edge_dim=hyperparameters['dim_lines'])


# model_name = 'lipchitz_gat'
model = GAT_NORM_DSSE(dim_feat= hyperparameters['dim_nodes'],
                dim_dense=hyperparameters['dim_hid'],
                dim_out=hyperparameters['dim_out'],
                heads=hyperparameters['heads'],
                num_layers=hyperparameters['gnn_layers'],
                edge_dim=hyperparameters['dim_lines'])

# model = DeepGAT_DSSE(dim_feat= hyperparameters['dim_nodes'],
#                     dim_dense=hyperparameters['dim_hid'],
#                     dim_out=hyperparameters['dim_out'],
#                     heads=hyperparameters['heads'],
#                     num_layers=hyperparameters['gnn_layers'],
#                     edge_dim=hyperparameters['dim_lines'],
#                     norm=hyperparameters['norm'],
#                     dropout=hyperparameters['dropout_rate'])
    

# OPTIMIZER SETUP
LR = 3e-3
WEIGHT_DECAY = 1e-5
optimizer = optim.Adamax(model.parameters(), lr = LR , weight_decay = WEIGHT_DECAY)

# LOSS FUNCTION SETUP
LAMBDA_WLS_VOLTAGE = 1e-8
LAMBDA_WLS_PHASE = 1e-4
LAMBDA_WLS_POWER_FLOW = 1e-8

wls_loss = WLSLoss(
    lambda_voltage=LAMBDA_WLS_VOLTAGE,
    lambda_phase=LAMBDA_WLS_PHASE,
    lambda_power_flow=LAMBDA_WLS_POWER_FLOW
)

LAMBDA_PHY_VOLTAGE = 1e-4
LAMBDA_PHY_ANGLE = 1e-4
LAMBDA_PHY_LOADING = 1e-4
LAMBDA_REGULARIZATION = 1e-4
physical_loss = PhysicalLoss(reg_weight=LAMBDA_REGULARIZATION)

LAMBDA_WLS = 1e0
LAMBDA_PHYSICAL = 1e0
loss_fn = CombinedWLSPhysicalLoss(
    wls_loss = wls_loss,
    physical_loss = physical_loss ,
    lambda_wls = LAMBDA_WLS ,
    lambda_physical = LAMBDA_PHYSICAL
)

mu_v = 1e-1
reg_coefs = {
'mu_v': mu_v,
'mu_theta': mu_v,
'lam_v': 1e-4,
'lam_p': 1e-8,
'lam_pf': 1e-6,
'lam_reg': 1e2
}

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


# TRAINING LOOP
model = model.to(DEVICE) 

metrics_list = [
            {'Train Loss': -1},
            {'RMSE V': -1},
            {'RMSE Theta': -1}, 
        ]

if LIVE_PLOT:
    live_plot.init_live_plot(metrics_list)

epochs = 1000
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    num_batches = len(train_loader)
    
    for data in train_loader:
        data = data.to(DEVICE)  # Move data to device
        
        optimizer.zero_grad()
        
        out = model(
            data.x,
            data.edge_index,
            data.edge_attr
        )
        
        _loss = loss_fn(
            input_data=data.x[:, :NUM_NODE_FEATURES],
            edge_input=data.edge_attr[:, :NUM_EDGE_FEATURES],
            output=out,
            x_mean=X_MEAN,
            x_std=X_STD,    
            edge_mean=PFLOW_MEAN,
            edge_std=PFLOW_STD,
            edge_index=data.edge_index,
            node_param=data.x[:, NUM_NODE_FEATURES:],
            edge_param=data.edge_attr[:, NUM_EDGE_FEATURES:]
        )

        # _loss = gsp_wls_edge(
        #     input = data.x,
        #     edge_input= data.edge_attr[:,:],
        #     output= out,
        #     x_mean= X_MEAN,
        #     x_std = X_STD,
        #     edge_mean = PFLOW_MEAN,
        #     edge_std = PFLOW_STD,
        #     edge_index = data.edge_index[:,:],
        #     reg_coefs = reg_coefs,
        #     num_samples = BATCH_SIZE,
        #     node_param = data.x,
        #     edge_param = data.edge_attr
        # )
        
        _loss.backward()
        optimizer.step()
        
        # total_train_loss_cmp +=  _loss_cmp.item()
        total_train_loss += _loss.item()
    
    # print(f"Loss: {_loss.item()} | Loss_cmp: {_loss_cmp.item()}")
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
        
        mae = MeanAbsoluteError().to(DEVICE)
        
        for data in test_loader:
            data = data.to(DEVICE)  # Move data to device
            
            out = model(
                data.x,
                data.edge_index,
                data.edge_attr
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
                node_param=data.x[:, NUM_NODE_FEATURES:],
                edge_param=data.edge_attr[:, NUM_EDGE_FEATURES:]
            )[:2]
            
            out_loading_lines, out_loading_trafos = get_pflow(
                out,
                data.edge_index,
                node_param=data.x[:, NUM_NODE_FEATURES:],
                edge_param=data.edge_attr[:, NUM_EDGE_FEATURES:]
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
        # print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} | RMSE V: {rmse_v:.6f} | RMSE Theta: {rmse_th:.6f}")

        # full metrics (+std)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} | RMSE V: {rmse_v:.6f} | RMSE Theta: {rmse_th:.6f} | RMSE Loading: {rmse_loading:.6f} | RMSE Loading Transformers: {rmse_loading_trafos:.6f}")
        print(f"MAE V: {mae_v:.6f} | MAE Theta: {mae_th:.6f} | MAE Loading: {mae_loading:.6f} | MAE Loading Transformers: {mae_loading_trafos:.6f}")
        print(f"Prop Std V: {prop_std_v:.6f} | Prop Std Theta: {prop_std_th:.6f}")

        
        metrics_list = [
            {'Train Loss': avg_train_loss},
            {'RMSE V': rmse_v},
            {'RMSE Theta': rmse_th}, 
        ]

        if LIVE_PLOT:
            live_plot.update_live_plot(epoch = epoch, metrics_list = metrics_list)

if LIVE_PLOT:
    live_plot.finalize_live_plot()


    
