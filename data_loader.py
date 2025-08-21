import pandas as pd
import pandapower as pp
import numpy as np
from data._dsml_data import data_from_pickles
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random

from torchmetrics.regression import MeanAbsoluteError
import torch.optim as optim

phase_shift = True
num_nfeat = 8
num_efeat = 6
num_nmeas = 4
num_emeas = 2


case = 'cigre14'
folder = 'data/'+case+'/'
batch_size = 64

# Set the measurement indices for each grid
if 'cigre' in case:
    meas_v = np.array([0,1,12,7,11,14])
    meas_pflow = np.array([0,10])
else:
    meas_v = np.array([35,16,52,47,6,48,59,27,37,56])
    meas_pflow = np.array([40,43,11,21,54,57])

# get data input
dataset, x_mean, x_std, pflow_mean, pflow_std = data_from_pickles(folder, num_nfeat, num_efeat, num_nmeas, num_emeas, meas_v, meas_pflow) # data is made of [nodes, edges, edge_features, labels/targets]
random.shuffle(dataset)

split_coef = 0.9
train_dataset = dataset[0:int(split_coef*len(dataset))]
test_dataset = dataset[int(split_coef*len(dataset)):]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)









