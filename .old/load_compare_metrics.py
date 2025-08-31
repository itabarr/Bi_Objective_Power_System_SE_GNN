import torch
import numpy as np

new_network_file = r"C:\Users\Michael\Desktop\University\Masters\Deep Learning\Project\repo\Bi_Objective_Power_System_SE_GNN\validation_metrics_lipchitz_gat_stabilized.pt"
old_network_file = r"C:\Users\Michael\Desktop\University\Masters\Deep Learning\Project\repo\Bi_Objective_Power_System_SE_GNN\validation_metrics_gat_original.pt"

# Load the dictionary
new_metrics = torch.load(new_network_file)
old_metrics = torch.load(old_network_file)
old_average_metrics = {}
new_average_metrics = {}

# Print each metric and its list of values
for key, value in new_metrics.items():
    avg_val = np.mean(value)
    old_average_metrics[key] = avg_val
    print(f"New {key}: {avg_val}")

for key, value in old_metrics.items():
    avg_val = np.mean(value)
    new_average_metrics[key] = avg_val
    print(f"Old {key}: {avg_val}")


print('check')