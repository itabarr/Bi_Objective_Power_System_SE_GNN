import torch


def get_device():
    if torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu") 
    print(f"Using device: {device}")

    return device

def angular_distance(pred_angles, true_angles): 
    """ Calculate the shortest angular distance between angles wrapped to [-π, π]"""
    diff = pred_angles - true_angles
    diff = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
    return diff

def angular_mae(pred_angles, true_angles):
    """Calculate Mean Absolute Error for angles with proper wrapping."""
    diff = angular_distance(pred_angles, true_angles)
    return torch.mean(torch.abs(diff))

def angular_rmse(pred_angles, true_angles):
    """Calculate Root Mean Square Error for angles with proper wrapping."""
    diff = angular_distance(pred_angles, true_angles)
    return torch.sqrt(torch.mean(diff**2))