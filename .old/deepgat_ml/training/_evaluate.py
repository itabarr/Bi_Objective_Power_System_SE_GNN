import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanAbsoluteError
import matplotlib.pyplot as plt

# --- Global storage for metrics and figure objects ---
rmse_v_history = []
rmse_th_history = []

fig, ax1, ax2 = None, None, None
line_v, line_th = None, None


def init_live_plot():
    """Initialize live plots (called once, on first evaluate call with animation=True)."""
    global fig, ax1, ax2, line_v, line_th

    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Voltage RMSE plot
    line_v, = ax1.plot([], [], "o-", markersize=4, alpha=0.7, label="Voltage RMSE")
    ax1.set_ylabel("Voltage RMSE")
    ax1.set_title("Voltage RMSE per Epoch")
    ax1.grid(True)
    ax1.legend()

    # Angle RMSE plot
    line_th, = ax2.plot([], [], "o-", markersize=4, alpha=0.7, label="Angle RMSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Angle RMSE")
    ax2.set_title("Angle RMSE per Epoch")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    fig.show()
    fig.canvas.draw()


def update_live_plot():
    """Update both subplots with new metrics (raw values, dynamic y-axis)."""
    global fig, ax1, ax2, line_v, line_th

    epochs = list(range(1, len(rmse_v_history) + 1))

    # Update Voltage RMSE plot
    line_v.set_data(epochs, rmse_v_history)
    ax1.relim()
    ax1.autoscale_view()

    # Update Angle RMSE plot
    line_th.set_data(epochs, rmse_th_history)
    ax2.relim()
    ax2.autoscale_view()

    # Refresh figure
    fig.canvas.draw()
    fig.canvas.flush_events()


def evaluate_model(model, test_loader, **kwargs):
    # Temporary kwargs
    num_nfeat = kwargs.get('num_nfeat', 8)
    num_efeat = kwargs.get('num_efeat', 6)
    x_mean = kwargs.get('x_mean', torch.zeros(8))
    x_std = kwargs.get('x_std', torch.ones(8))
    epoch = kwargs.get('epoch', 0)
    epochs = kwargs.get('epochs', 1)
    avg_train_loss = kwargs.get('avg_train_loss', 0.0)
    animation = kwargs.get('animation', False)

    model.eval()
    pred = 0
    pred_th = 0
    pred_mae = 0
    pred_mae_th = 0

    with torch.no_grad():
        for data in test_loader:
            # Forward pass
            out = model(data.x[:, :num_nfeat], data.edge_index, data.edge_attr[:, :num_efeat])

            # Denormalize output
            out = torch.concat([out[:, 0:1]*x_std[:1] + x_mean[:1], out[:, 1:]], axis=1)
            out[:, 1:] *= (1. - data.x[:, 9:10])  # Enforce theta_slack = 0

            # Metrics
            pred += torch.sqrt(F.mse_loss(out[:, :1], data.y[:, :1]))
            pred_th += torch.sqrt(F.mse_loss(out[:, 1:], data.y[:, 1:]))

            mae = MeanAbsoluteError()
            pred_mae += mae(out[:, :1], data.y[:, :1])
            pred_mae_th += mae(out[:, 1:], data.y[:, 1:])

    # Average
    rmse_v = float((pred/len(test_loader)).detach().float().numpy())
    rmse_th = float((pred_th/len(test_loader)).detach().float().numpy())
    mae_v = float((pred_mae/len(test_loader)).detach().float().numpy())
    mae_th = float((pred_mae_th/len(test_loader)).detach().float().numpy())

    # Save metrics
    rmse_v_history.append(rmse_v)
    rmse_th_history.append(rmse_th)

    # Print log
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Voltage RMSE: {rmse_v:.6f} | "
        f"Voltage MAE: {mae_v:.6f} | "
        f"Angle RMSE: {rmse_th:.6f} | "
        f"Angle MAE: {mae_th:.6f}"
    )

    # 🔥 Initialize & update live plots if requested
    if animation:
        global fig
        if fig is None:  # first call
            init_live_plot()
        update_live_plot()
