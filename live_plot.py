import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Globals
fig = None
axs = None
lines = []
metric_keys = []
history = []
epochs_hist = []

def init_live_plot(metrics_list):
    """
    Initialize live plots.
    
    Args:
        metrics_list (list[dict]): Structure defining subplots and their labels.
                                   Only keys are used for initialization.
    """
    global fig, axs, lines, metric_keys, history, epochs_hist
    
    plt.ion()
    num_plots = len(metrics_list)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3*num_plots), sharex=True)
    
    if num_plots == 1:
        axs = [axs]  # ensure iterable
    
    lines = []
    metric_keys = []
    history = [defaultdict(list) for _ in range(num_plots)]
    epochs_hist = []
    
    for ax, metrics in zip(axs, metrics_list):
        subplot_lines = {}
        subplot_keys = list(metrics.keys())
        
        for label in subplot_keys:
            line, = ax.plot([], [], label=label)
            subplot_lines[label] = line
        
        ax.set_title(" / ".join(subplot_keys))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        
        lines.append(subplot_lines)
        metric_keys.append(subplot_keys)
    
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


def update_live_plot(epoch, metrics_list):
    """
    Update plots with one new epoch of data.
    
    Args:
        epoch (int): New epoch index.
        metrics_list (list[dict]): Each dict corresponds to one subplot.
                                   Keys = label names, values = scalar values (not arrays).
    """
    global epochs_hist, history
    
    if fig is None:
        raise ValueError("Live plot not initialized. Call init_live_plot first.")
    
    # Save new data
    epochs_hist.append(epoch)
    for subplot_idx, metrics in enumerate(metrics_list):
        for label, value in metrics.items():
            history[subplot_idx][label].append(value)
    
    # Update plots with raw values (skip first 50 epochs)
    skip_epochs = 50

    # Only update plots if we have more than skip_epochs
    if len(epochs_hist) > skip_epochs:
        display_epochs = epochs_hist[skip_epochs:]

        for ax, subplot_lines, hist in zip(axs, lines, history):
            for label, values in hist.items():
                values = np.array(values)
                display_values = values[skip_epochs:]  # Skip first 50 values
                subplot_lines[label].set_data(display_epochs, display_values)

            ax.relim()               # Recalculate limits
            ax.autoscale_view(True, True, True)  # Autoscale both axes
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.pause(0.01)


def finalize_live_plot():
    plt.ioff()
