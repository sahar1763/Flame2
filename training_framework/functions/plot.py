import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

FitResult = namedtuple('FitResult', ['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])


def plot_fit(fit_res: FitResult, fig=None, log_loss=False, legend=None):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :return: The figure.
    """
    if fig is None:
        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(16, 10), sharex="col", sharey=False
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(["train", "val", "test"], ["loss", "acc"])
    for idx, (traintest, lossacc) in enumerate(p):
        ax = axes[idx]
        attr = f"{traintest}_{lossacc}"
        data = getattr(fit_res, attr)

         # Ensure all data is detached and converted to NumPy arrays
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):  # Handle lists of tensors
            data = np.array([d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data])
        elif not isinstance(data, np.ndarray):  # Raise error for unsupported types
            raise TypeError(f"Unsupported data type for {attr}: {type(data)}")

        
        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        ax.set_title(attr)
        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")
        if legend:
            ax.legend()
        ax.grid(True)

    return fig, axes

