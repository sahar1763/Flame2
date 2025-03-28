import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
from collections import namedtuple

FitResult = namedtuple('FitResult', ['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])


def tensors_as_images(
    tensors, nrows=1, figsize=(8, 8), titles=[], wspace=0.1, hspace=0.2, cmap=None
):
    """
    Plots a sequence of pytorch tensors as images.

    :param tensors: A sequence of pytorch tensors, should have shape CxWxH
    """
    assert nrows > 0

    num_tensors = len(tensors)

    ncols = math.ceil(num_tensors / nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        gridspec_kw=dict(wspace=wspace, hspace=hspace),
        subplot_kw=dict(yticks=[], xticks=[]),
    )
    axes_flat = axes.reshape(-1)

    # Plot each tensor
    for i in range(num_tensors):
        ax = axes_flat[i]

        image_tensor = tensors[i]
        assert image_tensor.dim() == 3  # Make sure shape is CxWxH

        image = image_tensor.numpy()
        image = image.transpose(1, 2, 0)
        image = image.squeeze()  # remove singleton dimensions if any exist

        # Scale to range 0..1
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)

        ax.imshow(image, cmap=cmap)

        if len(titles) > i and titles[i] is not None:
            ax.set_title(titles[i])

    # If there are more axes than tensors, remove their frames
    for j in range(num_tensors, len(axes_flat)):
        axes_flat[j].axis("off")

    return fig, axes


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

