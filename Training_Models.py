import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf
import torch.nn
import torch.nn.functional


seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})

from Models_Ours import CNN, ResNet18Custom

cnn_params = [
    dict(
        in_size=(3,254,254), out_classes=2,
        channels=[32]*4, pool_every=2, hidden_dims=[100]*2,
        conv_params=dict(kernel_size=3, stride=1, padding=1),
        activation_type='relu', activation_params=dict(),
        pooling_type='max', pooling_params=dict(kernel_size=2),
    )
]

net = CNN(**cnn_params[0])
print(net)


# Usage Example:
# Create the model for 10 output classes, training the last two layers + the added FC layer.
model = ResNet18Custom(out_classes=2, freeze_until=2)

# Print trainable parameters for verification
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")


#####

from experiments import load_experiment, cnn_experiment, ResNet_experiment
from plot import plot_fit

# Test experiment1 implementation on a few data samples and with a small model
cnn_experiment(
    'test_run', seed=seed, bs_train=128, bs_test=32,  batches=100, epochs=100, early_stopping=6,
    filters_per_layer=[32], layers_per_block=4, pool_every=4, hidden_dims=[100],
    model_type='cnn'
)

# There should now be a file 'test_run.json' in your `results/` folder.
# We can use it to load the results of the experiment.
cfg, fit_res = load_experiment('results/test_run_L1_K32-64.json')
_, _ = plot_fit(fit_res, train_test_overlay=True)

# And `cfg` contains the exact parameters to reproduce it
print('experiment config: ', cfg)

# Test experiment1 implementation on a few data samples and with a small model
#ResNet_experiment('test_run_res', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=3)
