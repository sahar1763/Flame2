import torch
import torch.nn as nn
import torchvision
import itertools as it
from torch import Tensor
from typing import Sequence
from torchvision.models import resnet18, ResNet18_Weights

from mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS


POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            hidden_dims: Sequence[int],
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
            *args, **kwargs
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__(*args, **kwargs)
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        P = self.pool_every  # Number of conv layers before each pooling
        N = len(self.channels)  # Total number of conv layers

        for i in range(N):
            # Add a convolutional layer followed by activation
            layers.append(nn.Conv2d(in_channels, self.channels[i], **self.conv_params))
            in_channels = self.channels[i]
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))

            # Add a pooling layer after every P layers, except the last group
            if (i + 1) % P == 0:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

        # raise NotImplementedError()
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            # Create a dummy input tensor with the same size as the input images
            dummy_input = torch.zeros(1, *self.in_size)
            # Pass the dummy input through the feature extractor
            output_tensor = self.feature_extractor(dummy_input)
            # Calculate the number of features in the output tensor
            num_features = output_tensor.numel() // output_tensor.size(
                0)  # Divide by batch size to get per-example features
            return num_features
            # raise NotImplementedError()
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        nonlin = ACTIVATIONS[self.activation_type](**self.activation_params)
        nonlins = [nonlin] * len(self.hidden_dims)
        mlp = MLP(self._n_features(), self.hidden_dims + [self.out_classes], nonlins=(nonlins + ["none"]))
        # raise NotImplementedError()
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.mlp(features)
        # raise NotImplementedError()
        # ========================
        return out





class ResNet18Custom(nn.Module):
    """
    A ResNet18-based model with an additional fully connected layer at the end.
    Only the last two layers of ResNet18 and the added layer are trainable.
    """

    def __init__(self, out_classes: int, freeze_until: int = 2, *args, **kwargs):
        """
        :param out_classes: Number of output classes for the model.
        :param freeze_until: Index of the ResNet18 layers to freeze. Default is 2,
                             which freezes all layers except the last two.
        """
        super().__init__(*args, **kwargs)
        # Load the pre-trained ResNet18 model
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all layers except the last freeze_until layers
        self.freeze_until = freeze_until
        flag = len(list(self.base_model.parameters())) - freeze_until
        for idx, param in enumerate(self.base_model.parameters()):
            param.requires_grad = idx >= flag

        # Replace the final fully connected layer of ResNet18
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove original FC layer

        # Add custom fully connected layer
        self.custom_fc = nn.Linear(in_features, out_classes)

    def forward(self, x):
        # Pass the input through the base ResNet18 (excluding its original FC layer)
        features = self.base_model(x)
        # Pass the extracted features through the custom FC layer
        output = self.custom_fc(features)

        return output