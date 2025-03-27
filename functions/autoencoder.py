import torch
import torch.nn as nn


class SeparableConv2d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Encoder2(nn.Module):
    def __init__(self, input_size=(3, 32, 32), latent_dim=64):
        super(Encoder2, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.IN = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.residual_conv = nn.Conv2d(8, 8, kernel_size=1, stride=2)

        self.block = nn.Sequential(
            SeparableConv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            SeparableConv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            SeparableConv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

        # Static projection (no dummy input needed)
        self.projector = nn.Linear(8, latent_dim)

    def encoder_features(self, x):
        x = self.IN(x)
        r = self.residual_conv(x)
        x = self.block(x)
        x = x + r
        x = self.block2(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        return x

    def encoder(self, x):
        x = self.encoder_features(x)
        x = self.projector(x)
        return x

    def forward(self, x):
        return self.encoder(x)
