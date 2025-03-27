import torch
import torch.nn as nn

class AutoencoderCIFAR10_original(nn.Module):
    def __init__(self, latent_dim=128):
        super(AutoencoderCIFAR10_original, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (224X224) -> (112X112)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (112X112) -> (56x56)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # (56x56) -> (28x28)
            #nn.BatchNorm2d(256),
            #nn.ReLU(),

            nn.Flatten(),  # Flatten the feature map
            nn.Linear(64 * 28 * 28, latent_dim),  # Convert to latent vector
        )
        
        # Decoder (Mirrors Encoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 28 * 28),  # Expand latent vector to match CNN shape
            nn.Unflatten(1, (64, 28, 28)),  # Reshape to (128, 4, 4)

            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (4x4) -> (8x8)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8x8) -> (16x16)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16x16) -> (32x32)
            nn.Sigmoid()  # Normalize output to [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


import torch
import torch.nn as nn

class AutoencoderCIFAR10_2(nn.Module): # כמו המקורי, אבל שכבה אחרונה משתנה דינאמית לפי גודל התמונה שנכנסת
    def __init__(self, input_size=(3, 224, 224), latent_dim=128):
        super(AutoencoderCIFAR10_2, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # CNN encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            conv_out = self.encoder_cnn(dummy_input)
            self.flattened_size = conv_out.view(1, -1).shape[1]

        # Fully connected encoder
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, latent_dim),
        )

    def encoder(self, x):
        x = self.encoder_cnn(x)
        x = self.encoder_fc(x)
        return x

    def forward(self, x):
        return self.encoder(x)


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

class AutoencoderCIFAR10_3(nn.Module): # אמור להיות כמו המודל של FLAME2, ומשתנה דינאמית לפי גודל התמונה שנכנסת
    def __init__(self, input_size=(3, 32, 32), latent_dim=8):
        super(AutoencoderCIFAR10_3, self).__init__()
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

        # Compute actual output dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size)
            out = self.encoder_features(dummy)
            self.encoder_output_dim = out.shape[1]

        # Project to latent_dim
        self.projector = nn.Linear(self.encoder_output_dim, latent_dim)

    def encoder_features(self, x):
        x = self.IN(x)
        r = self.residual_conv(x)
        x = self.block(x)
        x = x + r
        x = self.block2(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def encoder(self, x):
        x = self.encoder_features(x)
        x = self.projector(x)  # Project to latent_dim
        return x

    def forward(self, x):
        return self.encoder(x)


class AutoencoderCIFAR10(nn.Module): # אמור להיות כמו FLAME2 אבל בלי שינוי דינאמי
    def __init__(self, input_size=(3, 32, 32), latent_dim=64):
        super(AutoencoderCIFAR10, self).__init__()
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
