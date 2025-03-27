import torch
import torch.nn as nn

# Define Classifier Model
class Classifier(nn.Module):
    def __init__(self, input_dim=8, num_classes=4):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.fc(x)



# Define combined model (Autoencoder + Classifier)
class AutoencoderClassifier(nn.Module):
    def __init__(self, autoencoder, classifier):
        super(AutoencoderClassifier, self).__init__()
        self.autoencoder = autoencoder  # Autoencoder that generates latent code
        self.classifier = classifier    # Classifier that predicts based on latent code
    
    def forward(self, x):
        latent = self.autoencoder.encoder(x)  # Get latent code from encoder
        output = self.classifier(latent)      # Classify based on latent code
        return output



