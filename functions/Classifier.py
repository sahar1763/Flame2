import torch
import torch.nn as nn

# Define Classifier Model CIFAR10
class ClassifierCifar_original(nn.Module):
    def __init__(self, input_dim=128, num_classes=2):
        super(ClassifierCifar_original, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)  # 10 classes
        )


    def forward(self, x):
        return self.fc(x)



class ClassifierCifar(nn.Module):
    def __init__(self, input_dim=8, num_classes=4):
        super(ClassifierCifar, self).__init__()

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

        # # Freeze the decoder since we only care about the encoder
        # for param in self.autoencoder.decoder.parameters():
        #     param.requires_grad = False
    
    def forward(self, x):
        latent = self.autoencoder.encoder(x)  # Get latent code from encoder
        output = self.classifier(latent)      # Classify based on latent code
        return output



