# models/cnn_discriminator.py
import torch
import torch.nn as nn
import torch.optim as optim
from .config import ModelConfig

class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        


        ds = config.IMG_SIZE // (2 ** 5)
        
        self.features = nn.Sequential(
            nn.Conv2d(config.IN_CHANNELS, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * ds * ds, 1024),
            nn.Linear(1024, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.LR, betas=config.BETAS)

    def forward(self, x):
        x = self.features(x)
        logit = self.classifier(x)
        return logit

    def step(self, images, labels):
        self.optimizer.zero_grad()
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()