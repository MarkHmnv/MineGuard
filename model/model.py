"""
Realization of the model architecture described in the article
"Detection and classification of landmines using machine learning applied to metal detector data"
Link to the article: https://doi.org/10.1080/0952813X.2020.1735529
"""

import torch
import torch.nn as nn
from torchsummary import summary

from constants import NUM_CLASSES


class DualPathNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.first_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.15),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

        self.second_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.concat_layer = nn.Linear(192, 128)

        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.15),
            nn.Linear(32, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.first_conv_layers(x)
        x2 = self.second_conv_layers(x)
        x2 = torch.flatten(x2, start_dim=1)
        x = torch.cat((x1.view(x1.size(0), -1), x2), dim=1)
        x = self.concat_layer(x)
        x = self.fc_layers(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DualPathNet(num_classes=NUM_CLASSES)
    summary(model.cuda(), (1, 64, 87))