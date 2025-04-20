import torch
from torch import nn
import numpy
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            # normalized input planes
            nn.Conv2d(
                in_channels=3, 
                out_channels=24,
                kernel_size=5,
                stride=2),
            nn.ELU(),
            # CNN feature map
            nn.Conv2d(
                in_channels=24, 
                out_channels=36,
                kernel_size=5,
                stride=2),
            nn.ELU(),
            nn.Conv2d(
                in_channels=36, 
                out_channels=48,
                kernel_size=5,
                stride=2),
            nn.ELU(),
            nn.Conv2d(
                in_channels=48, 
                out_channels=64,
                kernel_size=3,
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64,
                kernel_size=3,
            ),
            nn.ELU(),
            # flatten CNN 3d features for linear models
            nn.Flatten(),
            # dense
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            # output node
            nn.Linear(10, 3),
        )

    def forward(self, x):
        return self.model(x)