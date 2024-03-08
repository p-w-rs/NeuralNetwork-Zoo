# models.py
import torch
from torch import nn


class FFNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1736),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1736, 868),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(868, 62),
        )

    def forward(self, x):
        return self.model(x)


class FullyConnectedResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        # Skip connection if dimensions match
        if input_dim == output_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += identity
        return self.relu(out)


class FFResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            FullyConnectedResBlock(28 * 28, 512),
            nn.Dropout(0.2),
            FullyConnectedResBlock(512, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 62),  # Adjust the output dimension as per your task
        )

    def forward(self, x):
        return self.model(x)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding="same"),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding="same"),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(1080, 84),
            nn.Tanh(),
            nn.Linear(84, 62),
        )

    def forward(self, x):
        return self.model(x)
