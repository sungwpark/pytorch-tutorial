import torch
import torch.nn as nn

class FCNetwork(nn.Module):
    def __init__(self, in_features):
        super(FCNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
