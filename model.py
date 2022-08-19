import torch
import torch.nn as nn

class FCNetwork(nn.Module):
    def __init__(self, in_features):
        super(FCNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
