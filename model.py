import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):
    def __init__(self, in_features):
        super(FCNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class Conv1dNet(nn.Module):
    def __init__(self):
        super(Conv1dNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 24)
        self.conv2 = nn.Conv1d(8, 8, 12)
        self.conv3 = nn.Conv1d(8, 8, 6)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool1d(x, 1)

        x = self.flatten(x)
        x = self.fc(x)

        return x
