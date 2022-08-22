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
        self.conv1 = nn.Conv1d(14, 8, 24)
        self.conv2 = nn.Conv1d(8, 8, 12)
        self.conv3 = nn.Conv1d(8, 8, 6)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
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

class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(14, 16, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        h = self.fc(h)

        return h


class TextLSTMNet(nn.Module):
    def __init__(self, vocab_size):
        super(TextLSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        #self.dropout = nn.Dropout(0.5)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text)
        out, (h, c) = self.lstm(embedded)
        h = self.fc(h)

        return F.sigmoid(h)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextTransformer, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super(Seq2SeqTransformer, self).__init__()