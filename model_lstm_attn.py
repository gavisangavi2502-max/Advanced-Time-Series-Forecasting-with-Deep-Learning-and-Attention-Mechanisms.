
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        weights = torch.softmax(self.attn(h), dim=1)
        ctx = (weights * h).sum(dim=1)
        return ctx, weights

class LSTMAttnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        ctx, w = self.attn(h)
        out = self.fc(ctx)
        return out
