import torch, torch.nn as nn
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)
    def forward(self, encoder_outputs):
        scores = self.proj(encoder_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (weights * encoder_outputs).sum(dim=1)
        return context, weights
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.att = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        context, weights = self.att(out)
        out = self.fc(context)
        return out, weights
