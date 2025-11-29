
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)
