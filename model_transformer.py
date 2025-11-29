import torch, torch.nn as nn
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, seq_len=48, horizon=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg = nn.Linear(d_model*seq_len, horizon)
    def forward(self, x):
        x = self.input_proj(x)
        z = self.encoder(x)
        out = self.reg(z.flatten(start_dim=1))
        return out
