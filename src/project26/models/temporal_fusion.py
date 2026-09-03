import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusionTabular(nn.Module):
    """Documented tabular-only stand-in for the proposed multimodal model."""
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_groups=8):
        super().__init__()
        self.n_groups = min(n_groups, max(input_dim, 1))
        self.group_size = max(input_dim // self.n_groups, 1)
        pad_dim = self.group_size * self.n_groups
        self.pad_dim = pad_dim
        self.token_proj = nn.Linear(self.group_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        B = x.shape[0]
        if x.shape[1] < self.pad_dim:
            x = F.pad(x, (0, self.pad_dim - x.shape[1]))
        else:
            x = x[:, :self.pad_dim]
        tokens = x.view(B, self.n_groups, self.group_size)
        h = self.encoder(self.token_proj(tokens)).mean(dim=1)
        return self.head(h)


HybridModel = TemporalFusionTabular
