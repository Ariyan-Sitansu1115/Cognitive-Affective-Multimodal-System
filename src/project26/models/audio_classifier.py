from __future__ import annotations

import torch
from torch import nn


class AudioMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.30):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)