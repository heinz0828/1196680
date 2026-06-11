import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """2-layer GRU baseline for time series prediction."""

    def __init__(self, in_features: int, hidden_size: int = 64,
                 num_layers: int = 2, horizon: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        h_last = h_n[-1]
        return self.fc(h_last)
