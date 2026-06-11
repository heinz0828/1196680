import torch
import torch.nn as nn
import math


class TransformerBaseline(nn.Module):
    """Transformer encoder baseline for time series prediction."""

    def __init__(self, in_features: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 window_size: int = 60, horizon: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, F)
        Returns:
            (B, horizon)
        """
        x = self.input_proj(x) + self.pos_encoding  # (B, W, d_model)
        x = self.encoder(x)                          # (B, W, d_model)
        x = x.mean(dim=1)                            # mean pooling -> (B, d_model)
        return self.fc(x)
