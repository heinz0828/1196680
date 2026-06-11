import torch.nn as nn


class PredictionHead(nn.Module):
    """两层MLP回归头"""

    def __init__(self, in_features, hidden=64, horizon=1, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, horizon)
        )

    def forward(self, x):
        return self.mlp(x)
