import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Multi-Layer Perceptron baseline. Flattens the window and predicts."""

    def __init__(self, in_features, window_size=60, hidden_size=128,
                 horizon=1, dropout=0.2):
        super().__init__()
        input_dim = in_features * window_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        return self.net(x)
