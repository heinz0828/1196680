import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Causal dilated convolution block with residual."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.ln1 = nn.GroupNorm(1, channels)  # equivalent to LayerNorm for Conv1d
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.ln2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = torch.relu(self.ln1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = torch.relu(self.ln2(out))
        out = self.dropout(out)

        return out + x  # residual


class TCNBaseline(nn.Module):
    """Temporal Convolutional Network (Bai et al., 2018)."""

    def __init__(self, in_features, hidden_size=64, num_layers=3,
                 kernel_size=3, horizon=1, dropout=0.2):
        super().__init__()
        # Project input features to hidden_size first
        self.input_proj = nn.Linear(in_features, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        blocks = []
        for i in range(num_layers):
            blocks.append(TemporalBlock(hidden_size, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        # x: (B, W, F)
        x = self.input_norm(self.input_proj(x))  # (B, W, H)
        x = x.permute(0, 2, 1)                    # (B, H, W)
        out = self.blocks(x)                       # (B, H, W)
        out = out[:, :, -1]                        # (B, H) last step
        return self.fc(out)
