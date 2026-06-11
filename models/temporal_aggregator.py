import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAggregator(nn.Module):
    """GRU + 时序注意力池化"""

    def __init__(self, d_model, gru_hidden=64, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(d_model, gru_hidden, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.attn_proj = nn.Linear(gru_hidden, gru_hidden)
        self.attn_score = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        # x: (B, W, d) -> (B, 2*gru_hidden)
        h_all, h_last = self.gru(x)
        h_last = h_last[-1]

        score = self.attn_score(torch.tanh(self.attn_proj(h_all)))
        attn = F.softmax(score, dim=1)
        context = (attn * h_all).sum(dim=1)

        return torch.cat([context, h_last], dim=-1)
