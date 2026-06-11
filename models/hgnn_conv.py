import torch
import torch.nn as nn


class AttentionHGNNConv(nn.Module):
    """带注意力门控的超图卷积层"""

    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.attn_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
        self.transform = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, H):
        # x: (B, F, d), H: (B, F, E)

        # 节点 -> 超边聚合
        D_e = H.sum(dim=1).clamp(min=1e-6)
        M_e = torch.bmm(H.transpose(1, 2), x) / D_e.unsqueeze(-1)

        # 超边注意力
        alpha = self.attn_gate(M_e)
        M_e_w = M_e * alpha

        # 超边 -> 节点广播
        x_agg = torch.bmm(H, M_e_w)
        D_v = torch.bmm(H, alpha).clamp(min=1e-6)
        x_agg = x_agg / D_v

        # 变换 + 残差
        x_out = torch.relu(self.transform(x_agg)) + x
        return self.dropout(self.layer_norm(x_out))
