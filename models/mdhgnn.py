import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .hgnn_conv import AttentionHGNNConv
from .temporal_aggregator import TemporalAggregator
from .prediction_head import PredictionHead


class FeatureNodeEncoder(nn.Module):
    """每个特征的时序 -> Conv1d -> 时序注意力池化 -> 节点嵌入"""

    def __init__(self, window_size: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, W, F) -> (B, F, d)
        B, W, F = x.shape
        x = x.permute(0, 2, 1).reshape(B * F, 1, W)
        x = self.conv(x).permute(0, 2, 1)          # (B*F, W, d)
        w = Fn.softmax(self.attn(x), dim=1)         # (B*F, W, 1)
        x = (w * x).sum(dim=1).reshape(B, F, -1)
        return self.norm(x)


class FeatureHypergraphConstructor(nn.Module):
    """Feature hypergraph with correlation, domain-prior, and adaptive channels."""

    def __init__(self, num_features, d_model, n_corr_heads=3,
                 n_adaptive_edges=8, use_domain_edges=True):
        super().__init__()
        self.num_features = num_features

        # A: 相关性软超边
        self.n_corr_heads = n_corr_heads
        self.corr_temps = nn.ParameterList([
            nn.Parameter(torch.tensor(float(i + 2))) for i in range(n_corr_heads)
        ])

        # B: 领域知识超边
        domain_H = self._build_domain_H(num_features) if use_domain_edges else torch.zeros(num_features, 0)
        self.register_buffer('domain_H', domain_H)

        # C: 自适应超边
        self.adapt_proj = nn.Linear(d_model, n_adaptive_edges) if n_adaptive_edges > 0 else None
        if self.adapt_proj is not None:
            nn.init.zeros_(self.adapt_proj.weight)
            nn.init.zeros_(self.adapt_proj.bias)
        self.adapt_tau = nn.Parameter(torch.ones(1))
        # Channel gates let the full model suppress a noisy channel instead of
        # forcing all constructed hyperedges to contribute equally.
        self.channel_logits = nn.Parameter(torch.tensor([1.2, -4.0, -6.0]))
        self.register_buffer('channel_mask', torch.ones(3))
        self.last_channel_completeness = None

    def set_channel_mask(self, corr=True, domain=True, adaptive=True):
        mask = torch.tensor(
            [float(corr), float(domain), float(adaptive)],
            device=self.channel_mask.device,
            dtype=self.channel_mask.dtype,
        )
        self.channel_mask.copy_(mask)

    def _build_domain_H(self, nf):
        # 特征顺序: 0-5 OHLCV+Change, 6-9 MA, 10-13 RSI/MACD,
        # 14-16 BB, 17 ATR, 18-20 ROC/WR/CCI, 21 OBV, 22-29 外部
        edges = []

        def edge(idx):
            e = torch.zeros(nf)
            for i in idx:
                if i < nf:
                    e[i] = 1.0
            return e

        edges.append(edge([0, 1, 2, 3]))                           # 价格
        edges.append(edge([3, 6, 7, 8, 9]))                        # 趋势
        if nf > 13: edges.append(edge([10, 11, 12, 13, 18, 19, 20]))  # 动量
        if nf > 17: edges.append(edge([14, 15, 16, 17]))              # 波动率
        if nf > 21: edges.append(edge([4, 21]))                       # 量能
        if nf > 5: edges.append(edge([3, 5]))                         # 铜价/涨跌幅
        if nf > 25: edges.append(edge([22, 23, 24, 25]))              # 外部收盘价
        if nf > 29: edges.append(edge([26, 27, 28, 29]))              # 外部收益率
        edges.append(torch.ones(nf))                                   # 全局

        return torch.stack(edges, dim=1)

    def forward(self, node_emb, x_raw):
        # node_emb: (B, F, d), x_raw: (B, W, F) -> H: (B, F, E_total)
        B, F, d = node_emb.shape

        H_parts = []

        # A: 从原始时序算特征间相关性
        H_corr = []
        for temp in self.corr_temps:
            x_t = x_raw.permute(0, 2, 1)                   # (B, F, W)
            x_t = x_t - x_t.mean(dim=-1, keepdim=True)
            x_t = Fn.normalize(x_t, p=2, dim=-1)
            sim = torch.bmm(x_t, x_t.transpose(1, 2)).abs()  # (B, F, F)
            H_corr.append(Fn.softmax(sim / temp.clamp(min=0.5), dim=1))
        if H_corr:
            H_parts.append(
                self.channel_mask[0] * torch.sigmoid(self.channel_logits[0]) * torch.cat(H_corr, dim=2)
            )

        # B: 领域知识（静态）
        if self.domain_H.shape[1] > 0:
            H_parts.append(
                self.channel_mask[1]
                * torch.sigmoid(self.channel_logits[1])
                * self.domain_H.unsqueeze(0).expand(B, -1, -1)
            )

        # C: 可学习
        if self.adapt_proj is not None:
            logits = self.adapt_proj(node_emb) / self.adapt_tau.clamp(min=0.1)
            H_parts.append(
                self.channel_mask[2] * torch.sigmoid(self.channel_logits[2]) * Fn.softmax(logits, dim=1)
            )

        channel_presence = node_emb.new_tensor([
            1.0 if H_corr else 0.0,
            1.0 if self.domain_H.shape[1] > 0 else 0.0,
            1.0 if self.adapt_proj is not None else 0.0,
        ])
        self.last_channel_completeness = (self.channel_mask * channel_presence).mean().pow(2.0)

        if not H_parts:
            return torch.ones(B, F, 1, device=node_emb.device, dtype=node_emb.dtype)
        return torch.cat(H_parts, dim=2)


class StructuralEncoder(nn.Module):
    """注意力池化: (B, F, d) -> (B, d)"""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, z):
        w = Fn.softmax(self.attn(z), dim=1)
        return (w * z).sum(dim=1)


class MDHGNN(nn.Module):
    """
    多尺度动态超图神经网络 (MDHGNN)

    双路径架构:
      路径A — 时序: 投影+位置编码 -> GRU -> 时序注意力
      路径B — 结构: 特征即节点超图 -> HGNN卷积 -> 注意力池化
      融合: alpha * 时序 + (1-alpha) * 结构 -> 预测头
    """

    def __init__(self, in_features, d_model=64, window_size=60,
                 n_hgnn_layers=2, k_neigs=None,
                 n_adaptive_edges=8, gru_hidden=64, horizon=1, dropout=0.3,
                 use_domain_edges=True):
        super().__init__()
        if k_neigs is None:
            k_neigs = [3, 5, 8]
        self.in_features = in_features

        # 路径A: 时序
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        self.temporal_agg = TemporalAggregator(d_model, gru_hidden, dropout=dropout)

        # 路径B: 超图结构
        self.node_encoder = FeatureNodeEncoder(window_size, d_model)
        self.hg_constructor = FeatureHypergraphConstructor(
            in_features, d_model, len(k_neigs), n_adaptive_edges,
            use_domain_edges=use_domain_edges)
        self.hgnn_layers = nn.ModuleList([
            AttentionHGNNConv(d_model, dropout) for _ in range(n_hgnn_layers)])
        self.struct_pool = StructuralEncoder(d_model)

        # 融合
        temporal_dim = 2 * gru_hidden
        fused_dim = d_model
        self.temporal_proj = nn.Linear(temporal_dim, fused_dim)
        self.struct_proj = nn.Linear(d_model, fused_dim)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.0))
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_drop = nn.Dropout(dropout)

        self.pred_head = PredictionHead(fused_dim, gru_hidden, horizon)
        self.use_gru_path = True

    def forward(self, x):
        # 路径A
        x_seq = self.input_proj(x) + self.pos_enc
        if self.use_gru_path:
            h_temporal = self.temporal_agg(x_seq)
        else:
            h_temporal = torch.cat([x_seq.mean(dim=1), x_seq[:, -1, :]], dim=1)

        # 路径B
        node_emb = self.node_encoder(x)
        H = self.hg_constructor(node_emb, x)
        z = node_emb
        for layer in self.hgnn_layers:
            z = layer(z, H)
        h_struct = self.struct_pool(z)
        if self.hg_constructor.last_channel_completeness is not None:
            h_struct = h_struct * self.hg_constructor.last_channel_completeness

        # 融合 + 预测
        alpha = torch.sigmoid(self.fusion_alpha)
        h_t = self.temporal_proj(h_temporal)
        h_s = self.struct_proj(h_struct)
        h_fused = self.fusion_drop(self.fusion_norm(alpha * h_t + (1 - alpha) * h_s))
        return self.pred_head(h_fused)
