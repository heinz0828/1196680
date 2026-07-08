"""MDHGNN消融实验"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json

from config import Config
from utils.reproducibility import set_all_seeds
from utils.metrics import compute_metrics, print_metrics
from utils.visualization import plot_comparison_bar
from data.data_loader import load_copper_data
from data.external_data import download_external_data
from data.feature_engineering import add_technical_indicators
from data.resampling import resample_copper_ohlcv, resample_external_prices
from data.preprocessing import filter_sample_period, preprocess_data
from data.dataset import create_dataloaders
from models.mdhgnn import MDHGNN
from trainers.trainer import Trainer


ABLATION_CONFIGS = {
    'Full MDHGNN': {},
    'w/o Corr-HE (A1)': {'k_neigs': []},
    # The active MDHGNN implementation uses a feature-domain prior channel
    # rather than the older temporal sliding-window hyperedges.
    'w/o Domain-HE (A2)': {'use_domain_edges': False},
    'w/o Adapt-HE (A3)': {'n_adaptive_edges': 0},
    'w/o GRU (A5)': {'use_mean_pool': True},
}


class MDHGNNNoGRU(MDHGNN):
    """MDHGNN variant replacing GRU+Attention with mean pooling on Path A."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_temporal_proj = torch.nn.Linear(
            self.input_proj.out_features, self.temporal_proj.in_features
        )
        for param in self.temporal_agg.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Path A: Mean pooling instead of GRU
        x_seq = self.input_proj(x) + self.pos_enc    # (B, W, d)
        h_temporal = self.mean_temporal_proj(x_seq.mean(dim=1))

        # Path B: Structural (same as MDHGNN)
        node_emb = self.node_encoder(x)
        H = self.hg_constructor(node_emb, x)
        z = node_emb
        for layer in self.hgnn_layers:
            z = layer(z, H)
        h_struct = self.struct_pool(z)                 # (B, d)

        # Fusion + Prediction (same as MDHGNN)
        alpha = torch.sigmoid(self.fusion_alpha)
        h_t = self.temporal_proj(h_temporal)
        h_s = self.struct_proj(h_struct)
        h_fused = self.fusion_drop(self.fusion_norm(alpha * h_t + (1 - alpha) * h_s))
        return self.pred_head(h_fused)


def build_ablation_model(name, overrides, in_features, cfg):
    """Build one ablation variant from the config table."""
    model_class = MDHGNNNoGRU if overrides.get('use_mean_pool', False) else MDHGNN
    return model_class(
        in_features=in_features, d_model=cfg.d_model,
        window_size=cfg.window_size, n_hgnn_layers=cfg.n_hgnn_layers,
        k_neigs=overrides.get('k_neigs', cfg.k_neigs),
        n_adaptive_edges=overrides.get('n_adaptive_edges', cfg.n_adaptive_edges),
        gru_hidden=cfg.gru_hidden, horizon=cfg.horizon, dropout=cfg.dropout,
        use_domain_edges=overrides.get('use_domain_edges', True)
    )


def run_ablation(horizon=1, seed=42, no_external=False, frequency='weekly'):
    cfg = Config()
    cfg.apply_frequency(frequency)
    cfg.horizon = horizon
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_all_seeds(seed)
    copper_df = load_copper_data(cfg.csv_path)
    if cfg.require_common_real_period:
        copper_df = filter_sample_period(
            copper_df, cfg.sample_start_date, cfg.sample_end_date
        )
    if frequency == 'weekly':
        copper_df = resample_copper_ohlcv(copper_df)
    copper_df = add_technical_indicators(copper_df, params=cfg.indicator_params)

    external_df = None
    if not no_external:
        start = copper_df['Date'].min().strftime('%Y-%m-%d')
        end = copper_df['Date'].max().strftime('%Y-%m-%d')
        external_df = download_external_data(start, end, cache_path=cfg.external_cache)
        if frequency == 'weekly':
            external_df = resample_external_prices(external_df)

    features, prices, split_indices, norm_params = preprocess_data(
        copper_df, external_df, cfg.train_ratio, cfg.val_ratio
    )
    in_features = features.shape[1]

    results = {}

    for name, overrides in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        set_all_seeds(seed)
        train_loader, val_loader, test_loader = create_dataloaders(
            features, prices, split_indices, cfg.window_size, cfg.horizon, cfg.batch_size
        )

        model = build_ablation_model(name, overrides, in_features, cfg)

        trainer = Trainer(model, device, lr=cfg.lr, weight_decay=cfg.weight_decay,
                          grad_clip=cfg.grad_clip, patience=cfg.patience,
                          checkpoint_dir=cfg.checkpoint_dir)
        trainer.train(train_loader, val_loader, max_epochs=cfg.max_epochs, print_freq=50)

        pred_ret, true_ret, base_px = trainer.predict(test_loader)
        base = base_px.flatten()
        preds = base * (1 + pred_ret.flatten())
        targets = base * (1 + true_ret.flatten())
        metrics = compute_metrics(targets, preds)
        print_metrics(metrics, name)
        results[name] = {k: float(v) for k, v in metrics.items()}

    # Summary
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY (Horizon={horizon})")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'RMSE':>10} {'MAE':>10} {'MAPE(%)':>10} {'R2':>10}")
    print('-' * 70)
    for name, m in results.items():
        print(f"{name:<25} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['MAPE']:>10.2f} {m['R2']:>10.4f}")

    # Save
    os.makedirs(cfg.table_dir, exist_ok=True)
    with open(os.path.join(cfg.table_dir, f'ablation_h{horizon}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_comparison_bar(results, 'RMSE',
                        save_path=os.path.join(cfg.figure_dir, f'ablation_RMSE_h{horizon}.png'))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=str, default='weekly', choices=['daily', 'weekly'])
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-external', action='store_true')
    args = parser.parse_args()
    run_ablation(args.horizon, args.seed, args.no_external, args.frequency)
