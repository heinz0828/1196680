"""Masked ablation on a trained Full MDHGNN.

This protocol trains the complete model once per seed, then evaluates the same
checkpoint with individual channels disabled. It answers a different question
from retraining each reduced variant: how much each component contributes to the
trained full model's prediction pipeline.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.data_loader import load_copper_data
from data.dataset import create_dataloaders
from data.external_data import download_external_data
from data.feature_engineering import add_technical_indicators
from data.preprocessing import preprocess_data
from data.resampling import resample_copper_ohlcv, resample_external_prices
from models.mdhgnn import MDHGNN
from trainers.trainer import Trainer
from utils.metrics import compute_metrics, print_metrics
from utils.reproducibility import set_all_seeds
from utils.visualization import plot_comparison_bar


MASKED_VARIANTS = {
    "Full MDHGNN": {"mask": (True, True, True), "use_gru": True},
    "w/o Corr-HE (A1)": {"mask": (False, True, True), "use_gru": True},
    "w/o Domain-HE (A2)": {"mask": (True, False, True), "use_gru": True},
    "w/o Adapt-HE (A3)": {"mask": (True, True, False), "use_gru": True},
    "w/o GRU (A5)": {"mask": (True, True, True), "use_gru": False},
}


def prepare_data(cfg, frequency, no_external):
    copper_df = load_copper_data(cfg.csv_path)
    if frequency == "weekly":
        copper_df = resample_copper_ohlcv(copper_df)
    copper_df = add_technical_indicators(copper_df, params=cfg.indicator_params)

    external_df = None
    if not no_external:
        start = copper_df["Date"].min().strftime("%Y-%m-%d")
        end = copper_df["Date"].max().strftime("%Y-%m-%d")
        external_df = download_external_data(start, end, cache_path=cfg.external_cache)
        if frequency == "weekly":
            external_df = resample_external_prices(external_df)

    return preprocess_data(copper_df, external_df, cfg.train_ratio, cfg.val_ratio)


def returns_to_prices(pred_returns, true_returns, base_prices):
    pred_r = pred_returns.reshape(pred_returns.shape[0], -1)
    true_r = true_returns.reshape(true_returns.shape[0], -1)
    base = base_prices.flatten()
    return base * (1 + pred_r[:, -1]), base * (1 + true_r[:, -1])


@torch.no_grad()
def evaluate_variant(model, device, test_loader, variant_cfg):
    corr, domain, adaptive = variant_cfg["mask"]
    model.hg_constructor.set_channel_mask(corr=corr, domain=domain, adaptive=adaptive)
    model.use_gru_path = variant_cfg["use_gru"]
    model.eval()

    preds, targets, bases = [], [], []
    for batch in test_loader:
        x_batch = batch[0].to(device)
        y_hat = model(x_batch)
        preds.append(y_hat.cpu().numpy())
        targets.append(batch[1].numpy())
        bases.append(batch[2].numpy())

    pred_ret = np.concatenate(preds, axis=0)
    true_ret = np.concatenate(targets, axis=0)
    base_px = np.concatenate(bases, axis=0)
    pred_px, true_px = returns_to_prices(pred_ret, true_ret, base_px)
    return compute_metrics(true_px, pred_px)


def summarize(per_seed, seeds):
    metrics = ["RMSE", "MAE", "MAPE", "R2"]
    summary = {}
    for variant in MASKED_VARIANTS:
        summary[variant] = {}
        for metric in metrics:
            values = np.array([per_seed[str(seed)][variant][metric] for seed in seeds], dtype=float)
            summary[variant][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
    return summary


def run_masked_ablation(horizon=1, seeds=None, no_external=False, frequency="weekly"):
    if seeds is None:
        seeds = [42, 123, 456]

    cfg = Config()
    cfg.apply_frequency(frequency)
    cfg.horizon = horizon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, prices, split_indices, _ = prepare_data(cfg, frequency, no_external)
    in_features = features.shape[1]
    per_seed = {}

    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"Masked ablation seed={seed}, horizon={horizon}")
        print(f"{'=' * 70}")
        set_all_seeds(seed)
        train_loader, val_loader, test_loader = create_dataloaders(
            features, prices, split_indices, cfg.window_size, cfg.horizon, cfg.batch_size
        )

        model = MDHGNN(
            in_features=in_features,
            d_model=cfg.d_model,
            window_size=cfg.window_size,
            n_hgnn_layers=cfg.n_hgnn_layers,
            k_neigs=cfg.k_neigs,
            n_adaptive_edges=cfg.n_adaptive_edges,
            gru_hidden=cfg.gru_hidden,
            horizon=cfg.horizon,
            dropout=cfg.dropout,
        )
        trainer = Trainer(
            model,
            device,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
            patience=cfg.patience,
            checkpoint_dir=cfg.checkpoint_dir,
        )
        trainer.train(train_loader, val_loader, max_epochs=cfg.max_epochs, print_freq=50)

        per_seed[str(seed)] = {}
        for variant, variant_cfg in MASKED_VARIANTS.items():
            metrics = evaluate_variant(trainer.model, device, test_loader, variant_cfg)
            print_metrics(metrics, variant)
            per_seed[str(seed)][variant] = {k: float(v) for k, v in metrics.items()}

    summary = summarize(per_seed, seeds)
    mean_for_plot = {
        variant: {metric: values[metric]["mean"] for metric in ["RMSE", "MAE", "MAPE", "R2"]}
        for variant, values in summary.items()
    }

    os.makedirs(cfg.table_dir, exist_ok=True)
    per_seed_path = os.path.join(cfg.table_dir, f"masked_ablation_h{horizon}_per_seed.json")
    summary_path = os.path.join(cfg.table_dir, f"masked_ablation_h{horizon}_multi_seed.json")
    manifest_path = os.path.join(cfg.table_dir, f"masked_ablation_h{horizon}_manifest.json")
    official_per_seed_path = os.path.join(cfg.table_dir, f"ablation_h{horizon}_per_seed.json")
    official_summary_path = os.path.join(cfg.table_dir, f"ablation_h{horizon}_multi_seed.json")
    official_manifest_path = os.path.join(cfg.table_dir, f"ablation_h{horizon}_manifest.json")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frequency": frequency,
        "horizon": horizon,
        "seeds": seeds,
        "protocol": (
            "Train Full MDHGNN once per seed; evaluate the same checkpoint "
            "with one channel/path masked at inference."
        ),
        "variants": MASKED_VARIANTS,
        "result_files": [
            os.path.basename(official_per_seed_path),
            os.path.basename(official_summary_path),
            os.path.basename(per_seed_path),
            os.path.basename(summary_path),
        ],
    }

    with open(per_seed_path, "w", encoding="utf-8") as f:
        json.dump(per_seed, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(official_per_seed_path, "w", encoding="utf-8") as f:
        json.dump(per_seed, f, indent=2)
    with open(official_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(official_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    os.makedirs(cfg.figure_dir, exist_ok=True)
    masked_figure_path = os.path.join(cfg.figure_dir, f"masked_ablation_RMSE_h{horizon}.png")
    plot_comparison_bar(
        mean_for_plot,
        "RMSE",
        save_path=masked_figure_path,
    )
    masked_base, _ = os.path.splitext(masked_figure_path)
    official_base = os.path.join(cfg.figure_dir, f"ablation_RMSE_h{horizon}")
    for ext in [".png", ".pdf", ".svg"]:
        src = masked_base + ext
        dst = official_base + ext
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", default="weekly", choices=["daily", "weekly"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--no-external", action="store_true")
    args = parser.parse_args()
    run_masked_ablation(
        horizon=args.horizon,
        seeds=args.seeds,
        no_external=args.no_external,
        frequency=args.frequency,
    )
