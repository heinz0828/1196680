"""Redraw publication-ready figures from saved checkpoints.

This script does not train any model. It rebuilds the weekly test set, loads
existing checkpoints, runs test-set inference, and writes figures with date
axes, units, consistent model names/colors, and explicit test-period labels.
"""
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from config import Config
from data.data_loader import load_copper_data
from data.dataset import create_dataloaders
from data.external_data import download_external_data
from data.feature_engineering import add_technical_indicators
from data.preprocessing import filter_sample_period
from data.resampling import resample_copper_ohlcv, resample_external_prices
from experiments.run_all import build_model, get_default_model_names, returns_to_prices
from utils.reproducibility import set_all_seeds


MODEL_COLORS = {
    "MDHGNN": "#D55E00",
    "GRU": "#0072B2",
    "MLP": "#009E73",
    "Transformer": "#CC79A7",
    "Naive": "#7A7A7A",
}

MODEL_LABELS = {
    "MDHGNN": "MDHGNN",
    "GRU": "GRU",
    "MLP": "MLP",
    "Transformer": "Transformer",
    "Naive": "Naive persistence",
}

METRIC_LABELS = {
    "RMSE": "RMSE (USD/lb)",
    "MAE": "MAE (USD/lb)",
    "MAPE": "MAPE (%)",
    "DA": "Direction accuracy (%)",
    "Sharpe": "Sharpe ratio",
}


def save_figure(fig: plt.Figure, out_base: str) -> None:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def preprocess_data_with_dates(
    copper_df: pd.DataFrame,
    external_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    """Mirror data.preprocessing.preprocess_data while retaining dates."""
    df = copper_df.copy()

    if external_df is not None and len(external_df) > 0:
        df["Date"] = pd.to_datetime(df["Date"])
        external_df = external_df.copy()
        external_df["Date"] = pd.to_datetime(external_df["Date"])
        df = pd.merge(df, external_df, on="Date", how="left")
        ext_cols = [c for c in external_df.columns if c != "Date"]
        for col in ext_cols:
            df[col] = df[col].ffill()

    feature_cols = [c for c in df.columns if c != "Date"]

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    df[feature_cols] = df[feature_cols].ffill()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    dates = pd.to_datetime(df["Date"]).to_numpy()
    prices = df["Price"].values.copy().astype(np.float32)
    features = df[feature_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    split_indices = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }

    train_data = features[:train_end]
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0
    features = (features - mean) / std

    norm_params = {
        "mean": mean,
        "std": std,
        "feature_cols": feature_cols,
        "price_col_idx": feature_cols.index("Price"),
    }

    return features, prices, dates, split_indices, norm_params


def build_weekly_dataset(cfg: Config):
    set_all_seeds(42)
    copper_df = load_copper_data(cfg.csv_path)
    if cfg.require_common_real_period:
        copper_df = filter_sample_period(
            copper_df, cfg.sample_start_date, cfg.sample_end_date
        )
    raw_range = (
        copper_df["Date"].min().strftime("%Y-%m-%d"),
        copper_df["Date"].max().strftime("%Y-%m-%d"),
        int(len(copper_df)),
    )

    copper_df = resample_copper_ohlcv(copper_df)
    weekly_range = (
        copper_df["Date"].min().strftime("%Y-%m-%d"),
        copper_df["Date"].max().strftime("%Y-%m-%d"),
        int(len(copper_df)),
    )

    copper_df = add_technical_indicators(copper_df, params=cfg.indicator_params)
    start = copper_df["Date"].min().strftime("%Y-%m-%d")
    end = copper_df["Date"].max().strftime("%Y-%m-%d")
    external_df = download_external_data(start, end, cache_path=cfg.external_cache)
    external_df = resample_external_prices(external_df)

    features, prices, dates, split_indices, norm_params = preprocess_data_with_dates(
        copper_df, external_df, cfg.train_ratio, cfg.val_ratio
    )
    final_range = (
        pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
        pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        int(len(dates)),
    )
    ranges = {
        "common_real_daily": raw_range,
        "weekly_before_indicators": weekly_range,
        "final_after_indicators": final_range,
        "train": (
            pd.Timestamp(dates[split_indices["train"][0]]).strftime("%Y-%m-%d"),
            pd.Timestamp(dates[split_indices["train"][1] - 1]).strftime("%Y-%m-%d"),
        ),
        "validation": (
            pd.Timestamp(dates[split_indices["val"][0]]).strftime("%Y-%m-%d"),
            pd.Timestamp(dates[split_indices["val"][1] - 1]).strftime("%Y-%m-%d"),
        ),
        "test": (
            pd.Timestamp(dates[split_indices["test"][0]]).strftime("%Y-%m-%d"),
            pd.Timestamp(dates[split_indices["test"][1] - 1]).strftime("%Y-%m-%d"),
        ),
    }
    return features, prices, dates, split_indices, norm_params, ranges


@torch.no_grad()
def predict_from_checkpoint(
    model_name: str,
    horizon: int,
    seed: int,
    cfg: Config,
    in_features: int,
    test_loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = build_model(model_name, in_features, cfg).to(device)
    if model_name != "Naive":
        ckpt = os.path.join(cfg.checkpoint_dir, f"{model_name.lower()}_h{horizon}_s{seed}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
    model.eval()

    pred_list, true_list, base_list = [], [], []
    for batch in test_loader:
        x_b = batch[0].to(device)
        y_hat = model(x_b)
        pred_list.append(y_hat.cpu().numpy())
        true_list.append(batch[1].numpy())
        base_list.append(batch[2].numpy())
    return (
        np.concatenate(pred_list, axis=0),
        np.concatenate(true_list, axis=0),
        np.concatenate(base_list, axis=0),
    )


def target_dates_for_test(dates: np.ndarray, prices: np.ndarray, split_indices: Dict,
                          window_size: int, horizon: int) -> np.ndarray:
    start_idx, end_idx = split_indices["test"]
    valid_start = max(start_idx, window_size)
    valid_end = min(end_idx, len(prices) - horizon)
    sample_indices = np.arange(valid_start, valid_end)
    return pd.to_datetime(dates[sample_indices + horizon - 1])


def style_date_axis(ax) -> None:
    locator = mdates.MonthLocator(interval=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Date")


def plot_single_prediction(model_name: str, horizon: int, dates: np.ndarray,
                           actual: np.ndarray, pred: np.ndarray,
                           test_period: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, actual, label="Actual price", color="#111111", linewidth=1.8)
    ax.plot(
        dates,
        pred,
        label=f"{MODEL_LABELS[model_name]} predicted price",
        color=MODEL_COLORS[model_name],
        linewidth=1.4,
        linestyle="--",
    )
    style_date_axis(ax)
    ax.set_ylabel("Copper price (USD/lb)")
    ax.set_title(f"{MODEL_LABELS[model_name]} prediction vs actual, h={horizon}\nTest period: {test_period}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"publication_pred_{model_name}_h{horizon}"))


def plot_all_predictions(horizon: int, dates: np.ndarray, actual: np.ndarray,
                         predictions: Dict[str, np.ndarray],
                         test_period: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(dates, actual, label="Actual price", color="#111111", linewidth=2.0)
    for name, pred in predictions.items():
        ax.plot(
            dates,
            pred,
            label=MODEL_LABELS[name],
            color=MODEL_COLORS[name],
            linewidth=1.2 if name != "MDHGNN" else 1.6,
            linestyle="-" if name == "MDHGNN" else "--",
            alpha=0.9,
        )
    style_date_axis(ax)
    ax.set_ylabel("Copper price (USD/lb)")
    ax.set_title(f"Test-set copper price predictions, h={horizon}\nTest period: {test_period}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"publication_all_preds_h{horizon}"))


def plot_metric_bar(horizon: int, metric: str, results: Dict, out_dir: str,
                    model_names: List[str]) -> None:
    values = [results[name][metric] for name in model_names]
    std_values = [results[name].get("std", {}).get(metric, 0.0) for name in model_names]
    labels = [MODEL_LABELS[name] for name in model_names]
    colors = [MODEL_COLORS[name] for name in model_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_names))
    bars = ax.bar(x, values, yerr=std_values, capsize=4, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"Model comparison on test set, h={horizon}")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, f"publication_compare_{metric}_h{horizon}"))


def main() -> None:
    cfg = Config()
    cfg.apply_frequency("weekly")
    model_names = get_default_model_names()
    horizons = [1, 2]
    seed_for_curves = 456
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, prices, dates, split_indices, _, ranges = build_weekly_dataset(cfg)
    in_features = features.shape[1]
    out_dir = cfg.figure_dir

    manifest = {
        "script": "experiments/redraw_publication_figures.py",
        "training": "No training was performed. Existing checkpoints were loaded for test-set inference.",
        "seed_for_prediction_curves": seed_for_curves,
        "frequency": "weekly",
        "date_ranges": ranges,
        "figures": [],
    }

    for horizon in horizons:
        cfg.horizon = horizon
        _, _, test_loader = create_dataloaders(
            features, prices, split_indices, cfg.window_size,
            cfg.horizon, cfg.batch_size
        )
        target_dates = target_dates_for_test(dates, prices, split_indices, cfg.window_size, horizon)
        test_period = f"{target_dates[0].strftime('%Y-%m-%d')} to {target_dates[-1].strftime('%Y-%m-%d')}"

        predictions = {}
        actual_prices = None
        for name in model_names:
            pred_ret, true_ret, base_px = predict_from_checkpoint(
                name, horizon, seed_for_curves, cfg, in_features, test_loader, device
            )
            pred_prices, true_prices = returns_to_prices(pred_ret, true_ret, base_px)
            predictions[name] = pred_prices
            if actual_prices is None:
                actual_prices = true_prices
            plot_single_prediction(name, horizon, target_dates, actual_prices,
                                   pred_prices, test_period, out_dir)
            manifest["figures"].append(f"publication_pred_{name}_h{horizon}.png")

        plot_all_predictions(horizon, target_dates, actual_prices, predictions, test_period, out_dir)
        manifest["figures"].append(f"publication_all_preds_h{horizon}.png")

        result_path = os.path.join(cfg.table_dir, f"results_h{horizon}.json")
        with open(result_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        for metric in ["RMSE", "DA", "Sharpe"]:
            plot_metric_bar(horizon, metric, results, out_dir, model_names)
            manifest["figures"].append(f"publication_compare_{metric}_h{horizon}.png")

    os.makedirs(cfg.table_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.table_dir, "publication_figure_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {manifest_path}")
    print("Redrew publication figures without retraining.")


if __name__ == "__main__":
    main()
