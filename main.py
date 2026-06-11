import argparse
import os
import sys
import torch
import numpy as np

from config import Config
from utils.reproducibility import set_all_seeds
from utils.metrics import compute_metrics, print_metrics
from utils.visualization import plot_predictions, plot_loss_curves
from data.data_loader import load_copper_data
from data.external_data import download_external_data
from data.feature_engineering import add_technical_indicators
from data.resampling import resample_copper_ohlcv, resample_external_prices
from data.preprocessing import preprocess_data
from data.dataset import create_dataloaders
from models.mdhgnn import MDHGNN
from trainers.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description='MDHGNN return forecasting with price reconstruction'
    )
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--frequency', type=str, default='weekly', choices=['daily', 'weekly'])
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--no-external', action='store_true', help='Skip external data')
    args = parser.parse_args()

    cfg = Config()
    cfg.apply_frequency(args.frequency)
    cfg.horizon = args.horizon
    cfg.seed = args.seed
    cfg.max_epochs = args.epochs

    set_all_seeds(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Frequency: {cfg.frequency}, Horizon: {cfg.horizon}, Seed: {cfg.seed}")

    # Load and prepare data
    print("\n--- Loading copper futures data ---")
    copper_df = load_copper_data(cfg.csv_path)
    print(f"Daily data: {len(copper_df)} rows, {copper_df['Date'].min()} to {copper_df['Date'].max()}")

    # Resample to weekly if needed
    if cfg.frequency == 'weekly':
        copper_df = resample_copper_ohlcv(copper_df)
        print(f"Weekly data: {len(copper_df)} rows")

    print("\n--- Adding technical indicators ---")
    copper_df = add_technical_indicators(copper_df, params=cfg.indicator_params)

    # External data
    external_df = None
    if not args.no_external:
        print("\n--- Loading external market data ---")
        start = copper_df['Date'].min().strftime('%Y-%m-%d')
        end = copper_df['Date'].max().strftime('%Y-%m-%d')
        external_df = download_external_data(start, end, cache_path=cfg.external_cache)
        if cfg.frequency == 'weekly':
            external_df = resample_external_prices(external_df)
        print(f"External data: {len(external_df)} rows")

    # Preprocess
    print("\n--- Preprocessing ---")
    features, prices, split_indices, norm_params = preprocess_data(
        copper_df, external_df, cfg.train_ratio, cfg.val_ratio
    )
    in_features = features.shape[1]
    print(f"Features shape: {features.shape}, Price target: {prices.shape}")
    print(f"Train: {split_indices['train']}, Val: {split_indices['val']}, Test: {split_indices['test']}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        features, prices, split_indices,
        cfg.window_size, cfg.horizon, cfg.batch_size
    )
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Build model
    print(f"\n--- Building MDHGNN (in_features={in_features}, d_model={cfg.d_model}) ---")
    model = MDHGNN(
        in_features=in_features,
        d_model=cfg.d_model,
        window_size=cfg.window_size,
        n_hgnn_layers=cfg.n_hgnn_layers,
        k_neigs=cfg.k_neigs,
        n_adaptive_edges=cfg.n_adaptive_edges,
        gru_hidden=cfg.gru_hidden,
        horizon=cfg.horizon,
        dropout=cfg.dropout
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Train
    if args.mode == 'train':
        print("\n--- Training MDHGNN ---")
        trainer = Trainer(
            model, device,
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip, patience=cfg.patience,
            checkpoint_dir=cfg.checkpoint_dir
        )
        history = trainer.train(train_loader, val_loader,
                                max_epochs=cfg.max_epochs, print_freq=cfg.print_freq)

        # Save checkpoint
        trainer.save_checkpoint(f'mdhgnn_h{cfg.horizon}_s{cfg.seed}')

        # Plot loss
        plot_loss_curves(history, 'MDHGNN',
                         save_path=os.path.join(cfg.figure_dir, f'loss_h{cfg.horizon}.png'))

        # Evaluate on test set (convert returns to prices)
        print("\n--- Evaluating on test set ---")
        pred_ret, true_ret, base_px = trainer.predict(test_loader)
        base = base_px.flatten()
        preds = base * (1 + pred_ret.flatten())
        targets = base * (1 + true_ret.flatten())
        metrics = compute_metrics(targets, preds)
        print_metrics(metrics, 'MDHGNN')

        # Plot predictions
        plot_predictions(targets, preds, 'MDHGNN',
                         save_path=os.path.join(cfg.figure_dir, f'pred_mdhgnn_h{cfg.horizon}.png'))

        print("\nDone!")


if __name__ == '__main__':
    main()
