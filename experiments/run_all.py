"""多模型多种子对比实验"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json

from config import Config
from utils.reproducibility import set_all_seeds
from utils.metrics import (compute_metrics, print_metrics,
                           compute_return_metrics, print_return_metrics,
                           compute_strategy_return, diebold_mariano_test)
from utils.visualization import (plot_predictions, plot_loss_curves,
                                  plot_comparison_bar, plot_all_predictions,
                                  plot_multi_horizon_comparison)
from data.data_loader import load_copper_data
from data.external_data import download_external_data
from data.feature_engineering import add_technical_indicators
from data.resampling import resample_copper_ohlcv, resample_external_prices
from data.preprocessing import preprocess_data
from data.dataset import create_dataloaders
from models.mdhgnn import MDHGNN
from models.baselines.mlp_baseline import MLPBaseline
from models.baselines.lstm_baseline import LSTMBaseline
from models.baselines.gru_baseline import GRUBaseline
from models.baselines.tcn_baseline import TCNBaseline
from models.baselines.transformer_baseline import TransformerBaseline
from models.baselines.naive_baseline import NaiveBaseline
from trainers.trainer import Trainer


def get_default_model_names():
    return ['MDHGNN', 'GRU', 'MLP', 'Transformer', 'Naive']


def returns_to_prices(pred_returns, true_returns, base_prices):
    pred_r = pred_returns.reshape(pred_returns.shape[0], -1)
    true_r = true_returns.reshape(true_returns.shape[0], -1)
    base = base_prices.flatten()
    pred_prices = base * (1 + pred_r[:, -1])
    true_prices = base * (1 + true_r[:, -1])
    return pred_prices, true_prices


def build_model(name, in_features, cfg):
    if name == 'MDHGNN':
        return MDHGNN(
            in_features=in_features, d_model=cfg.d_model,
            window_size=cfg.window_size, n_hgnn_layers=cfg.n_hgnn_layers,
            k_neigs=cfg.k_neigs,
            n_adaptive_edges=cfg.n_adaptive_edges, gru_hidden=cfg.gru_hidden,
            horizon=cfg.horizon, dropout=cfg.dropout
        )
    elif name == 'LSTM':
        return LSTMBaseline(in_features, hidden_size=64, horizon=cfg.horizon, dropout=cfg.dropout)
    elif name == 'GRU':
        return GRUBaseline(in_features, hidden_size=64, horizon=cfg.horizon, dropout=cfg.dropout)
    elif name == 'MLP':
        return MLPBaseline(in_features, window_size=cfg.window_size,
                           hidden_size=128, horizon=cfg.horizon, dropout=cfg.dropout)
    elif name == 'TCN':
        return TCNBaseline(in_features, hidden_size=64, horizon=cfg.horizon, dropout=cfg.dropout)
    elif name == 'Transformer':
        return TransformerBaseline(in_features, d_model=64, window_size=cfg.window_size,
                                   horizon=cfg.horizon, dropout=cfg.dropout)
    elif name == 'Naive':
        return NaiveBaseline(horizon=cfg.horizon)
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate_model(model, device, test_loader):
    """推理，返回 (pred_returns, true_returns, base_prices)"""
    model.eval()
    preds_list, targets_list, bases_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_b = batch[0].to(device)
            y_hat = model(x_b)
            preds_list.append(y_hat.cpu().numpy())
            targets_list.append(batch[1].numpy())
            bases_list.append(batch[2].numpy())
    return (np.concatenate(preds_list), np.concatenate(targets_list),
            np.concatenate(bases_list))


def compute_dm_results(all_errors, model_names, horizon, reference='MDHGNN'):
    def flatten_errors(value):
        if value is None:
            return None
        if isinstance(value, list):
            if len(value) == 0:
                return None
            return np.concatenate(value)
        return np.asarray(value).flatten()

    reference_errors = flatten_errors(all_errors.get(reference))
    if reference_errors is None or len(reference_errors) == 0:
        return {}

    dm_results = {}
    for name in model_names:
        if name == reference:
            continue
        other_errors = flatten_errors(all_errors.get(name))
        if other_errors is None or len(other_errors) == 0:
            continue
        min_len = min(len(reference_errors), len(other_errors))
        dm = diebold_mariano_test(
            reference_errors[:min_len], other_errors[:min_len], horizon
        )
        dm_results[name] = {
            'DM_stat': float(dm['DM_stat']),
            'p_value': float(dm['p_value']),
            'n_errors': int(min_len),
            'direction': (
                f'{reference} better' if dm['DM_stat'] < 0 else f'{name} better'
            ),
        }
    return dm_results


def run_single_horizon(horizon, seeds, cfg, features, prices, split_indices,
                       in_features, device, model_names):
    """单个horizon下跑所有模型，返回汇总指标"""
    cfg.horizon = horizon
    all_price = {n: [] for n in model_names}
    all_ret = {n: [] for n in model_names}
    all_strat = {n: [] for n in model_names}
    all_errors = {n: [] for n in model_names}  # for DM test
    all_preds = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}, Horizon: {horizon}")
        print(f"{'='*60}")

        for name in model_names:
            set_all_seeds(seed)
            train_loader, val_loader, test_loader = create_dataloaders(
                features, prices, split_indices, cfg.window_size, cfg.horizon, cfg.batch_size
            )

            model = build_model(name, in_features, cfg)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n--- {name} (params: {n_params:,}) ---")

            if name == 'Naive':
                model = model.to(device)
                pred_ret, true_ret, base_px = evaluate_model(model, device, test_loader)
                if seed == seeds[-1]:
                    pred_prices, true_prices = returns_to_prices(pred_ret, true_ret, base_px)
                    all_preds[name] = pred_prices
                    plot_predictions(true_prices, pred_prices, name,
                                     save_path=os.path.join(cfg.figure_dir, f'pred_{name}_h{horizon}.png'))
            else:
                trainer = Trainer(model, device, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  grad_clip=cfg.grad_clip, patience=cfg.patience,
                                  checkpoint_dir=cfg.checkpoint_dir)
                trainer.train(train_loader, val_loader,
                              max_epochs=cfg.max_epochs, print_freq=50)
                pred_ret, true_ret, base_px = trainer.predict(test_loader)
                trainer.save_checkpoint(f'{name.lower()}_h{horizon}_s{seed}')

                if seed == seeds[-1]:
                    pred_prices, true_prices = returns_to_prices(pred_ret, true_ret, base_px)
                    all_preds[name] = pred_prices
                    plot_predictions(true_prices, pred_prices, name,
                                     save_path=os.path.join(cfg.figure_dir, f'pred_{name}_h{horizon}.png'))

            pred_prices, true_prices = returns_to_prices(pred_ret, true_ret, base_px)

            price_m = compute_metrics(true_prices, pred_prices)
            ret_m = compute_return_metrics(pred_ret, true_ret)
            strat_m = compute_strategy_return(pred_ret, true_ret,
                                               trading_periods_per_year=cfg.trading_periods_per_year)

            # Merge strategy into return metrics for printing
            ret_m.update(strat_m)
            print_metrics(price_m, name)
            print_return_metrics(ret_m, name)

            all_price[name].append(price_m)
            all_ret[name].append(ret_m)
            all_strat[name].append(strat_m)
            all_errors[name].append((true_ret.flatten() - pred_ret.flatten()))

    # ── Compute mean ± std ──
    def summarize(metrics_list, keys):
        mean_d, std_d = {}, {}
        for k in keys:
            vals = [m[k] for m in metrics_list]
            mean_d[k] = float(np.mean(vals))
            std_d[k] = float(np.std(vals))
        return mean_d, std_d

    price_keys = ['RMSE', 'MAE', 'MAPE', 'R2']
    ret_keys = ['Ret_RMSE', 'Ret_R2', 'DA', 'DA_up', 'DA_down']
    strat_keys = ['Sharpe', 'AnnReturn', 'MaxDrawdown']

    # ── Print price-level ──
    print(f"\n{'='*90}")
    print(f"PRICE-LEVEL RESULTS (h={horizon}, {len(seeds)} seeds)")
    print(f"{'='*90}")
    print(f"{'Model':<15} {'RMSE':>14} {'MAE':>14} {'MAPE(%)':>14} {'R2':>14}")
    print('-' * 75)

    price_summary = {}
    for name in model_names:
        m, s = summarize(all_price[name], price_keys)
        print(f"{name:<15} {m['RMSE']:>6.4f}±{s['RMSE']:.4f}  "
              f"{m['MAE']:>6.4f}±{s['MAE']:.4f}  "
              f"{m['MAPE']:>6.2f}±{s['MAPE']:.2f}  "
              f"{m['R2']:>6.4f}±{s['R2']:.4f}")
        price_summary[name] = {**m, 'std': s}

    # ── Print return-level ──
    print(f"\n{'='*90}")
    print(f"RETURN-LEVEL RESULTS (h={horizon}, {len(seeds)} seeds)")
    print(f"{'='*90}")
    print(f"{'Model':<15} {'Ret_RMSE(bp)':>14} {'Ret_R2':>10} {'DA(%)':>10} {'Sharpe':>10} {'AnnRet(%)':>12}")
    print('-' * 75)

    ret_summary = {}
    for name in model_names:
        rm, rs = summarize(all_ret[name], ret_keys)
        sm, ss = summarize(all_strat[name], strat_keys)
        combined = {**rm, **sm}
        print(f"{name:<15} {rm['Ret_RMSE']:>8.2f}±{rs.get('Ret_RMSE',0):.2f}  "
              f"{rm['Ret_R2']:>7.4f}  "
              f"{rm['DA']:>6.1f}±{rs.get('DA',0):.1f}  "
              f"{sm['Sharpe']:>7.3f}  "
              f"{sm['AnnReturn']:>8.2f}±{ss.get('AnnReturn',0):.2f}")
        ret_summary[name] = {**combined, 'std': {**rs, **ss}}

    # ── DM test: MDHGNN vs each baseline ──
    dm_results = {}
    if 'MDHGNN' in all_errors and len(seeds) > 0:
        print(f"\n{'='*60}")
        print(f"DIEBOLD-MARIANO TEST: MDHGNN vs baselines (h={horizon})")
        print(f"{'='*60}")
        dm_results = compute_dm_results(all_errors, model_names, horizon)
        for name, dm in dm_results.items():
            sig = '***' if dm['p_value'] < 0.01 else '**' if dm['p_value'] < 0.05 else '*' if dm['p_value'] < 0.1 else ''
            print(f"  vs {name:<15} DM={dm['DM_stat']:>7.3f}  p={dm['p_value']:.4f} {sig:>3}  ({dm['direction']})")

    # ── Save figures ──
    os.makedirs(cfg.figure_dir, exist_ok=True)
    plot_comparison_bar(ret_summary, 'DA',
                        save_path=os.path.join(cfg.figure_dir, f'compare_DA_h{horizon}.png'))
    plot_comparison_bar(ret_summary, 'Sharpe',
                        save_path=os.path.join(cfg.figure_dir, f'compare_Sharpe_h{horizon}.png'))
    if all_preds and true_prices is not None:
        plot_all_predictions(all_preds, true_prices,
                             save_path=os.path.join(cfg.figure_dir, f'all_preds_h{horizon}.png'))

    # ── Save JSON ──
    os.makedirs(cfg.table_dir, exist_ok=True)
    combined_summary = {}
    for name in model_names:
        price_data = price_summary.get(name, {})
        ret_data = ret_summary.get(name, {})
        price_std = price_data.get('std', {})
        ret_std = ret_data.get('std', {})
        combined_summary[name] = {
            **{k: v for k, v in price_data.items() if k != 'std'},
            **{k: v for k, v in ret_data.items() if k != 'std'},
            'std': {**price_std, **ret_std},
        }
    with open(os.path.join(cfg.table_dir, f'results_h{horizon}.json'), 'w') as f:
        json.dump(combined_summary, f, indent=2)
    with open(os.path.join(cfg.table_dir, f'dm_h{horizon}.json'), 'w') as f:
        json.dump(dm_results, f, indent=2)

    return combined_summary


def run_experiment(horizons=None, seeds=None, no_external=False, frequency='weekly'):
    cfg = Config()
    cfg.apply_frequency(frequency)

    if horizons is None:
        horizons = [1, 2, 4, 8] if frequency == 'weekly' else [1, 5, 10, 20]
    if seeds is None:
        seeds = [42, 123, 456]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names = get_default_model_names()

    # Load data once
    set_all_seeds(42)
    copper_df = load_copper_data(cfg.csv_path)

    if frequency == 'weekly':
        copper_df = resample_copper_ohlcv(copper_df)
        print(f"Weekly copper data: {len(copper_df)} rows")

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
    print(f"Features: {in_features}, Samples: {len(features)}")

    # Run each horizon
    multi_horizon = {}
    for h in horizons:
        print(f"\n\n{'#'*80}")
        print(f"# HORIZON = {h}")
        print(f"{'#'*80}")
        multi_horizon[h] = run_single_horizon(
            h, seeds, cfg, features, prices, split_indices,
            in_features, device, model_names
        )

    # ── Multi-horizon summary ──
    if len(horizons) > 1:
        print(f"\n\n{'#'*90}")
        print(f"# MULTI-HORIZON SUMMARY ({len(seeds)} seeds)")
        print(f"{'#'*90}")

        print(f"\n{'Model':<15}", end='')
        for h in horizons:
            print(f"  h={h:<3} DA%  Sharpe", end='')
        print()
        print('-' * (15 + len(horizons) * 18))

        for name in model_names:
            print(f"{name:<15}", end='')
            for h in horizons:
                d = multi_horizon[h].get(name, {})
                print(f"  {d.get('DA', 0):>5.1f}  {d.get('Sharpe', 0):>6.3f}", end='')
            print()

        # Multi-horizon figures
        for metric in ['DA', 'Sharpe', 'Ret_R2']:
            plot_multi_horizon_comparison(
                multi_horizon, metric, model_names,
                save_path=os.path.join(cfg.figure_dir, f'multi_horizon_{metric}.png')
            )

        # Save combined JSON
        with open(os.path.join(cfg.table_dir, 'results_multi_horizon.json'), 'w') as f:
            json.dump({str(h): multi_horizon[h] for h in horizons}, f, indent=2)

    return multi_horizon


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=str, default='weekly', choices=['daily', 'weekly'])
    parser.add_argument('--horizons', type=int, nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--no-external', action='store_true')
    args = parser.parse_args()

    run_experiment(horizons=args.horizons, seeds=args.seeds,
                   no_external=args.no_external, frequency=args.frequency)
