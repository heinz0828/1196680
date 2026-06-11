"""Summarize per-seed ablation JSON files."""
import json
import os

import numpy as np


def main():
    table_dir = os.path.join('results', 'tables')
    seeds = [42, 123, 456]
    data = {}

    for seed in seeds:
        path = os.path.join(table_dir, f'ablation_h1_s{seed}.json')
        with open(path, 'r') as f:
            data[seed] = json.load(f)

    variants = list(data[seeds[0]].keys())
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    summary = {}

    for variant in variants:
        summary[variant] = {}
        for metric in metrics:
            values = np.array([data[seed][variant][metric] for seed in seeds], dtype=float)
            summary[variant][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
            }

    out_path = os.path.join(table_dir, 'ablation_h1_multi_seed.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Saved: {out_path}')
    print(f"{'Variant':<22} {'RMSE':>15} {'MAE':>15} {'MAPE':>15} {'R2':>15}")
    print('-' * 85)
    for variant, metrics_dict in summary.items():
        print(
            f"{variant:<22} "
            f"{metrics_dict['RMSE']['mean']:.4f}+/-{metrics_dict['RMSE']['std']:.4f} "
            f"{metrics_dict['MAE']['mean']:.4f}+/-{metrics_dict['MAE']['std']:.4f} "
            f"{metrics_dict['MAPE']['mean']:.2f}+/-{metrics_dict['MAPE']['std']:.2f} "
            f"{metrics_dict['R2']['mean']:.4f}+/-{metrics_dict['R2']['std']:.4f}"
        )


if __name__ == '__main__':
    main()
