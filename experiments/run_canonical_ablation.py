"""Compatibility entry for the reported ablation protocol.

The project reports masked ablation: train the Full MDHGNN once per seed, then
mask one channel/path at inference. Keeping this filename avoids breaking older
commands while ensuring the generated official ablation tables use the same
protocol as README.docx.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_masked_ablation import run_masked_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", default="weekly", choices=["daily", "weekly"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--no-external", action="store_true")
    args = parser.parse_args()

    summary = run_masked_ablation(
        horizon=args.horizon,
        seeds=args.seeds,
        no_external=args.no_external,
        frequency=args.frequency,
    )

    print("\nOfficial masked ablation summary")
    for variant, values in summary.items():
        rmse = values["RMSE"]
        print(f"{variant:<22} RMSE={rmse['mean']:.4f}+/-{rmse['std']:.4f}")


if __name__ == "__main__":
    main()
