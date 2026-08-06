"""A/B comparison of training-window strategies for the veteran model.

Arms:
  rolling_5yr   — trailing 5-season window (PRD default)
  expanding     — all history before the target season
  expanding_w   — all history, exponential recency weights (decay 0.85/season)

Reads the cached feature matrix (data/output/veteran_features.parquet) and
writes comparison metrics + a plot to data/output/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import OUTPUT_DIR, run_veteran_backtest

ARMS = {
    "rolling_5yr": dict(window=5, recency_decay=None),
    "expanding": dict(window=None, recency_decay=None),
    "expanding_w0.85": dict(window=None, recency_decay=0.85),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training-window strategies.")
    parser.add_argument("--features", default=str(OUTPUT_DIR / "veteran_features.parquet"))
    args = parser.parse_args()

    veterans = pd.read_parquet(args.features)
    all_metrics = []
    for name, kwargs in ARMS.items():
        print(f"\n=== {name} ===")
        _, seasons = run_veteran_backtest(veterans, verbose=False, **kwargs)
        seasons = seasons.dropna(subset=["mae"]).assign(arm=name)
        all_metrics.append(seasons)
        print(f"{name}: MAE={seasons['mae'].mean():.4f} R2={seasons['r2'].mean():.3f} "
              f"dir.acc={seasons['directional_accuracy'].mean():.3f}")

    df = pd.concat(all_metrics, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "window_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, g in df.groupby("arm"):
        ax.plot(g["season"], g["mae"], marker="o", ms=3, label=name)
    ax.set_ylabel("MAE (PPG)")
    ax.set_title("Training-window A/B: veteran backtest MAE by season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "window_comparison.png", dpi=150)

    summary = df.groupby("arm")[["mae", "r2", "directional_accuracy"]].mean().round(4)
    print("\n", summary.sort_values("mae"))


if __name__ == "__main__":
    main()
