"""Tail evaluation: do the quantile models actually catch breakouts and slumps?

Definitions (delta = actual PPG - last-season PPG, evaluated players only):
  breakout  — delta >= +0.30 PPG
  slump     — delta <= -0.30 PPG

Risk scores from the quantile models:
  upside    — q90 - q50 (how much room above the median the model sees)
  downside  — q50 - q10

Metrics: 80% interval coverage (calibration), AUC of each score for its event,
precision@50 per season. Run after `python -m src.backtest --quantiles`.
"""

from __future__ import annotations

import pandas as pd

from .backtest import OUTPUT_DIR
from .constants import season_label
from .metrics import breakout_auc, interval_coverage, precision_at_k

BREAKOUT_THRESHOLD = 0.30


def evaluate_tails(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    if preds is None:
        preds = pd.read_parquet(OUTPUT_DIR / "veteran_predictions.parquet")
    ev = preds[preds["eval_ok"]].dropna(subset=["ppg_next"]).copy()
    ev["season"] = ev["pred_season"].map(season_label)
    ev["delta"] = ev["ppg_next"] - ev["ppg_last1"]
    ev["upside"] = ev["pred_ppg_q90"] - ev["pred_ppg_q50"]
    ev["downside"] = ev["pred_ppg_q50"] - ev["pred_ppg_q10"]
    ev["neg_delta"] = -ev["delta"]

    coverage = interval_coverage(ev["ppg_next"], ev["pred_ppg_q10"], ev["pred_ppg_q90"])
    auc_up, n_break = breakout_auc(ev, "upside", "delta", BREAKOUT_THRESHOLD)
    auc_down, n_slump = breakout_auc(ev, "downside", "neg_delta", BREAKOUT_THRESHOLD)

    # Also: does a naive "young players break out" score beat the model?
    print(f"evaluated player-seasons : {len(ev)}")
    print(f"80% interval coverage    : {coverage:.3f} (target 0.80)")
    print(f"breakouts (>= +{BREAKOUT_THRESHOLD} PPG) : {n_break}  upside AUC = {auc_up:.3f}")
    print(f"slumps    (<= -{BREAKOUT_THRESHOLD} PPG) : {n_slump}  downside AUC = {auc_down:.3f}")

    per_season = []
    for season, g in ev.groupby("season"):
        per_season.append(
            {
                "season": season,
                "p@50_breakout": precision_at_k(g, "upside", "delta", BREAKOUT_THRESHOLD, 50),
                "p@50_slump": precision_at_k(g, "downside", "neg_delta", BREAKOUT_THRESHOLD, 50),
                "base_rate_breakout": (g["delta"] >= BREAKOUT_THRESHOLD).mean(),
                "base_rate_slump": (g["neg_delta"] >= BREAKOUT_THRESHOLD).mean(),
            }
        )
    ps = pd.DataFrame(per_season)
    print(f"\nprecision@50 breakouts: {ps['p@50_breakout'].mean():.3f} "
          f"(base rate {ps['base_rate_breakout'].mean():.3f})")
    print(f"precision@50 slumps   : {ps['p@50_slump'].mean():.3f} "
          f"(base rate {ps['base_rate_slump'].mean():.3f})")

    cols = ["skaterFullName", "season", "ppg_last1", "pred_ppg_q50", "pred_ppg_q90", "ppg_next", "delta"]
    print("\nHighest-upside calls that delivered (top 10 by upside among actual breakouts):")
    hits = ev[ev["delta"] >= BREAKOUT_THRESHOLD].nlargest(10, "upside")
    print(hits[cols].to_string(index=False))

    out = ev[["playerId", "skaterFullName", "season", "ppg_last1", "pred_ppg",
              "pred_ppg_q10", "pred_ppg_q50", "pred_ppg_q90", "ppg_next", "delta",
              "upside", "downside"]]
    out.to_csv(OUTPUT_DIR / "tail_predictions.csv", index=False)
    ps.to_csv(OUTPUT_DIR / "tail_metrics_by_season.csv", index=False)
    return ev


if __name__ == "__main__":
    evaluate_tails()
