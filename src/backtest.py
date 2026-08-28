"""Rolling walk-forward backtest engine (PRD §6).

Veteran: for each target season Y (1990-91 .. 2025-26), train on all
(player-season -> next-season) pairs whose *target* season falls in the 5
seasons strictly before Y, then predict season Y for everyone who played in
Y-1. Rookie: for each draft class D (1990..2024), train on pairs from earlier
classes only.

Leakage guards:
  - training targets are always strictly before the prediction season
  - the cancelled 2004-05 season is skipped by season arithmetic
  - shortened seasons are flagged, not skipped
  - players with < MIN_TARGET_GP games in the target season are excluded from
    evaluation but kept in training
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    FIRST_BACKTEST_SEASON,
    LAST_BACKTEST_SEASON,
    MIN_TARGET_GP,
    SHORTENED_SEASONS,
    UNDRAFTED_OVR,
    season_label,
    season_window,
)
from .features import ROOKIE_FEATURES, VETERAN_FEATURES
from .metrics import mae, r2, rmse, spearman, directional_accuracy, interval_coverage
from .model import predict, train_model

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "output"

ROOKIE_FIRST_CLASS = 1990
ROOKIE_LAST_CLASS = 2024


def _seasons_before(season_id: int, n: int) -> list[int]:
    """The n seasons strictly preceding `season_id`."""
    all_seasons = season_window(season_id - n * 10000 - 20000, season_id)
    return [s for s in all_seasons if s < season_id][-n:]


def run_veteran_backtest(
    veterans: pd.DataFrame,
    first: int = FIRST_BACKTEST_SEASON,
    last: int = LAST_BACKTEST_SEASON,
    window: int | None = 5,
    recency_decay: float | None = None,
    quantiles: tuple[float, ...] = (),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward veteran backtest. Returns (predictions, per-season metrics).

    window=None trains on all history before Y (expanding window); an int uses
    a rolling window of that many seasons. recency_decay (e.g. 0.85) applies
    exponential sample weights by target-season age: decay ** years_before_Y.
    quantiles (e.g. (0.1, 0.9)) additionally trains pinball-loss models per
    season and adds pred_ppg_qXX columns — the q90−q50 spread is the model's
    breakout-risk score, q50−q10 its slump-risk score.
    """
    predictions: list[pd.DataFrame] = []
    season_rows: list[dict] = []

    for Y in season_window(first, last):
        if window is None:
            train_mask = veterans["next_season_actual"] < Y
        else:
            train_targets = _seasons_before(Y, window)
            train_mask = veterans["next_season_actual"].isin(train_targets)
        train = veterans[train_mask]
        test = veterans[veterans["next_season_actual"] == Y]
        if train.empty or test.empty:
            continue

        weights = None
        if recency_decay is not None:
            years_back = (Y - train["next_season_actual"]) / 10001.0
            weights = recency_decay ** years_back

        trained = train_model(
            train,
            VETERAN_FEATURES,
            "ppg_next",
            order_col="next_season_actual",
            **({"sample_weight": weights} if weights is not None else {}),
        )
        test = test.copy()
        test["pred_ppg"] = predict(trained, test)
        test["pred_season"] = Y  # the season being predicted (seasonId is the feature season)

        for q in quantiles:
            qtrained = train_model(
                train,
                VETERAN_FEATURES,
                "ppg_next",
                order_col="next_season_actual",
                quantile=q,
                **({"sample_weight": weights} if weights is not None else {}),
            )
            test[f"pred_ppg_q{int(round(q * 100))}"] = predict(qtrained, test)

        # Independent quantile models can cross; enforce monotonicity per row
        # (pool-adjacent-violators style: sorting the row is a valid rearrangement).
        qcols = [f"pred_ppg_q{int(round(q * 100))}" for q in sorted(quantiles)]
        if len(qcols) > 1:
            test[qcols] = np.sort(test[qcols].to_numpy(), axis=1)

        ev = test[test["eval_ok"] & test["ppg_next"].notna()]
        row = {
            "seasonId": Y,
            "season": season_label(Y),
            "shortened": SHORTENED_SEASONS.get(Y, ""),
            "n_eval": len(ev),
            "n_train": len(train),
            "best_iteration": trained.best_iteration,
        }
        if len(ev) > 1:
            row.update(
                mae=mae(ev["ppg_next"], ev["pred_ppg"]),
                rmse=rmse(ev["ppg_next"], ev["pred_ppg"]),
                r2=r2(ev["ppg_next"], ev["pred_ppg"]),
                mae_baseline=mae(ev["ppg_next"], ev["ppg_last1"]),
                directional_accuracy=directional_accuracy(
                    ev["ppg_next"], ev["pred_ppg"], ev["ppg_last1"]
                ),
                top_features=", ".join(list(trained.feature_importance)[:5]),
            )
        season_rows.append(row)
        pred_cols = ["playerId", "skaterFullName", "seasonId", "pred_season", "ppg_last1",
                     "pred_ppg", "ppg_next", "gp_next", "eval_ok"] + [
            c for c in test.columns if c.startswith("pred_ppg_q")
        ]
        predictions.append(test[pred_cols])
        if verbose:
            print(
                f"{season_label(Y)}: n={row['n_eval']:4d} "
                f"MAE={row.get('mae', float('nan')):.4f} "
                f"(baseline {row.get('mae_baseline', float('nan')):.4f})"
            )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(season_rows)


def _draft_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Draft-position expectation baseline (PRD §5.1): mean rookie PPG of
    each draft bucket among *earlier* classes only — leak-free."""
    bins = [0, 5, 10, 31, 62, 100, UNDRAFTED_OVR]
    labels = ["1-5", "6-10", "11-31", "32-62", "63-100", "101+"]
    base = train.dropna(subset=["ppg_rookie"])
    means = base.groupby(pd.cut(base["draft_ovr"], bins=bins, labels=labels),
                         observed=True)["ppg_rookie"].mean()
    global_mean = base["ppg_rookie"].mean()
    bucket = pd.cut(test["draft_ovr"], bins=bins, labels=labels)
    return bucket.map(means).fillna(global_mean)


def run_rookie_backtest(
    rookies: pd.DataFrame,
    first_class: int = ROOKIE_FIRST_CLASS,
    last_class: int = ROOKIE_LAST_CLASS,
    window: int | None = None,
    recency_decay: float | None = None,
    quantiles: tuple[float, ...] = (),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward rookie backtest by draft class. Returns (predictions, metrics).

    window=None trains on all earlier classes (expanding); an int restricts
    training to rookies who debuted within that many years of the class.
    recency_decay applies exponential sample weights by debut-season age.
    quantiles additionally train pinball-loss models per class and add
    pred_ppg_qXX columns (q90−q50 spread = upside score for steals lists).
    """
    df = rookies.copy()
    df["class_year"] = df["draft_year"].fillna(df["rookie_season"] // 10000)

    predictions: list[pd.DataFrame] = []
    class_rows: list[dict] = []

    for D in range(first_class, last_class + 1):
        target_season = D * 10000 + D + 1
        train = df[df["class_year"] < D]
        if window is not None:
            train = train[train["rookie_season"] // 10000 > D - 1 - window]
        test = df[df["class_year"] == D]
        if len(train) < 30 or test.empty:
            continue

        weights = None
        if recency_decay is not None:
            years_back = (target_season - train["rookie_season"]) / 10001.0
            weights = recency_decay ** years_back

        trained = train_model(
            train,
            ROOKIE_FEATURES,
            "ppg_rookie",
            order_col="rookie_season",
            **({"sample_weight": weights} if weights is not None else {}),
        )
        test = test.copy()
        test["pred_ppg"] = predict(trained, test)
        test["pred_baseline"] = _draft_baseline(train, test)

        for q in quantiles:
            qtrained = train_model(
                train,
                ROOKIE_FEATURES,
                "ppg_rookie",
                order_col="rookie_season",
                quantile=q,
                **({"sample_weight": weights} if weights is not None else {}),
            )
            test[f"pred_ppg_q{int(round(q * 100))}"] = predict(qtrained, test)

        # Enforce monotone quantile curves per row (rearrangement).
        qcols = [f"pred_ppg_q{int(round(q * 100))}" for q in sorted(quantiles)]
        if len(qcols) > 1:
            test[qcols] = np.sort(test[qcols].to_numpy(), axis=1)

        ev = test[test["ppg_rookie"].notna()]
        row = {
            "draft_class": D,
            "n_eval": len(test),
            "n_train": len(train),
            "mae": mae(test["ppg_rookie"], test["pred_ppg"]),
            "mae_baseline": mae(test["ppg_rookie"], test["pred_baseline"]),
            "rmse": rmse(test["ppg_rookie"], test["pred_ppg"]),
            "spearman": spearman(test["ppg_rookie"], test["pred_ppg"]),
            "top_features": ", ".join(list(trained.feature_importance)[:5]),
        }
        if len(ev) > 1 and len(qcols) >= 2:
            row["interval_coverage_80"] = interval_coverage(
                ev["ppg_rookie"],
                ev[f"pred_ppg_q{int(round(sorted(quantiles)[0] * 100))}"],
                ev[f"pred_ppg_q{int(round(sorted(quantiles)[-1] * 100))}"],
            )
        class_rows.append(row)
        pred_cols = [
            "playerId", "skaterFullName", "draft_year", "draft_ovr", "rookie_season",
            "age", "pred_ppg", "pred_baseline", "ppg_rookie",
        ] + qcols
        predictions.append(test[pred_cols])
        if verbose:
            print(
                f"draft {D}: n={row['n_eval']:3d} MAE={row['mae']:.4f} "
                f"(baseline {row['mae_baseline']:.4f}) rho={row['spearman']:.3f}"
            )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(class_rows)


def build_dataset(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full data -> features pipeline; returns (player_seasons, veteran_rows, rookie_rows)."""
    from .data import (
        fetch_all_bios,
        fetch_team_meta,
        load_prenhl_stats,
        load_skater_seasons,
        load_team_seasons,
    )
    from .features import (
        attach_team_quality,
        build_player_seasons,
        build_rookie_features,
        build_veteran_features,
    )

    skaters = load_skater_seasons(refresh=refresh)
    teams = load_team_seasons(refresh=refresh)
    team_meta = fetch_team_meta(refresh=refresh)
    bios = fetch_all_bios(sorted(skaters["playerId"].unique().tolist()), refresh=refresh)

    player_seasons = build_player_seasons(skaters, bios)
    player_seasons = attach_team_quality(player_seasons, skaters, teams, team_meta)

    veterans = build_veteran_features(player_seasons)
    rookies = build_rookie_features(player_seasons, load_prenhl_stats())
    return player_seasons, veterans, rookies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the walk-forward backtests.")
    parser.add_argument("--refresh", action="store_true", help="re-pull NHL data")
    parser.add_argument("--veterans-only", action="store_true")
    parser.add_argument("--rookies-only", action="store_true")
    parser.add_argument("--rookie-window", type=int, default=None,
                        help="restrict rookie training to debuts within N years of the class")
    parser.add_argument("--rookie-decay", type=float, default=None,
                        help="exponential recency weight per season for rookie training")
    parser.add_argument("--quantiles", action="store_true",
                        help="also train q10/q50/q90 pinball-loss models (tail predictions)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    player_seasons, veterans, rookies = build_dataset(refresh=args.refresh)
    player_seasons.to_parquet(OUTPUT_DIR / "player_seasons.parquet", index=False)
    veterans.to_parquet(OUTPUT_DIR / "veteran_features.parquet", index=False)
    rookies.to_parquet(OUTPUT_DIR / "rookie_features.parquet", index=False)

    if not args.rookies_only:
        q = (0.1, 0.5, 0.9) if args.quantiles else ()
        preds, seasons = run_veteran_backtest(veterans, quantiles=q)
        preds.to_parquet(OUTPUT_DIR / "veteran_predictions.parquet", index=False)
        seasons.to_csv(OUTPUT_DIR / "veteran_season_metrics.csv", index=False)
        ok = seasons.dropna(subset=["mae"])
        print(f"\nVeteran backtest: MAE={ok['mae'].mean():.4f} "
              f"(baseline {ok['mae_baseline'].mean():.4f}), "
              f"dir.acc={ok['directional_accuracy'].mean():.3f}")

    if not args.veterans_only:
        q = (0.1, 0.5, 0.9) if args.quantiles else ()
        preds, classes = run_rookie_backtest(
            rookies,
            window=args.rookie_window,
            recency_decay=args.rookie_decay,
            quantiles=q,
        )
        preds.to_parquet(OUTPUT_DIR / "rookie_predictions.parquet", index=False)
        classes.to_csv(OUTPUT_DIR / "rookie_class_metrics.csv", index=False)
        ok = classes.dropna(subset=["mae"])
        print(f"\nRookie backtest: MAE={ok['mae'].mean():.4f} "
              f"(draft baseline {ok['mae_baseline'].mean():.4f}), "
              f"Spearman rho={ok['spearman'].mean():.3f}, "
              f"classes rho>0.55: {(ok['spearman'] > 0.55).sum()}/{len(ok)}")
        if "interval_coverage_80" in ok:
            print(f"80% interval coverage: {ok['interval_coverage_80'].mean():.3f}")


if __name__ == "__main__":
    main()
