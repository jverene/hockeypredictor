"""Evaluation metrics (PRD §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # scipy ships with scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray, baseline: pd.Series) -> float:
    """% of players whose up/down move vs `baseline` (last-season PPG) was right."""
    actual_dir = np.sign(y_true - baseline)
    pred_dir = np.sign(pd.Series(y_pred, index=y_true.index) - baseline)
    return float((actual_dir == pred_dir).mean())


def spearman(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Rank correlation between predicted and actual."""
    return float(spearmanr(y_true, y_pred).statistic)


def season_metrics(df: pd.DataFrame, pred_col: str = "pred_ppg", target_col: str = "ppg_next") -> dict:
    ev = df.dropna(subset=[target_col])
    out = {
        "n": len(ev),
        "mae": mae(ev[target_col], ev[pred_col]),
        "rmse": rmse(ev[target_col], ev[pred_col]),
        "r2": r2(ev[target_col], ev[pred_col]) if len(ev) > 1 else np.nan,
        "mae_baseline": mae(ev[target_col], ev["ppg_last1"]) if "ppg_last1" in ev else np.nan,
    }
    if "ppg_last1" in ev:
        out["directional_accuracy"] = directional_accuracy(ev[target_col], ev[pred_col], ev["ppg_last1"])
    return out
