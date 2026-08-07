"""Evaluation metrics (PRD §7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # scipy ships with scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


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


def interval_coverage(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Share of actuals inside [lower, upper]; a calibrated 80% interval → ~0.80."""
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def breakout_auc(
    df: pd.DataFrame,
    score_col: str,
    delta_col: str = "delta",
    threshold: float = 0.3,
) -> tuple[float, int]:
    """AUC of a risk score for detecting breakouts (or slumps).

    `delta` = actual minus last-season PPG. A breakout is delta >= +threshold
    (for slumps pass the negated delta). The score is typically the model's
    upside spread (q90 - q50) or downside spread (q50 - q10). Returns
    (AUC, n_events). AUC 0.5 = no skill; >0.6 = real signal.
    """
    events = df[delta_col] >= threshold
    n = int(events.sum())
    if n < 10 or n == len(df):
        return float("nan"), n
    return float(roc_auc_score(events, df[score_col])), n


def precision_at_k(df: pd.DataFrame, score_col: str, delta_col: str = "delta",
                   threshold: float = 0.3, k: int = 50) -> float:
    """Of the K highest-risk-score players, how many actually broke out/slumped?"""
    top = df.nlargest(min(k, len(df)), score_col)
    return float((top[delta_col] >= threshold).mean())
