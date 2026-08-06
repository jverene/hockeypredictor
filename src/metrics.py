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
