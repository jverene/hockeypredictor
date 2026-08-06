"""XGBoost training wrappers for the veteran and rookie modes (PRD §5).

Both modes share the same architecture — small, regularized gradient boosting
with early stopping on a *temporal* holdout (last season of the training
window), never a random shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)
EARLY_STOPPING_ROUNDS = 20


@dataclass
class TrainedModel:
    model: xgb.XGBRegressor
    features: list[str]
    best_iteration: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)


def _temporal_split(X: pd.DataFrame, y: pd.Series, order: pd.Series, holdout_frac: float = 0.2):
    """Hold out the most-recent slice (by `order`) for early stopping."""
    unique = np.sort(order.unique())
    n_holdout = max(1, int(round(len(unique) * holdout_frac)))
    holdout_vals = set(unique[-n_holdout:])
    mask = order.isin(holdout_vals)
    return X[~mask], y[~mask], X[mask], y[mask]


def train_model(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    order_col: str,
    params: dict | None = None,
) -> TrainedModel:
    """Train an XGBoost regressor with a temporal early-stopping holdout."""
    df = train_df.dropna(subset=[target])
    X, y, order = df[features], df[target], df[order_col]
    X_tr, y_tr, X_val, y_val = _temporal_split(X, y, order)

    cfg = {**DEFAULT_PARAMS, **(params or {})}
    model = xgb.XGBRegressor(early_stopping_rounds=EARLY_STOPPING_ROUNDS, **cfg)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    importance = dict(zip(features, model.feature_importances_.tolist()))
    return TrainedModel(
        model=model,
        features=features,
        best_iteration=int(getattr(model, "best_iteration", cfg["n_estimators"] - 1)),
        feature_importance=dict(sorted(importance.items(), key=lambda kv: -kv[1])),
    )


def predict(trained: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    return trained.model.predict(df[trained.features])
