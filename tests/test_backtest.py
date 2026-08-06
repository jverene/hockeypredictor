"""Backtest leakage guards: training targets must always precede the prediction season."""

import numpy as np
import pandas as pd

from src.backtest import _seasons_before, run_veteran_backtest
from src.features import VETERAN_FEATURES


def _fake_veterans() -> pd.DataFrame:
    """Deterministic panel: ppg_next = 0.5 * ppg_last1 + noise-free constant."""
    rows = []
    seasons = [19851986, 19861987, 19871988, 19881989, 19891990, 19901991]
    for pid in range(1, 61):
        skill = 0.3 + 0.01 * pid
        for i, s in enumerate(seasons):
            row = {f: 0.5 for f in VETERAN_FEATURES}
            row.update(
                playerId=pid,
                skaterFullName=f"P{pid}",
                seasonId=s,
                ppg_last1=skill,
                gp_last1=80,
                years_since_debut=i + 1,
                next_season_actual=s + 10001,
                ppg_next=skill,
                gp_next=80,
                eval_ok=True,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_seasons_before_window_size():
    before = _seasons_before(19901991, 5)
    assert len(before) == 5
    assert max(before) == 19891990


def test_backtest_never_trains_on_future():
    veterans = _fake_veterans()
    preds, seasons = run_veteran_backtest(veterans, first=19881989, last=19901991, verbose=False)

    assert set(preds["pred_season"]) == {19881989, 19891990, 19901991}
    # With a perfectly learnable target the model should do very well.
    assert seasons["mae"].mean() < 0.05


def test_backtest_window_is_strict(monkeypatch):
    """Training frames passed to the model must only contain earlier target seasons."""
    seen = []

    import src.backtest as bt

    real_train = bt.train_model

    def spy(train, features, target, order_col):
        seen.append((train[order_col].min(), train[order_col].max()))
        return real_train(train, features, target, order_col)

    monkeypatch.setattr(bt, "train_model", spy)
    run_veteran_backtest(_fake_veterans(), first=19891990, last=19901991, verbose=False)

    for (tmin, tmax), Y in zip(seen, [19891990, 19901991]):
        assert tmax < Y, f"leakage: training target {tmax} not before {Y}"
        assert tmin >= Y - 5 * 10001, "rolling window exceeded"
