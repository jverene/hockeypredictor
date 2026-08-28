"""Backtest leakage guards: training targets must always precede the prediction season."""

import numpy as np
import pandas as pd

from src.backtest import _seasons_before, run_rookie_backtest, run_veteran_backtest
from src.features import ROOKIE_FEATURES, VETERAN_FEATURES


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


def _fake_rookies() -> pd.DataFrame:
    """Deterministic rookie panel: ppg_rookie = eq_ppg (perfectly learnable)."""
    rows = []
    for pid in range(1, 121):
        skill = 0.2 + 0.01 * (pid % 40)
        cls = 1995 + (pid % 6)  # draft classes 1995..2000
        row = {f: np.nan for f in ROOKIE_FEATURES}
        row.update(
            playerId=pid,
            skaterFullName=f"R{pid}",
            draft_year=cls,
            draft_ovr=30 + (pid % 5),
            rookie_season=cls * 10000 + cls + 1,
            age=19.0,
            eq_ppg=skill,
            ppg_rookie=skill,
            is_defense=pid % 7 == 0,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_rookie_backtest_never_trains_on_future(monkeypatch):
    seen = []

    import src.backtest as bt

    real_train = bt.train_model

    def spy(train, features, target, order_col, **kwargs):
        seen.append(train[order_col].max())
        return real_train(train, features, target, order_col, **kwargs)

    monkeypatch.setattr(bt, "train_model", spy)
    rookies = _fake_rookies()
    preds, classes = run_rookie_backtest(rookies, first_class=1998, last_class=2000, verbose=False)

    # Training debut seasons must precede each class's rookie season:
    # class 1998 trains through 1997-98 debuts, 1999 through 1998-99, etc.
    assert len(seen) == 3
    assert seen == [19971998, 19981999, 19992000]
    # Baseline and predictions present.
    assert {"pred_ppg", "pred_baseline"} <= set(preds.columns)
    assert classes["mae"].mean() < 0.05


def test_rookie_backtest_window_and_decay_run():
    rookies = _fake_rookies()
    _, tight = run_rookie_backtest(rookies, first_class=1999, last_class=2000,
                                   window=3, verbose=False)
    _, decayed = run_rookie_backtest(rookies, first_class=1999, last_class=2000,
                                     recency_decay=0.9, verbose=False)
    assert tight["n_train"].max() <= 60
    assert decayed["mae"].mean() < 0.05
