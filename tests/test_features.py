"""Feature tests on a tiny synthetic career panel."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    _norm_name,
    build_player_seasons,
    build_rookie_features,
    build_veteran_features,
)


def _bios():
    return pd.DataFrame(
        {
            "playerId": [1, 2],
            "birthDate": ["1990-06-15", "1988-01-01"],
            "draftYear": [2008, 2006],
            "draftOverall": [10, np.nan],
            "heightInInches": [72, 74],
            "weightInPounds": [190, 210],
        }
    )


def _skaters():
    # Player 1: three seasons; player 2: two seasons.
    rows = [
        (1, "Alice Alpha", "C", 20102011, "TOR", 80, 20, 30, 50),
        (1, "Alice Alpha", "C", 20112012, "TOR", 82, 25, 35, 60),
        (1, "Alice Alpha", "C", 20122013, "TOR", 48, 15, 25, 40),
        (2, "Bob Beta", "D", 20112012, "MTL", 70, 5, 15, 20),
        (2, "Bob Beta", "D", 20122013, "MTL", 40, 3, 10, 13),
    ]
    return pd.DataFrame(
        rows,
        columns=["playerId", "skaterFullName", "positionCode", "seasonId", "teams", "gamesPlayed", "goals", "assists", "points"],
    )


def _panel():
    return build_player_seasons(_skaters(), _bios())


def test_veteran_lags_and_targets():
    vet = build_veteran_features(_panel())
    alice_2011 = vet[(vet["playerId"] == 1) & (vet["seasonId"] == 20112012)].iloc[0]

    assert alice_2011["years_since_debut"] == 1
    assert alice_2011["ppg_last1"] == 50 / 80
    assert alice_2011["gp_last1"] == 80
    assert alice_2011["career_gp"] == 80
    assert alice_2011["career_ppg"] == 50 / 80
    assert alice_2011["ppg_next"] == 40 / 48  # target is next season's PPG
    assert alice_2011["gp_next"] == 48
    assert alice_2011["eval_ok"]  # 48 >= 15
    assert alice_2011["age_x_ppg_last1"] == alice_2011["age"] * alice_2011["ppg_last1"]


def test_veteran_excludes_first_season():
    vet = build_veteran_features(_panel())
    # No rows for a player's debut season (needs >=1 prior NHL season).
    assert not ((vet["playerId"] == 1) & (vet["seasonId"] == 20102011)).any()
    assert not ((vet["playerId"] == 2) & (vet["seasonId"] == 20112012)).any()


def test_veteran_missing_next_season_is_nan():
    vet = build_veteran_features(_panel())
    alice_2012 = vet[(vet["playerId"] == 1) & (vet["seasonId"] == 20122013)].iloc[0]
    assert np.isnan(alice_2012["ppg_next"])
    assert not alice_2012["eval_ok"]


def test_rookie_first_full_season():
    rooks = build_rookie_features(_panel(), pd.DataFrame())
    # Player 1 rookie = 2010-11 (first >=15 GP). Player 2 rookie = 2011-12.
    r1 = rooks[rooks["playerId"] == 1].iloc[0]
    r2 = rooks[rooks["playerId"] == 2].iloc[0]
    assert r1["rookie_season"] == 20102011
    assert r1["ppg_rookie"] == 50 / 80
    assert r1["draft_ovr"] == 10
    assert r2["draft_ovr"] == 300  # undrafted sentinel
    assert r2["is_defense"] == 1
    assert np.isnan(r1["eq_ppg"])  # no pre-NHL data -> NaN, not crash


def test_rookie_nhle_aggregation():
    prenhl = pd.DataFrame(
        {
            "player_name": ["Alice Alpha", "Alice Alpha"],
            "birth_date": ["1990-06-15", "1990-06-15"],
            "season": [2009, 2009],
            "league": ["OHL", "AHL"],
            "team": ["X", "Y"],
            "gp": [60, 20],
            "goals": [30, 5],
            "assists": [40, 10],
        }
    )
    rooks = build_rookie_features(_panel(), prenhl)
    r1 = rooks[rooks["playerId"] == 1].iloc[0]
    # EQ_PPG = (60*(70/60)*0.28 + 20*(15/20)*0.45) / 80
    expected = (70 * 0.28 + 15 * 0.45) / 80
    assert r1["eq_ppg"] == pytest.approx(expected)
    assert r1["pre_gp"] == 80
    assert r1["top_league_level"] == 3  # AHL outranks OHL


def test_rookie_nhle_multi_season_window():
    # Alice: dominant 2008 season, then a 10-GP partial in 2009 (injury).
    # The 3-year lookback keeps both; single-season matching would have seen
    # only the partial year.
    prenhl = pd.DataFrame(
        {
            "player_name": ["Alice Alpha", "Alice Alpha"],
            "birth_date": ["1990-06-15", "1990-06-15"],
            "season": [2008, 2009],
            "league": ["OHL", "OHL"],
            "team": ["X", "X"],
            "gp": [60, 10],
            "goals": [40, 3],
            "assists": [50, 4],
        }
    )
    rooks = build_rookie_features(_panel(), prenhl)
    r1 = rooks[rooks["playerId"] == 1].iloc[0]
    assert r1["eq_ppg"] == pytest.approx((90 * 0.28 + 7 * 0.28) / 70)  # GP-weighted
    assert r1["pre_gp"] == 70
    assert r1["prenhl_seasons"] == 2
    assert r1["seasons_since_last_prenhl"] == 0
    # Best *qualifying* season is the healthy 2008 one.
    assert r1["eq_ppg_best"] == pytest.approx(90 * 0.28 / 60)
    assert r1["age_x_eq_ppg"] == pytest.approx(r1["age"] * r1["eq_ppg"])
    assert r1["draft_ovr_x_eq_ppg"] == pytest.approx(10 * r1["eq_ppg"])


def test_rookie_gap_year_still_gets_nhle():
    # Only pre-NHL data is from 2007; debut is 2010-11 (pre_year=2009).
    # Exact-season matching produced all-NaN eq_* here; the window recovers it.
    prenhl = pd.DataFrame(
        {
            "player_name": ["Alice Alpha"],
            "birth_date": ["1990-06-15"],
            "season": [2007],
            "league": ["OHL"],
            "team": ["X"],
            "gp": [60],
            "goals": [30],
            "assists": [40],
        }
    )
    rooks = build_rookie_features(_panel(), prenhl)
    r1 = rooks[rooks["playerId"] == 1].iloc[0]
    assert not np.isnan(r1["eq_ppg"])
    assert r1["seasons_since_last_prenhl"] == 2


def _panel_with_coffee():
    skaters = _skaters()
    extra = pd.DataFrame(
        [
            (3, "Cara Gamma", "C", 20102011, "TOR", 10, 2, 3, 5),
            (3, "Cara Gamma", "C", 20112012, "TOR", 50, 15, 15, 30),
        ],
        columns=skaters.columns,
    )
    bios = pd.concat(
        [
            _bios(),
            pd.DataFrame(
                {
                    "playerId": [3],
                    "birthDate": ["1991-03-03"],
                    "draftYear": [2009],
                    "draftOverall": [55],
                    "heightInInches": [71],
                    "weightInPounds": [185],
                }
            ),
        ],
        ignore_index=True,
    )
    return build_player_seasons(pd.concat([skaters, extra], ignore_index=True), bios)


def test_rookie_cup_of_coffee_features():
    rooks = build_rookie_features(_panel_with_coffee(), pd.DataFrame())
    r3 = rooks[rooks["playerId"] == 3].iloc[0]
    assert r3["rookie_season"] == 20112012
    assert bool(r3["had_cup_of_coffee"]) is True
    assert r3["coffee_gp"] == 10
    assert r3["coffee_ppg"] == pytest.approx(5 / 10)
    # No pre-NHL frame: eq_* NaN but the NHL stint itself is known signal.
    assert np.isnan(r3["eq_ppg"])
    r1 = rooks[rooks["playerId"] == 1].iloc[0]
    assert r1["coffee_gp"] == 0  # no earlier stint


def test_rookie_era_context_uses_only_past_seasons():
    rooks = build_rookie_features(_panel(), pd.DataFrame())
    bob = rooks[rooks["playerId"] == 2].iloc[0]
    # Bob's rookie year is 2011-12; among prior seasons the panel only has
    # 2010-11 -> league PPG falls back to that single season.
    assert bob["nhl_lg_ppg_prev"] == pytest.approx(50 / 80)


def test_rookie_era_context_no_future_leakage():
    panel = _panel()
    leaked = panel.copy()
    mask = (leaked["playerId"] == 1) & (leaked["seasonId"] == 20122013)
    leaked.loc[mask, "points"] = leaked.loc[mask, "points"] * 20
    base = build_rookie_features(panel, pd.DataFrame())
    after = build_rookie_features(leaked, pd.DataFrame())
    same = base["nhl_lg_ppg_prev"].fillna(-1).eq(after["nhl_lg_ppg_prev"].fillna(-1))
    assert same.all()


def test_name_normalization():
    assert _norm_name("Montréal Dupont") == _norm_name("montreal dupont")
    assert _norm_name("Tomáš  Tatar") == "tomas tatar"
