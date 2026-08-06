import pytest

from src.constants import (
    CANCELLED_SEASONS,
    NHLE_FACTORS,
    next_season,
    prev_season,
    season_label,
    season_window,
)


def test_next_season_skips_cancelled_2004_05():
    assert next_season(20032004) == 20052006
    assert prev_season(20052006) == 20032004


def test_next_season_normal():
    assert next_season(20162017) == 20172018


def test_season_window_skips_cancelled():
    window = season_window(20022003, 20062007)
    assert window == [20022003, 20032004, 20052006, 20062007]
    assert all(s not in CANCELLED_SEASONS for s in window)


def test_season_label():
    assert season_label(19901991) == "1990-91"
    assert season_label(20052006) == "2005-06"


def test_nhle_factors_cover_prd_leagues():
    for league in ["NHL", "KHL", "SHL", "LIIGA", "AHL", "NLA", "DEL", "NCAA", "OHL", "WHL", "QMJHL", "USHL"]:
        assert league in NHLE_FACTORS
    assert NHLE_FACTORS["NHL"] == 1.00
    assert NHLE_FACTORS["USHL"] < NHLE_FACTORS["AHL"] < NHLE_FACTORS["KHL"]
