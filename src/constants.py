"""Project-wide constants: seasons, lockout handling, NHLe factors."""

from __future__ import annotations

# NHL season ids are YYYYYYYY, e.g. 19901991 = the 1990-91 season.
FIRST_BACKTEST_SEASON = 19901991
LAST_BACKTEST_SEASON = 20252026

# Rolling training window length (seasons) for the veteran model.
TRAIN_WINDOW = 5

# Earliest season of raw data we need: FIRST_BACKTEST_SEASON - TRAIN_WINDOW.
FIRST_DATA_SEASON = 19851986

# 2004-05 was cancelled outright (lockout). It does not exist in the data;
# listed here so all season arithmetic skips it explicitly.
CANCELLED_SEASONS: set[int] = {20042005}

# Shortened seasons (flagged, not skipped; PPG is per-game so still comparable).
SHORTENED_SEASONS: dict[int, str] = {
    19941995: "lockout (48 games)",
    20122013: "lockout (48 games)",
    20192020: "COVID cutoff (~70 games)",
    20202021: "COVID (56 games)",
}

# Minimum games in the *target* season for a player to be evaluated.
MIN_TARGET_GP = 15

# Minimum games in a pre-NHL season for it to count toward rookie features.
MIN_PRENHL_GP = 15

# NHL Equivalency factors: translate one point in league X to NHL points.
NHLE_FACTORS: dict[str, float] = {
    "NHL": 1.00,
    "KHL": 0.55,
    "SHL": 0.58,
    "LIIGA": 0.54,
    "AHL": 0.45,
    "NLA": 0.43,
    "DEL": 0.39,
    "CZECH": 0.40,  # addition beyond PRD: Czech Extraliga (literature ~0.3-0.45)
    "NCAA": 0.32,
    "OHL": 0.28,
    "WHL": 0.28,
    "QMJHL": 0.28,
    "USHL": 0.22,
}

# The NHL API's bio seasonTotals use historical league names; map them onto
# the canonical NHLe leagues above. Leagues without a literature-backed factor
# (ECHL, IHL, tournaments, European second tiers) stay unmapped and are dropped.
LEAGUE_ALIASES: dict[str, str] = {
    "SWEDEN": "SHL",        # Elitserien/SHL, pre-2013 naming
    "FINLAND": "LIIGA",     # SM-liiga, pre-2013 naming
    "GERMANY": "DEL",
    "SWISS": "NLA",
    "NL": "NLA",            # Swiss league rebranded National League in 2017
    "RUSSIA": "KHL",        # Russian Superleague, pre-2008
    "RUS-KHL": "KHL",
    "CZREP": "CZECH",
    "CZECHIA": "CZECH",
    "OMJHL": "OHL",         # Ontario Major Junior, pre-1981 naming
    "WCHA": "NCAA",         # NCAA conferences
    "CCHA": "NCAA",
    "H-EAST": "NCAA",
    "ECAC": "NCAA",
    "BIG-10": "NCAA",
    "NCHC": "NCAA",
    "AHA": "NCAA",
}

# Ordinal league level for rookie features (higher = closer to NHL).
LEAGUE_LEVELS: dict[str, int] = {
    "NHL": 5,
    "KHL": 4,
    "SHL": 4,
    "LIIGA": 4,
    "NLA": 4,
    "DEL": 4,
    "CZECH": 4,
    "AHL": 3,
    "NCAA": 2,
    "OHL": 1,
    "WHL": 1,
    "QMJHL": 1,
    "USHL": 1,
}

# Draft position used for undrafted players (PRD 4.2).
UNDRAFTED_OVR = 300

REGULAR_SEASON_GAME_TYPE = 2


def next_season(season_id: int) -> int:
    """Return the following season id, skipping the cancelled 2004-05 season."""
    year = season_id // 10000
    nxt = (year + 1) * 10000 + (year + 2)
    if nxt in CANCELLED_SEASONS:
        nxt = (year + 2) * 10000 + (year + 3)
    return nxt


def prev_season(season_id: int) -> int:
    """Return the preceding season id, skipping the cancelled 2004-05 season."""
    year = season_id // 10000
    prv = (year - 1) * 10000 + year
    if prv in CANCELLED_SEASONS:
        prv = (year - 2) * 10000 + (year - 1)
    return prv


def season_window(start: int, end: int) -> list[int]:
    """Inclusive list of season ids from start to end, skipping cancelled seasons."""
    seasons: list[int] = []
    s = start
    while s <= end:
        if s not in CANCELLED_SEASONS:
            seasons.append(s)
        s = (s // 10000 + 1) * 10000 + (s // 10000 + 2)
    return seasons


def season_label(season_id: int) -> str:
    """Human label, e.g. 19901991 -> '1990-91'."""
    y = season_id // 10000
    return f"{y}-{str((y + 1) % 100).zfill(2)}"
