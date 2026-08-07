"""Feature engineering for the veteran and rookie prediction modes (PRD §4).

All features are computed using only information available *before* the target
season — lags, career aggregates, and era context never look forward.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from .constants import (
    LEAGUE_ALIASES,
    LEAGUE_LEVELS,
    MIN_PRENHL_GP,
    MIN_TARGET_GP,
    NHLE_FACTORS,
    UNDRAFTED_OVR,
    next_season,
)

VETERAN_FEATURES = [
    "age",
    "age_sq",
    "age_vs_peak",
    "years_since_debut",
    "is_defense",
    "ppg_last1",
    "ppg_last2",
    "ppg_last3",
    "ppg_rolling_3yr",
    "gpg_last1",
    "apg_last1",
    "gp_last1",
    "career_ppg",
    "career_gp",
    "team_pts_pct",
    "era_adj_factor",
    "age_x_ppg_last1",
]

ROOKIE_FEATURES = [
    "eq_ppg",
    "eq_gpg",
    "eq_apg",
    "pre_gp",
    "top_league_level",
    "age",
    "age_sq",
    "age_vs_league_avg",
    "draft_ovr",
    "draft_year",
    "years_post_draft",
    "is_defense",
    "height",
    "weight",
]

PEAK_AGE = 26.0


def _norm_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for NHL<->EP name matching."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join("".join(c for c in s if c.isalnum() or c == " ").lower().split())


def _age_at_season_start(birth: pd.Series, season_id: pd.Series | int) -> pd.Series:
    """Biological age on Oct 1 of the season's first year (vectorized)."""
    start = pd.to_datetime((pd.Series(season_id, index=birth.index) // 10000).astype(str) + "-10-01")
    return (start - pd.to_datetime(birth)).dt.days / 365.25


# ---------------------------------------------------------------------------
# Veteran panel
# ---------------------------------------------------------------------------


def build_player_seasons(skaters: pd.DataFrame, bios: pd.DataFrame) -> pd.DataFrame:
    """One row per (playerId, seasonId) with per-game rates and context columns.

    Multi-team seasons are summed; era context is the league-wide scoring rate.
    """
    df = skaters.copy()
    df["is_defense"] = (df["positionCode"] == "D").astype(int)

    # League-average scoring rate that season (era adjustment context).
    league = (
        df.groupby("seasonId")
        .apply(lambda g: g["points"].sum() / g["gamesPlayed"].sum(), include_groups=False)
        .rename("era_adj_factor")
        .reset_index()
    )

    df = (
        df.groupby(["playerId", "seasonId"], as_index=False)
        .agg(
            skaterFullName=("skaterFullName", "first"),
            is_defense=("is_defense", "max"),
            gp=("gamesPlayed", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            points=("points", "sum"),
        )
    )
    df = df.merge(league, on="seasonId", how="left")

    bio_cols = bios[["playerId", "birthDate", "draftYear", "draftOverall", "heightInInches", "weightInPounds"]]
    df = df.merge(bio_cols, on="playerId", how="left")
    return df.sort_values(["playerId", "seasonId"]).reset_index(drop=True)


def attach_team_quality(
    player_seasons: pd.DataFrame,
    skaters: pd.DataFrame,
    teams: pd.DataFrame,
    team_meta: pd.DataFrame,
) -> pd.DataFrame:
    """GP-weighted team points percentage per player-season (PRD `team_pts_pct`).

    Skater rows carry team abbrevs (comma-joined for multi-team seasons); the
    team summary is keyed by teamId. The stats-REST team directory reconciles
    abbrevs -> teamIds across historical franchises (QUE, HFD, WIN, ...).
    """
    abbrev_to_id = team_meta.set_index("triCode")["teamId"].to_dict()
    team_pct = teams.set_index(["seasonId", "teamId"])["pointPct"].to_dict()

    stint = skaters[["playerId", "seasonId", "teams", "gamesPlayed"]].copy()
    stint["abbrev"] = stint["teams"].str.split(",")
    stint = stint.explode("abbrev")
    stint["teamId"] = stint["abbrev"].str.strip().map(abbrev_to_id)
    stint["pointPct"] = [
        team_pct.get((s, t), np.nan) for s, t in zip(stint["seasonId"], stint["teamId"])
    ]
    # Even split of GP across stints (API gives no per-stint GP; skater rows
    # already carry per-stint gamesPlayed, so weight by that).
    stint["gp_weight"] = stint["gamesPlayed"] / stint.groupby(["playerId", "seasonId"])[
        "gamesPlayed"
    ].transform("sum")
    stint["weighted"] = stint["pointPct"] * stint["gp_weight"]
    quality = stint.groupby(["playerId", "seasonId"])["weighted"].sum().rename("team_pts_pct")
    return player_seasons.merge(quality.reset_index(), on=["playerId", "seasonId"], how="left")


def build_veteran_features(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for players with >=1 prior NHL season (PRD §4.1).

    Each row describes a player after season `seasonId`; the target `ppg_next`
    is the following season's PPG (NaN if the player did not play).
    """
    df = player_seasons.copy()
    df["ppg"] = df["points"] / df["gp"]
    df["gpg"] = df["goals"] / df["gp"]
    df["apg"] = df["assists"] / df["gp"]
    df["age"] = _age_at_season_start(df["birthDate"], df["seasonId"])

    g = df.groupby("playerId")
    df["years_since_debut"] = g.cumcount()
    df["career_gp"] = g["gp"].cumsum() - df["gp"]
    df["career_pts"] = g["points"].cumsum() - df["points"]
    df["career_ppg"] = np.where(df["career_gp"] > 0, df["career_pts"] / df["career_gp"], np.nan)

    df["ppg_last1"] = g["ppg"].shift(1)
    df["ppg_last2"] = g["ppg"].shift(2)
    df["ppg_last3"] = g["ppg"].shift(3)
    df["gpg_last1"] = g["gpg"].shift(1)
    df["apg_last1"] = g["apg"].shift(1)
    df["gp_last1"] = g["gp"].shift(1)

    pts3 = g["points"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    gp3 = g["gp"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["ppg_rolling_3yr"] = (pts3.shift(1) / gp3.shift(1)).where(gp3.shift(1) > 0)

    # Next-season target.
    df["next_season_actual"] = df["seasonId"].map(next_season)
    nxt = df[["playerId", "seasonId", "ppg", "gp"]].rename(
        columns={"seasonId": "next_season_actual", "ppg": "ppg_next", "gp": "gp_next"}
    )
    df = df.merge(nxt, on=["playerId", "next_season_actual"], how="left")

    # Rows predict season Y from data through Y-1: require >=1 prior season.
    out = df[df["years_since_debut"] >= 1].copy()
    out["age_sq"] = out["age"] ** 2
    out["age_vs_peak"] = (out["age"] - PEAK_AGE).abs()
    out["age_x_ppg_last1"] = out["age"] * out["ppg_last1"]
    out["eval_ok"] = out["gp_next"] >= MIN_TARGET_GP
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rookie panel
# ---------------------------------------------------------------------------


def _league_avg_ages(prenhl: pd.DataFrame) -> pd.DataFrame:
    """Average age per (league, season) for the `age_vs_league_avg` feature."""
    if prenhl.empty:
        return pd.DataFrame(columns=["league", "season", "league_avg_age"])
    tmp = prenhl.dropna(subset=["birth_date"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["league", "season", "league_avg_age"])
    tmp["age"] = tmp.apply(
        lambda r: (pd.Timestamp(f"{int(str(r['season'])[:4])}-10-01") - pd.to_datetime(r["birth_date"])).days
        / 365.25,
        axis=1,
    )
    return tmp.groupby(["league", "season"])["age"].mean().rename("league_avg_age").reset_index()


def build_rookie_features(player_seasons: pd.DataFrame, prenhl: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for first-year NHL players (PRD §4.2).

    Rookie season = first NHL season with >= MIN_TARGET_GP games. Pre-NHL
    production is translated via NHLe factors; when pre-NHL data is missing
    the eq_* features are NaN (XGBoost handles NaN natively).
    """
    df = player_seasons.copy()
    df["ppg"] = df["points"] / df["gp"]
    df["gpg"] = df["goals"] / df["gp"]
    df["apg"] = df["assists"] / df["gp"]

    first_full = (
        df[df["gp"] >= MIN_TARGET_GP].groupby("playerId")["seasonId"].min().rename("rookie_season")
    )
    rookies = df.merge(first_full, on="playerId", how="inner")
    rookies = rookies[rookies["seasonId"] == rookies["rookie_season"]].copy()

    # Cup-of-coffee flag: any NHL stint before the rookie season.
    first_ever = df.groupby("playerId")["seasonId"].min().rename("first_nhl_season")
    rookies = rookies.merge(first_ever, on="playerId", how="left")
    rookies["had_cup_of_coffee"] = rookies["rookie_season"] > rookies["first_nhl_season"]

    rookies["age"] = _age_at_season_start(rookies["birthDate"], rookies["seasonId"])
    rookies["age_sq"] = rookies["age"] ** 2
    rookies["draft_ovr"] = rookies["draftOverall"].fillna(UNDRAFTED_OVR)
    rookies["draft_year"] = rookies["draftYear"]
    rookies["years_post_draft"] = (rookies["seasonId"] // 10000) - rookies["draft_year"]
    rookies["height"] = rookies["heightInInches"]
    rookies["weight"] = rookies["weightInPounds"]
    rookies["ppg_rookie"] = rookies["ppg"]

    if prenhl.empty:
        for col in ["eq_ppg", "eq_gpg", "eq_apg", "pre_gp", "top_league_level", "age_vs_league_avg"]:
            rookies[col] = np.nan
        rookies.loc[rookies["had_cup_of_coffee"], "top_league_level"] = LEAGUE_LEVELS["NHL"]
        return rookies.reset_index(drop=True)

    # NHLe-weighted pre-NHL production from the season before the rookie year.
    pre = prenhl.copy()
    pre["key"] = pre["player_name"].map(_norm_name) + "|" + pd.to_datetime(
        pre["birth_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    pre["points"] = pre["goals"] + pre["assists"]
    pre["league"] = pre["league"].astype(str).str.upper().map(lambda l: LEAGUE_ALIASES.get(l, l))
    pre["nhle"] = pre["league"].map(NHLE_FACTORS)
    pre = pre[(pre["gp"] >= MIN_PRENHL_GP) & pre["nhle"].notna()]

    rookies["key"] = rookies["skaterFullName"].map(_norm_name) + "|" + pd.to_datetime(
        rookies["birthDate"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    rookies["pre_year"] = (rookies["seasonId"] // 10000) - 1

    merged = pre.merge(
        rookies[["playerId", "key", "pre_year", "had_cup_of_coffee"]],
        left_on=["key", "season"],
        right_on=["key", "pre_year"],
        how="inner",
    )

    def _eq(g2: pd.DataFrame, stat: str) -> float:
        return float((g2["gp"] * (g2[stat] / g2["gp"]) * g2["nhle"]).sum() / g2["gp"].sum())

    agg = (
        merged.groupby("playerId")
        .apply(
            lambda g2: pd.Series(
                {
                    "eq_ppg": _eq(g2, "points"),
                    "eq_gpg": _eq(g2, "goals"),
                    "eq_apg": _eq(g2, "assists"),
                    "pre_gp": g2["gp"].sum(),
                    "top_league_level": g2["league"].astype(str).str.upper().map(LEAGUE_LEVELS).max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    rookies = rookies.merge(agg, on="playerId", how="left")
    rookies.loc[rookies["had_cup_of_coffee"], "top_league_level"] = LEAGUE_LEVELS["NHL"]

    avg_ages = _league_avg_ages(prenhl)
    if not avg_ages.empty:
        pre_league = (
            merged[["playerId", "league", "season"]]
            .drop_duplicates("playerId")
            .merge(avg_ages, on=["league", "season"], how="left")
        )
        rookies = rookies.merge(
            pre_league[["playerId", "league_avg_age"]], on="playerId", how="left"
        )
        rookies["age_vs_league_avg"] = rookies["age"] - rookies["league_avg_age"]
    else:
        rookies["age_vs_league_avg"] = np.nan

    return rookies.reset_index(drop=True)
