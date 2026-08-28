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
    prev_season,
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
    "eq_ppg_best",
    "pre_gp",
    "prenhl_seasons",
    "seasons_since_last_prenhl",
    "top_league_level",
    "had_cup_of_coffee",
    "coffee_gp",
    "coffee_ppg",
    "age",
    "age_sq",
    "age_vs_league_avg",
    "age_x_eq_ppg",
    "draft_ovr_x_eq_ppg",
    "draft_ovr",
    "is_undrafted",
    "years_post_draft",
    "draft_year",
    "is_defense",
    "height",
    "weight",
    "nhl_lg_ppg_prev",
]

# Pre-NHL production window: the ROOKIE_PRE_WINDOW calendar seasons before
# the rookie year, not just the single immediately-preceding season (gap
# years from injury/cancelled seasons previously nuked all eq_* features).
ROOKIE_PRE_WINDOW = 3

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

    Rookie season = first NHL season with >= MIN_TARGET_GP games. Every feature
    uses only information available *before* that season:

      - NHLe-weighted pre-NHL production over the ROOKIE_PRE_WINDOW calendar
        seasons before the rookie year (GP-weighted, so partial seasons are
        down-weighted instead of dropped), plus a best-single-season eq_ppg
        and a gap-year feature (`seasons_since_last_prenhl`) so an injured or
        cancelled final junior season no longer erases a player's history.
      - Cup-of-coffee aggregates from earlier NHL stints — known at
        prediction time, yet previously discarded (the NHL stint is excluded
        from the pre-NHL frame).
      - Era context: league-wide NHL PPG across the two seasons before the
        rookie year (rookie targets shift with scoring environment; the
        veteran model had `era_adj_factor`, the rookie model nothing).
      - Interactions `age × eq_ppg` and `draft_ovr × eq_ppg`.

    When pre-NHL data is missing entirely the eq_* features stay NaN
    (XGBoost handles NaN natively).
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

    # Production from those earlier NHL stints (per-game rates, summed GP).
    prior_nhl = rookies[["playerId", "rookie_season"]].merge(
        df[["playerId", "seasonId", "gp", "points"]], on="playerId", how="left"
    )
    prior_nhl = prior_nhl[prior_nhl["seasonId"] < prior_nhl["rookie_season"]]
    coffee = prior_nhl.groupby("playerId").agg(
        coffee_gp=("gp", "sum"),
        coffee_pts=("points", "sum"),
        coffee_seasons=("seasonId", "nunique"),
    )
    coffee["coffee_ppg"] = np.where(
        coffee["coffee_gp"] > 0, coffee["coffee_pts"] / coffee["coffee_gp"], np.nan
    )
    rookies = rookies.merge(coffee.drop(columns="coffee_pts"), on="playerId", how="left")
    rookies["coffee_gp"] = rookies["coffee_gp"].fillna(0)

    # Era context: NHL league-wide PPG in the two seasons before the rookie
    # year (prev_season skips the cancelled 2004-05).
    lg = (
        df.groupby("seasonId")
        .apply(lambda g: g["points"].sum() / g["gp"].sum(), include_groups=False)
        .rename("nhl_lg_ppg")
    )
    ctx1 = rookies["rookie_season"].map(prev_season)
    rookies["nhl_lg_ppg_prev"] = pd.concat(
        [ctx1.map(lg), ctx1.map(prev_season).map(lg)], axis=1
    ).mean(axis=1)  # skips NaN: falls back to the single available season

    rookies["age"] = _age_at_season_start(rookies["birthDate"], rookies["seasonId"])
    rookies["age_sq"] = rookies["age"] ** 2
    rookies["draft_ovr"] = rookies["draftOverall"].fillna(UNDRAFTED_OVR)
    rookies["is_undrafted"] = rookies["draftOverall"].isna().astype(int)
    rookies["draft_year"] = rookies["draftYear"]
    rookies["years_post_draft"] = (rookies["seasonId"] // 10000) - rookies["draft_year"]
    rookies["height"] = rookies["heightInInches"]
    rookies["weight"] = rookies["weightInPounds"]
    rookies["ppg_rookie"] = rookies["ppg"]

    eq_cols = [
        "eq_ppg", "eq_gpg", "eq_apg", "eq_ppg_best", "pre_gp",
        "prenhl_seasons", "seasons_since_last_prenhl", "top_league_level",
        "age_vs_league_avg",
    ]
    if prenhl.empty:
        for col in eq_cols:
            rookies[col] = np.nan
        rookies.loc[rookies["had_cup_of_coffee"], "top_league_level"] = LEAGUE_LEVELS["NHL"]
        rookies["age_x_eq_ppg"] = rookies["age"] * rookies["eq_ppg"]
        rookies["draft_ovr_x_eq_ppg"] = rookies["draft_ovr"] * rookies["eq_ppg"]
        return rookies.reset_index(drop=True)

    # NHLe-weighted pre-NHL production over a multi-season lookback window.
    pre = prenhl.copy()
    pre["key"] = pre["player_name"].map(_norm_name) + "|" + pd.to_datetime(
        pre["birth_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    pre["points"] = pre["goals"] + pre["assists"]
    pre["league"] = pre["league"].astype(str).str.upper().map(lambda l: LEAGUE_ALIASES.get(l, l))
    pre["nhle"] = pre["league"].map(NHLE_FACTORS)
    # Keep short/partial seasons: GP weighting down-weights them naturally
    # (dropping them used to zero out players after an injury or a cancelled
    # season entirely).
    pre = pre[(pre["gp"] > 0) & pre["nhle"].notna()]

    rookies["key"] = rookies["skaterFullName"].map(_norm_name) + "|" + pd.to_datetime(
        rookies["birthDate"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    rookies["pre_year"] = (rookies["seasonId"] // 10000) - 1

    merged = pre.merge(
        rookies[["playerId", "key", "pre_year"]],
        left_on=["key"],
        right_on=["key"],
        how="inner",
    )
    merged = merged[
        (merged["season"] <= merged["pre_year"])
        & (merged["season"] > merged["pre_year"] - ROOKIE_PRE_WINDOW)
    ]

    def _eq(g2: pd.DataFrame, stat: str) -> float:
        return float((g2[stat] * g2["nhle"]).sum() / g2["gp"].sum())

    def _agg(g2: pd.DataFrame) -> pd.Series:
        # Only substantial seasons set the ordinal league level (a 2-game KHL
        # call-up should not read as "KHL player").
        full = g2[g2["gp"] >= MIN_PRENHL_GP]
        return pd.Series(
            {
                "eq_ppg": _eq(g2, "points"),
                "eq_gpg": _eq(g2, "goals"),
                "eq_apg": _eq(g2, "assists"),
                "pre_gp": g2["gp"].sum(),
                "top_league_level": full["league"].astype(str).str.upper()
                .map(LEAGUE_LEVELS).max(),
                "prenhl_seasons": g2["season"].nunique(),
                "last_prenhl_season": g2["season"].max(),
            }
        )

    agg = (
        merged.groupby("playerId")
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    rookies = rookies.merge(agg, on="playerId", how="left")
    rookies.loc[rookies["had_cup_of_coffee"], "top_league_level"] = LEAGUE_LEVELS["NHL"]

    # Best single pre-NHL season (substantial samples only) — captures peak
    # talent even when the final season was disrupted.
    per_season = (
        merged.assign(eq_pts=merged["points"] * merged["nhle"])
        .groupby(["playerId", "season"])
        .agg(eq_pts=("eq_pts", "sum"), gp=("gp", "sum"))
        .reset_index()
    )
    best = (
        per_season[per_season["gp"] >= MIN_PRENHL_GP]
        .assign(season_eq_ppg=lambda d: d["eq_pts"] / d["gp"])
        .groupby("playerId")["season_eq_ppg"]
        .max()
        .rename("eq_ppg_best")
        .reset_index()
    )
    rookies = rookies.merge(best, on="playerId", how="left")

    # Gap years since meaningful pre-NHL action (0 = played the season right
    # before the debut; NaN = no usable pre-NHL data at all).
    rookies["seasons_since_last_prenhl"] = (
        rookies["pre_year"] - rookies["last_prenhl_season"]
    )
    rookies.drop(columns=["last_prenhl_season"], inplace=True)

    avg_ages = _league_avg_ages(prenhl)
    if not avg_ages.empty:
        # Most recent league-season in the window defines the age context.
        recent = (
            merged.sort_values("season")
            .groupby("playerId", as_index=False)
            .tail(1)[["playerId", "league", "season"]]
        )
        pre_league = recent.merge(avg_ages, on=["league", "season"], how="left")
        rookies = rookies.merge(
            pre_league[["playerId", "league_avg_age"]], on="playerId", how="left"
        )
        rookies["age_vs_league_avg"] = rookies["age"] - rookies["league_avg_age"]
    else:
        rookies["age_vs_league_avg"] = np.nan

    rookies["age_x_eq_ppg"] = rookies["age"] * rookies["eq_ppg"]
    rookies["draft_ovr_x_eq_ppg"] = rookies["draft_ovr"] * rookies["eq_ppg"]

    return rookies.reset_index(drop=True)
