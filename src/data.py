"""NHL data ingestion: stats REST API + api-web, with on-disk JSON caching.

Data sources (all free, no auth):
  - api.nhle.com/stats/rest/en/skater/summary  — per-season skater box-score totals
  - api.nhle.com/stats/rest/en/team/summary    — per-season team results
  - api-web.nhle.com/v1/player/{id}/landing    — bio (DOB, position, size, draft)

Everything fetched is cached under data/cache/ so re-runs are offline and the
API is never hammered. Use `--refresh` to re-pull.

Pre-NHL (junior/minor/European) stats are NOT available from the NHL API. The
rookie layer loads them from `data/prenhl_stats.csv` if present (schema
documented in `load_prenhl_stats`); the intended producers are an
EliteProspects scrape or the Kaggle "Hockey stats 30 leagues" dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from .constants import (
    FIRST_DATA_SEASON,
    LAST_BACKTEST_SEASON,
    REGULAR_SEASON_GAME_TYPE,
    season_label,
    season_window,
)

STATS_REST = "https://api.nhle.com/stats/rest/en"
API_WEB = "https://api-web.nhle.com/v1"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
PRENHL_CSV = Path(__file__).resolve().parent.parent / "data" / "prenhl_stats.csv"

_PAGE_SIZE = 100
_TIMEOUT = 30
_RETRIES = 3


# ---------------------------------------------------------------------------
# Low-level HTTP with cache
# ---------------------------------------------------------------------------


def _cache_path(namespace: str, key: str) -> Path:
    path = CACHE_DIR / namespace
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{key}.json"


def _get_json(url: str, params: dict | None, namespace: str, key: str, refresh: bool) -> dict:
    path = _cache_path(namespace, key)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    last_err: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                time.sleep(min(wait, 60))
                continue
            resp.raise_for_status()
            payload = resp.json()
            path.write_text(json.dumps(payload))
            return payload
        except Exception as err:  # noqa: BLE001 - retried below
            last_err = err
            time.sleep(min(2**attempt * 2.0, 60))
    raise RuntimeError(f"GET {url} failed after {_RETRIES} attempts: {last_err}")


def _paged(url: str, base_params: dict, namespace: str, key: str, refresh: bool) -> list[dict]:
    """Fetch all pages of a stats-REST endpoint."""
    rows: list[dict] = []
    start = 0
    while True:
        params = {**base_params, "limit": _PAGE_SIZE, "start": start}
        payload = _get_json(url, params, namespace, f"{key}_{start}", refresh)
        batch = payload.get("data", [])
        rows.extend(batch)
        total = payload.get("total", 0)
        start += _PAGE_SIZE
        if start >= total or not batch:
            return rows


# ---------------------------------------------------------------------------
# Public fetchers
# ---------------------------------------------------------------------------


def fetch_skater_season(season_id: int, refresh: bool = False) -> pd.DataFrame:
    """Regular-season totals for every skater in one season."""
    rows = _paged(
        f"{STATS_REST}/skater/summary",
        {"cayenneExp": f"gameTypeId={REGULAR_SEASON_GAME_TYPE} and seasonId={season_id}"},
        "skater_summary",
        str(season_id),
        refresh,
    )
    df = pd.DataFrame(rows)
    keep = [
        "playerId",
        "skaterFullName",
        "positionCode",
        "seasonId",
        "teamAbbrevs",
        "gamesPlayed",
        "goals",
        "assists",
        "points",
    ]
    return df[keep].rename(columns={"teamAbbrevs": "teams"})


def fetch_team_season(season_id: int, refresh: bool = False) -> pd.DataFrame:
    """Regular-season results for every team in one season."""
    rows = _paged(
        f"{STATS_REST}/team/summary",
        {"cayenneExp": f"gameTypeId={REGULAR_SEASON_GAME_TYPE} and seasonId={season_id}"},
        "team_summary",
        str(season_id),
        refresh,
    )
    df = pd.DataFrame(rows)
    keep = ["teamId", "teamFullName", "seasonId", "gamesPlayed", "wins", "losses", "otLosses", "points", "pointPct"]
    return df[keep]


def fetch_team_meta(refresh: bool = False) -> pd.DataFrame:
    """Team directory: (teamId, triCode, fullName) incl. historical franchises."""
    rows = _paged(f"{STATS_REST}/team", {}, "team_meta", "all", refresh)
    df = pd.DataFrame(rows)
    return df[["id", "triCode", "fullName"]].rename(columns={"id": "teamId"})


def fetch_player_bio(player_id: int, refresh: bool = False) -> dict:
    """Bio + draft details for one player.

    On persistent API failure (e.g. rate limiting), returns a stub with null
    fields instead of raising — XGBoost handles NaN features, and a cached
    stub can be re-pulled later with refresh=True.
    """
    try:
        payload = _get_json(f"{API_WEB}/player/{player_id}/landing", None, "bio", str(player_id), refresh)
    except RuntimeError:
        return {"playerId": player_id}
    draft = payload.get("draftDetails") or {}
    return {
        "playerId": payload.get("playerId"),
        "firstName": (payload.get("firstName") or {}).get("default"),
        "lastName": (payload.get("lastName") or {}).get("default"),
        "position": payload.get("position"),
        "birthDate": payload.get("birthDate"),
        "heightInInches": payload.get("heightInInches"),
        "weightInPounds": payload.get("weightInPounds"),
        "draftYear": draft.get("year"),
        "draftRound": draft.get("round"),
        "draftOverall": draft.get("overallPick"),
    }


def fetch_all_bios(player_ids: list[int], refresh: bool = False, workers: int = 3) -> pd.DataFrame:
    """Bios for many players, cached individually. Polite parallelism."""
    missing = [pid for pid in player_ids if refresh or not _cache_path("bio", str(pid)).exists()]
    total = len(player_ids)
    if missing:
        print(f"Fetching {len(missing)}/{total} player bios (rest cached)...")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(lambda pid: fetch_player_bio(pid, refresh), missing):
                done += 1
                if done % 500 == 0:
                    print(f"  bios: {done}/{len(missing)}")
    records = [fetch_player_bio(pid) for pid in player_ids]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def load_skater_seasons(refresh: bool = False) -> pd.DataFrame:
    """All skater regular-season totals from FIRST_DATA_SEASON onward."""
    seasons = season_window(FIRST_DATA_SEASON, LAST_BACKTEST_SEASON)
    frames = []
    for s in seasons:
        df = fetch_skater_season(s, refresh)
        df["seasonId"] = s
        frames.append(df)
        print(f"skaters {season_label(s)}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def load_team_seasons(refresh: bool = False) -> pd.DataFrame:
    seasons = season_window(FIRST_DATA_SEASON, LAST_BACKTEST_SEASON)
    frames = []
    for s in seasons:
        df = fetch_team_season(s, refresh)
        df["seasonId"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_prenhl_stats(path: Path = PRENHL_CSV) -> pd.DataFrame:
    """Pre-NHL (junior/minor/European) season stats for the rookie layer.

    Primary source: the `seasonTotals` array in the cached player bios — the
    NHL API publishes every player's junior/college/European regular-season
    stints. Historical league names are canonicalized via LEAGUE_ALIASES;
    leagues without a literature-backed NHLe factor are dropped here.

    Override: if `data/prenhl_stats.csv` exists it is used instead (e.g. an
    EliteProspects or Kaggle export) with schema:
        player_name, birth_date, season, league, team, gp, goals, assists
    where `season` is the season's start year.
    """
    cols = ["player_name", "birth_date", "season", "league", "team", "gp", "goals", "assists"]
    if path.exists():
        df = pd.read_csv(path)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return df

    from .constants import LEAGUE_ALIASES, NHLE_FACTORS

    rows = []
    for bio_file in (CACHE_DIR / "bio").glob("*.json"):
        payload = json.loads(bio_file.read_text())
        name = " ".join(
            filter(None, [(payload.get("firstName") or {}).get("default"),
                          (payload.get("lastName") or {}).get("default")])
        )
        birth = payload.get("birthDate")
        for stint in payload.get("seasonTotals", []):
            if stint.get("gameTypeId") != REGULAR_SEASON_GAME_TYPE:
                continue
            league = str(stint.get("leagueAbbrev", "")).upper()
            league = LEAGUE_ALIASES.get(league, league)
            if league not in NHLE_FACTORS or league == "NHL":
                continue
            if not stint.get("gamesPlayed"):
                continue
            team = stint.get("teamName")
            rows.append({
                "player_name": name,
                "birth_date": birth,
                "season": int(stint["season"]) // 10000,
                "league": league,
                "team": team.get("default") if isinstance(team, dict) else team,
                "gp": stint["gamesPlayed"],
                "goals": stint.get("goals", 0),
                "assists": stint.get("assists", 0),
            })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull and cache NHL data.")
    parser.add_argument("--refresh", action="store_true", help="ignore cache and re-pull")
    args = parser.parse_args()

    skaters = load_skater_seasons(refresh=args.refresh)
    teams = load_team_seasons(refresh=args.refresh)
    team_meta = fetch_team_meta(refresh=args.refresh)
    bios = fetch_all_bios(sorted(skaters["playerId"].unique().tolist()), refresh=args.refresh)
    print(f"Done: {len(skaters)} skater-seasons, {len(teams)} team-seasons, {len(team_meta)} teams, {len(bios)} bios.")


if __name__ == "__main__":
    main()
