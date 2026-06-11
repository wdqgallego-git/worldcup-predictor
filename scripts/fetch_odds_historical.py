"""Fetch 2018/2022 World Cup pre-kickoff odds from BALLDONTLIE for the backtest.

Writes data/raw/historical_match_odds.csv in the match_odds schema. Only snapshots
captured strictly before kickoff are kept (closing-line discipline), preserving the
project's leakage rules. Requires BALLDONTLIE_API_KEY in the environment or .env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BALLDONTLIE_API_KEY_ENV  # noqa: E402  (also loads the root .env)
from data_loader import apply_team_mapping, load_team_name_mapping  # noqa: E402


DEFAULT_BASE_URL = os.environ.get("BALLDONTLIE_BASE_URL", "https://api.balldontlie.io/fifa/v1")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "historical_match_odds.csv"
WORLD_CUP_SEASONS = (2018, 2022)
REQUEST_PAUSE_SECONDS = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical World Cup closing odds from BALLDONTLIE.")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(WORLD_CUP_SEASONS), help="World Cup years.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="BALLDONTLIE FIFA API base URL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination CSV path.")
    return parser.parse_args()


def api_key() -> str:
    key = os.environ.get(BALLDONTLIE_API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"{BALLDONTLIE_API_KEY_ENV} is not set. Add it to the environment or the git-ignored root .env file. "
            "Never hardcode or commit API keys."
        )
    return key


def get_paginated(session: requests.Session, url: str, params: dict[str, object]) -> list[dict]:
    """Walk BALLDONTLIE cursor pagination and return all data entries."""
    entries: list[dict] = []
    cursor: object = None
    while True:
        page_params = dict(params)
        page_params["per_page"] = 100
        if cursor is not None:
            page_params["cursor"] = cursor
        response = session.get(url, params=page_params, timeout=30)
        if response.status_code in (401, 403):
            raise SystemExit(f"BALLDONTLIE rejected the API key (HTTP {response.status_code}).")
        response.raise_for_status()
        payload = response.json()
        entries.extend(payload.get("data", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            return entries
        time.sleep(REQUEST_PAUSE_SECONDS)


def first_present(entry: dict, *keys: str) -> object:
    """Return the first non-empty value among possibly nested keys like 'home_team.name'."""
    for key in keys:
        value: object = entry
        for part in key.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        if value not in (None, ""):
            return value
    return None


def extract_game(entry: dict) -> dict[str, object] | None:
    """Normalize one game payload defensively across plausible field spellings."""
    kickoff = first_present(entry, "datetime", "date", "kickoff", "start_time")
    team_a = first_present(entry, "home_team.name", "home_team_name", "home_team", "team_a")
    team_b = first_present(entry, "away_team.name", "away_team_name", "away_team", "team_b")
    game_id = first_present(entry, "id", "game_id")
    if not (kickoff and team_a and team_b and game_id is not None):
        return None
    return {
        "game_id": game_id,
        "kickoff": pd.Timestamp(str(kickoff)),
        "team_a": str(team_a),
        "team_b": str(team_b),
    }


def extract_odds_rows(entry: dict, game: dict[str, object]) -> dict[str, object] | None:
    """Normalize one odds payload into the match_odds schema; None when unusable."""
    odds_a = first_present(entry, "home_win", "odds_home_win", "home", "odds_team_a", "1")
    odds_draw = first_present(entry, "draw", "odds_draw", "x")
    odds_b = first_present(entry, "away_win", "odds_away_win", "away", "odds_team_b", "2")
    if odds_a is None or odds_draw is None or odds_b is None:
        return None
    snapshot = first_present(entry, "updated_at", "timestamp", "snapshot_time", "created_at")
    snapshot_ts = pd.Timestamp(str(snapshot)) if snapshot else pd.NaT
    kickoff: pd.Timestamp = game["kickoff"]
    if pd.notna(snapshot_ts):
        if snapshot_ts.tzinfo is None and kickoff.tzinfo is not None:
            snapshot_ts = snapshot_ts.tz_localize("UTC")
        if snapshot_ts >= kickoff:
            return None  # post-kickoff snapshot: excluded to preserve leakage discipline
        hours_before = float((kickoff - snapshot_ts) / pd.Timedelta(hours=1))
    else:
        hours_before = None
    return {
        "match_id": f"wc_{game['game_id']}",
        "date": kickoff.date().isoformat(),
        "team_a": game["team_a"],
        "team_b": game["team_b"],
        "bookmaker": str(first_present(entry, "vendor", "bookmaker", "book") or "balldontlie_consensus"),
        "odds_team_a": float(odds_a),
        "odds_draw": float(odds_draw),
        "odds_team_b": float(odds_b),
        "market_type": "1x2",
        "source_timestamp": snapshot_ts.isoformat() if pd.notna(snapshot_ts) else "",
        "over_under_line": first_present(entry, "over_under", "total", "over_under_line"),
        "odds_over": first_present(entry, "over", "odds_over"),
        "odds_under": first_present(entry, "under", "odds_under"),
        "spread": first_present(entry, "spread", "handicap"),
        "hours_before_kickoff": round(hours_before, 2) if hours_before is not None else "",
    }


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers["Authorization"] = api_key()
    rows: list[dict[str, object]] = []
    for season in args.seasons:
        print(f"Fetching {season} World Cup games...")
        games = [extract_game(entry) for entry in get_paginated(session, f"{args.base_url}/games", {"season": season})]
        games = [game for game in games if game is not None]
        print(f"  {len(games)} games with usable kickoff/team fields.")
        for game in games:
            time.sleep(REQUEST_PAUSE_SECONDS)
            odds_entries = get_paginated(session, f"{args.base_url}/odds", {"game_id": game["game_id"]})
            usable = [extract_odds_rows(entry, game) for entry in odds_entries]
            usable = [row for row in usable if row is not None]
            for row in usable:
                row["world_cup_year"] = season
            rows.extend(usable)
        print(f"  collected {len(rows)} pre-kickoff odds rows so far.")
    if not rows:
        raise SystemExit(
            "No usable pre-kickoff odds rows were returned. Check the API plan, base URL "
            "(--base-url / BALLDONTLIE_BASE_URL), and that the FIFA endpoints expose odds."
        )
    table = pd.DataFrame(rows)
    # Keep only the latest pre-kickoff snapshot per match and bookmaker (closing line).
    table = table.sort_values("source_timestamp").drop_duplicates(
        subset=["match_id", "bookmaker"], keep="last"
    )
    mapping = load_team_name_mapping()
    table = apply_team_mapping(table, ["team_a", "team_b"], mapping)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {len(table)} closing-odds rows to {output}")
    print("Run scripts/run_backtest.py to evaluate the market-blend promotion gate against these odds.")


if __name__ == "__main__":
    main()
