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
from collections import Counter
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BALLDONTLIE_API_KEY_ENV  # noqa: E402  (also loads the root .env)
from data_loader import apply_team_mapping, load_team_name_mapping  # noqa: E402


DEFAULT_BASE_URL = os.environ.get(
    "BALLDONTLIE_BASE_URL", "https://api.balldontlie.io/fifa/worldcup/v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "historical_match_odds.csv"
WORLD_CUP_SEASONS = (2018, 2022)
REQUEST_PAUSE_SECONDS = 15.0
MINIMUM_SAFE_PAUSE_SECONDS = 12.5
RESULTS_PATH = PROJECT_ROOT / "data" / "raw" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical World Cup closing odds from BALLDONTLIE.")
    parser.add_argument("--seasons", type=int, nargs="+", default=list(WORLD_CUP_SEASONS), help="World Cup years.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="BALLDONTLIE FIFA API base URL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Destination CSV path.")
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=REQUEST_PAUSE_SECONDS,
        help=f"Minimum delay between request starts. Values below {MINIMUM_SAFE_PAUSE_SECONDS} are rejected.",
    )
    return parser.parse_args()


def api_key() -> str:
    key = os.environ.get(BALLDONTLIE_API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"{BALLDONTLIE_API_KEY_ENV} is not set. Add it to the environment or the git-ignored root .env file. "
            "Never hardcode or commit API keys."
        )
    return key


class ConservativeRateLimiter:
    """Keep request starts below five per minute across every endpoint and page."""

    def __init__(self, pause_seconds: float):
        if pause_seconds < MINIMUM_SAFE_PAUSE_SECONDS:
            raise ValueError(
                f"request pause must be at least {MINIMUM_SAFE_PAUSE_SECONDS} seconds; "
                f"received {pause_seconds}"
            )
        self.pause_seconds = pause_seconds
        self.last_request_started: float | None = None

    def wait(self) -> None:
        if self.last_request_started is not None:
            elapsed = time.monotonic() - self.last_request_started
            remaining = self.pause_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_started = time.monotonic()


def get_paginated(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    limiter: ConservativeRateLimiter,
    request_stats: Counter,
) -> list[dict]:
    """Walk BALLDONTLIE cursor pagination and return all data entries."""
    entries: list[dict] = []
    cursor: object = None
    while True:
        page_params = dict(params)
        page_params["per_page"] = 100
        if cursor is not None:
            page_params["cursor"] = cursor
        limiter.wait()
        request_stats["requests_attempted"] += 1
        try:
            response = session.get(url, params=page_params, timeout=30)
        except requests.RequestException:
            request_stats["request_transport_errors"] += 1
            raise
        if response.status_code in (401, 403):
            request_stats[f"http_{response.status_code}"] += 1
            raise SystemExit(f"BALLDONTLIE rejected the API key (HTTP {response.status_code}).")
        if response.status_code == 429:
            request_stats["http_429"] += 1
            raise SystemExit("BALLDONTLIE rate limit reached (HTTP 429); fetch stopped without fabricating data.")
        if not response.ok:
            request_stats[f"http_{response.status_code}"] += 1
        response.raise_for_status()
        payload = response.json()
        entries.extend(payload.get("data", []))
        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            return entries


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


def american_to_decimal(value: object) -> float:
    """Convert BALLDONTLIE American odds to decimal odds."""
    american = float(value)
    if american == 0:
        raise ValueError("American odds cannot be zero.")
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


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


def extract_odds_row(entry: dict, game: dict[str, object]) -> tuple[dict[str, object] | None, str | None]:
    """Normalize one odds payload and return an explicit rejection reason when unusable."""
    odds_a = first_present(
        entry, "moneyline_home_odds", "home_win", "odds_home_win", "home", "odds_team_a", "1"
    )
    odds_draw = first_present(entry, "moneyline_draw_odds", "draw", "odds_draw", "x")
    odds_b = first_present(
        entry, "moneyline_away_odds", "away_win", "odds_away_win", "away", "odds_team_b", "2"
    )
    if odds_a is None or odds_draw is None or odds_b is None:
        return None, "missing_complete_1x2"
    snapshot = first_present(entry, "updated_at", "timestamp", "snapshot_time", "created_at")
    if not snapshot:
        return None, "missing_snapshot_timestamp"
    try:
        snapshot_ts = pd.Timestamp(str(snapshot))
    except (TypeError, ValueError):
        return None, "invalid_snapshot_timestamp"
    kickoff: pd.Timestamp = pd.Timestamp(game["kickoff"])
    if snapshot_ts.tzinfo is None and kickoff.tzinfo is not None:
        snapshot_ts = snapshot_ts.tz_localize("UTC")
    elif snapshot_ts.tzinfo is not None and kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    if snapshot_ts >= kickoff:
        return None, "post_kickoff_snapshot"
    hours_before = float((kickoff - snapshot_ts) / pd.Timedelta(hours=1))
    try:
        decimal_a = american_to_decimal(odds_a)
        decimal_draw = american_to_decimal(odds_draw)
        decimal_b = american_to_decimal(odds_b)
    except (TypeError, ValueError):
        return None, "invalid_moneyline_odds"
    total_over = first_present(entry, "total_over_odds", "over", "odds_over")
    total_under = first_present(entry, "total_under_odds", "under", "odds_under")
    try:
        decimal_over = american_to_decimal(total_over) if total_over is not None else None
        decimal_under = american_to_decimal(total_under) if total_under is not None else None
    except (TypeError, ValueError):
        decimal_over = None
        decimal_under = None
    return {
        "match_id": f"wc_{game['game_id']}",
        "date": kickoff.date().isoformat(),
        "team_a": game["team_a"],
        "team_b": game["team_b"],
        "bookmaker": str(first_present(entry, "vendor", "bookmaker", "book") or "balldontlie_consensus"),
        "odds_team_a": decimal_a,
        "odds_draw": decimal_draw,
        "odds_team_b": decimal_b,
        "market_type": "1x2",
        "source_timestamp": snapshot_ts.isoformat(),
        "over_under_line": first_present(entry, "total_value", "over_under", "total", "over_under_line"),
        "odds_over": decimal_over,
        "odds_under": decimal_under,
        "spread": first_present(entry, "spread_home_value", "spread", "handicap"),
        "hours_before_kickoff": round(hours_before, 2),
    }, None


def load_world_cup_fixture_keys(seasons: list[int], mapping: dict[str, str]) -> set[tuple[int, object, str, str]]:
    """Return canonical date/team keys for local 2018/2022 World Cup fixtures."""
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Local historical results are required: {RESULTS_PATH}")
    results = pd.read_csv(RESULTS_PATH)
    required = {"date", "home_team", "away_team", "tournament"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results.csv is missing columns required for fixture matching: {missing}")
    results["date"] = pd.to_datetime(results["date"], errors="raise").dt.date
    results["year"] = pd.to_datetime(results["date"]).dt.year
    results = results[
        results["year"].isin(seasons) & results["tournament"].astype(str).eq("FIFA World Cup")
    ].copy()
    results = results.rename(columns={"home_team": "team_a", "away_team": "team_b"})
    results = apply_team_mapping(results, ["team_a", "team_b"], mapping)
    keys: set[tuple[int, object, str, str]] = set()
    for row in results.itertuples(index=False):
        keys.add((int(row.year), row.date, str(row.team_a), str(row.team_b)))
        keys.add((int(row.year), row.date, str(row.team_b), str(row.team_a)))
    return keys


def print_fetch_summary(
    request_stats: Counter,
    raw_odds_rows: int,
    retained_rows: int,
    closing_rows: int,
    matched_world_cup_games: int,
    matched_odds_rows: int,
    odds_rejections: Counter,
    game_rejections: Counter,
    years_covered: list[int] | None = None,
    market_types: list[str] | None = None,
) -> None:
    """Print sanitized acquisition accounting without exposing request headers or secrets."""
    print(f"requests_attempted: {request_stats['requests_attempted']}")
    print(f"raw_odds_rows_fetched: {raw_odds_rows}")
    print(f"pre_kickoff_rows_retained_before_closing_dedup: {retained_rows}")
    print(f"closing_rows_written: {closing_rows}")
    print(f"world_cup_games_matched: {matched_world_cup_games}")
    print(f"odds_rows_matched_to_world_cup_fixtures: {matched_odds_rows}")
    print(f"rows_rejected_total: {sum(odds_rejections.values())}")
    for reason, count in sorted(odds_rejections.items()):
        print(f"rejected_{reason}: {count}")
    for reason, count in sorted(game_rejections.items()):
        print(f"rejected_game_{reason}: {count}")
    print(f"years_covered: {years_covered or []}")
    print(f"market_types_available: {', '.join(market_types or []) or 'none'}")
    access_errors = sum(count for key, count in request_stats.items() if key != "requests_attempted")
    print(f"api_access_errors: {access_errors}")


def main() -> None:
    args = parse_args()
    limiter = ConservativeRateLimiter(args.request_pause_seconds)
    session = requests.Session()
    session.headers["Authorization"] = api_key()
    request_stats: Counter = Counter()
    odds_rejections: Counter = Counter()
    game_rejections: Counter = Counter()
    mapping = load_team_name_mapping()
    world_cup_fixture_keys = load_world_cup_fixture_keys(args.seasons, mapping)
    rows: list[dict[str, object]] = []
    raw_odds_rows = 0
    matched_world_cup_games = 0
    for season in args.seasons:
        print(f"Fetching {season} World Cup matches...")
        game_entries = get_paginated(
            session, f"{args.base_url}/matches", {"seasons[]": season}, limiter, request_stats
        )
        games = []
        for entry in game_entries:
            game = extract_game(entry)
            if game is None:
                game_rejections["missing_kickoff_team_or_id"] += 1
                continue
            mapped = apply_team_mapping(
                pd.DataFrame([{"team_a": game["team_a"], "team_b": game["team_b"]}]),
                ["team_a", "team_b"],
                mapping,
            ).iloc[0]
            game["team_a"] = mapped["team_a"]
            game["team_b"] = mapped["team_b"]
            key = (season, pd.Timestamp(game["kickoff"]).date(), str(game["team_a"]), str(game["team_b"]))
            if key not in world_cup_fixture_keys:
                game_rejections["not_matched_to_local_world_cup_fixture"] += 1
                continue
            games.append(game)
        matched_world_cup_games += len(games)
        print(f"  {len(game_entries)} raw match rows; {len(games)} matched local World Cup fixtures.")
        games_by_id = {str(game["game_id"]): game for game in games}
        odds_entries = get_paginated(
            session, f"{args.base_url}/odds", {"seasons[]": season}, limiter, request_stats
        )
        raw_odds_rows += len(odds_entries)
        for entry in odds_entries:
            odds_match_id = first_present(entry, "match_id", "game_id")
            game = games_by_id.get(str(odds_match_id))
            if game is None:
                odds_rejections["odds_match_not_in_local_world_cup_fixtures"] += 1
                continue
            row, rejection_reason = extract_odds_row(entry, game)
            if rejection_reason:
                odds_rejections[rejection_reason] += 1
                continue
            row["world_cup_year"] = season
            rows.append(row)
        print(f"  collected {len(rows)} pre-kickoff odds rows so far.")
    if not rows:
        print_fetch_summary(
            request_stats,
            raw_odds_rows,
            0,
            0,
            matched_world_cup_games,
            0,
            odds_rejections,
            game_rejections,
        )
        raise SystemExit(
            "No usable pre-kickoff odds rows were returned. Check the API plan, base URL "
            "(--base-url / BALLDONTLIE_BASE_URL), and that the FIFA endpoints expose odds."
        )
    retained_before_closing = len(rows)
    table = pd.DataFrame(rows)
    # Keep only the latest pre-kickoff snapshot per match and bookmaker (closing line).
    table = table.sort_values("source_timestamp").drop_duplicates(
        subset=["match_id", "bookmaker"], keep="last"
    )
    table = apply_team_mapping(table, ["team_a", "team_b"], mapping)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    closing_duplicates_removed = retained_before_closing - len(table)
    odds_rejections["older_pre_kickoff_snapshot_replaced_by_closing_line"] += closing_duplicates_removed
    market_types = ["1X2"]
    if table[["over_under_line", "odds_over", "odds_under"]].notna().any(axis=None):
        market_types.append("over-under")
    if table["spread"].notna().any():
        market_types.append("spread")
    print(f"Wrote {len(table)} closing-odds rows to {output}")
    print_fetch_summary(
        request_stats,
        raw_odds_rows,
        retained_before_closing,
        len(table),
        matched_world_cup_games,
        retained_before_closing,
        odds_rejections,
        game_rejections,
        years_covered=sorted(table["world_cup_year"].astype(int).unique().tolist()),
        market_types=market_types,
    )
    print("Run scripts/run_backtest.py to evaluate the market-blend promotion gate against these odds.")


if __name__ == "__main__":
    main()
