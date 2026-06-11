"""Emit a pre-filled, match_odds.csv-shaped sheet for manual odds entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_fixtures
from odds_priors import MATCH_ODDS_COLUMNS


SHEET_CONTEXT_COLUMNS = ["match_id", "date", "time", "group", "team_a", "team_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a match_odds.csv-shaped entry sheet for the coming fixtures. "
            "Fill the odds columns by hand, then paste the rows into data/raw/match_odds.csv "
            "(or pass --output data/raw/match_odds.csv to write there directly)."
        )
    )
    parser.add_argument("--from", dest="from_date", required=True, help="First fixture date, e.g. 2026-06-11.")
    parser.add_argument("--days", type=int, default=7, help="Number of days of fixtures to include. Defaults to 7.")
    parser.add_argument(
        "--bookmakers",
        default="",
        help="Optional comma-separated bookmaker names; one entry row is emitted per book per match.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to data/odds_entry/odds_sheet_<from>_<days>d.csv.",
    )
    return parser.parse_args()


def build_odds_sheet(fixtures: pd.DataFrame, from_date: str, days: int, bookmakers: list[str]) -> pd.DataFrame:
    """Return blank odds rows for every group fixture inside the date window."""
    if days < 1:
        raise ValueError("--days must be at least 1.")
    start = pd.Timestamp(from_date)
    end = start + pd.Timedelta(days=days - 1)
    window = fixtures[
        fixtures["stage"].eq("group_stage")
        & fixtures["date"].between(start, end)
        & ~fixtures["has_placeholder_team"]
    ].sort_values(["date", "match_id"])
    if window.empty:
        raise ValueError(f"No resolved group fixtures between {start.date()} and {end.date()}.")
    rows = []
    for fixture in window.itertuples(index=False):
        for bookmaker in bookmakers or [""]:
            row = {column: "" for column in SHEET_CONTEXT_COLUMNS + MATCH_ODDS_COLUMNS[4:]}
            row.update(
                {
                    "match_id": fixture.match_id,
                    "date": fixture.date.date().isoformat(),
                    "time": fixture.time,
                    "group": fixture.group,
                    "team_a": fixture.team_a,
                    "team_b": fixture.team_b,
                    "bookmaker": bookmaker,
                    "market_type": "1x2",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=SHEET_CONTEXT_COLUMNS + MATCH_ODDS_COLUMNS[4:])


def main() -> None:
    args = parse_args()
    fixtures = load_fixtures(PROJECT_ROOT / "data" / "raw" / "fixtures.csv")
    bookmakers = [book.strip() for book in args.bookmakers.split(",") if book.strip()]
    sheet = build_odds_sheet(fixtures, args.from_date, args.days, bookmakers)
    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "odds_entry" / f"odds_sheet_{args.from_date}_{args.days}d.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(output, index=False)
    print(f"Wrote {len(sheet)} blank odds row(s) covering {sheet['match_id'].nunique()} fixture(s) to {output}")
    print("Fill odds_team_a/odds_draw/odds_team_b (decimal). over_under_line, odds_over, odds_under, spread are optional.")
    print("Leave source_timestamp, hours_before_kickoff, and implied_prob_* blank: the scripts stamp them automatically.")


if __name__ == "__main__":
    main()
