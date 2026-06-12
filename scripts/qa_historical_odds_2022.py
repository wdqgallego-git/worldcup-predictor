"""QA the isolated 2022 historical-odds shadow-test input."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_odds_2022 import HISTORICAL_ODDS_2022_COLUMNS, assign_backtest_identity, load_2022_backtest  # noqa: E402


INPUT = PROJECT_ROOT / "data" / "raw" / "historical_match_odds_2022.csv"
OUTPUT = PROJECT_ROOT / "outputs" / "historical_odds_2022_qa.txt"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    backtests = load_2022_backtest(PROJECT_ROOT / "outputs" / "backtest_predictions.csv")
    if not INPUT.exists():
        lines = [
            "Historical Odds 2022 QA", "=======================", "status: no_data",
            "rows_loaded: 0", "matches_covered: 0", "group_stage_coverage: 0 of 48",
            "all_tournament_coverage: 0 of 64", "odds_status_flags: missing_input_file=1",
            "sufficient_for_shadow_test: False", "reason: data/raw/historical_match_odds_2022.csv does not exist.",
        ]
        OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(OUTPUT.read_text(encoding="utf-8"))
        return
    odds = pd.read_csv(INPUT)
    missing = sorted(set(HISTORICAL_ODDS_2022_COLUMNS) - set(odds.columns))
    flags: Counter = Counter()
    if missing:
        flags["missing_required_columns"] = len(missing)
    if odds.empty:
        flags["empty_file"] += 1
    if not odds.empty and not missing:
        odds["date"] = pd.to_datetime(odds["date"], errors="coerce").dt.normalize()
        for column in ("odds_team_a", "odds_draw", "odds_team_b"):
            odds[column] = pd.to_numeric(odds[column], errors="coerce")
        invalid_1x2 = odds[["odds_team_a", "odds_draw", "odds_team_b"]].isna().any(axis=1)
        flags["missing_1x2"] = int(invalid_1x2.sum())
        invalid_decimal = odds[["odds_team_a", "odds_draw", "odds_team_b"]].le(1.0).any(axis=1)
        flags["invalid_decimal_odds"] = int(invalid_decimal.sum())
        overround = 1.0 / odds["odds_team_a"] + 1.0 / odds["odds_draw"] + 1.0 / odds["odds_team_b"] - 1.0
        flags["suspect_overround"] = int(((overround < -0.02) | (overround > 0.30)).sum())
        flags["missing_source_name"] = int(odds["source_name"].fillna("").astype(str).str.strip().eq("").sum())
        flags["missing_source_url"] = int(odds["source_url"].fillna("").astype(str).str.strip().eq("").sum())
        flags["missing_confidence"] = int(odds["confidence"].fillna("").astype(str).str.strip().eq("").sum())
        flags["placeholder_rows"] = int(odds.astype(str).apply(lambda column: column.str.contains("PLACEHOLDER", case=False, na=False)).any(axis=1).sum())
        hours = pd.to_numeric(odds["hours_before_kickoff"], errors="coerce")
        flags["post_kickoff_leakage"] = int(hours.lt(0).sum())
        timestamps = pd.to_datetime(odds["source_timestamp"], errors="coerce", utc=True)
        if odds["kickoff_time"].notna().any():
            kickoff = pd.to_datetime(
                odds["date"].astype(str) + " " + odds["kickoff_time"].astype(str),
                errors="coerce",
                utc=True,
            )
            flags["post_kickoff_timestamp"] = int(
                (timestamps.notna() & kickoff.notna() & timestamps.ge(kickoff)).sum()
            )
        matched, unmatched = assign_backtest_identity(odds, backtests)
        matched_ids = set(pd.to_numeric(matched["match_id"], errors="coerce").dropna().astype(int))
        covered_backtests = backtests[backtests["shadow_match_id"].isin(matched_ids)]
        group_count = int(covered_backtests["penka_phase"].eq("group").sum())
        match_count = int(len(covered_backtests))
        valid = sum(flags.values()) == 0 and match_count >= 30
        unmatched_matches = unmatched[["date", "team_a", "team_b"]].drop_duplicates().astype(str)
        known_teams = set(backtests["team_a"]) | set(backtests["team_b"])
        unmatched_teams = sorted(
            (set(odds["team_a"].dropna().astype(str)) | set(odds["team_b"].dropna().astype(str)))
            - known_teams
        )
        distributions = odds.groupby(["source_name", "bookmaker"], dropna=False).size().sort_values(ascending=False)
        lines = [
            "Historical Odds 2022 QA", "=======================",
            f"status: {'usable' if valid else 'not_usable'}", f"rows_loaded: {len(odds)}",
            f"matches_covered: {match_count}", f"group_stage_coverage: {group_count} of 48",
            f"all_tournament_coverage: {match_count} of 64",
            "odds_status_flags: " + (", ".join(f"{key}={value}" for key, value in flags.items() if value) or "ok=all_rows"),
            f"unmatched_matches: {len(unmatched_matches)}",
            f"unmatched_teams: {', '.join(unmatched_teams) if unmatched_teams else 'none'}",
            f"overround_min: {overround.min():.4f}" if len(overround) else "overround_min: n/a",
            f"overround_median: {overround.median():.4f}" if len(overround) else "overround_median: n/a",
            f"overround_max: {overround.max():.4f}" if len(overround) else "overround_max: n/a",
            "bookmaker_source_distribution:",
            *(f"  {source} / {book}: {count}" for (source, book), count in distributions.items()),
            "unmatched_match_examples:",
            *("  " + " | ".join(row) for row in unmatched_matches.head(20).itertuples(index=False, name=None)),
            f"sufficient_for_shadow_test: {valid}",
            "thin_sample_warning: fewer than 30 matched matches" if match_count < 30 else "thin_sample_warning: none",
        ]
    else:
        lines = [
            "Historical Odds 2022 QA", "=======================", "status: not_usable",
            f"rows_loaded: {len(odds)}", "matches_covered: 0", "group_stage_coverage: 0 of 48",
            "all_tournament_coverage: 0 of 64",
            "odds_status_flags: " + (", ".join(f"{key}={value}" for key, value in flags.items()) or "none"),
            "sufficient_for_shadow_test: False",
        ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
