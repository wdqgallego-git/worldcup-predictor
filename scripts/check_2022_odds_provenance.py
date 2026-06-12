"""Create a light, non-bypassing provenance comparison for reference matches."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_odds_2022 import load_2022_backtest  # noqa: E402
from odds_priors import normalized_implied_probabilities  # noqa: E402


REFERENCES = [
    (2022, "Qatar", "Ecuador"),
    (2022, "Argentina", "France"),
    (2022, "Morocco", "Portugal"),
    (2018, "France", "Croatia"),
    (2018, "Germany", "Mexico"),
]
SOURCES = ("betexplorer.com", "aiscore.com", "oddsportal.com")


def consensus_probabilities(rows: pd.DataFrame) -> np.ndarray | None:
    if rows.empty:
        return None
    probabilities = []
    for row in rows.itertuples(index=False):
        try:
            probabilities.append(normalized_implied_probabilities([row.odds_team_a, row.odds_draw, row.odds_team_b]))
        except ValueError:
            continue
    return np.mean(probabilities, axis=0) if probabilities else None


def main() -> None:
    odds_path = PROJECT_ROOT / "data" / "raw" / "historical_match_odds_2022.csv"
    reference_path = PROJECT_ROOT / "data" / "raw" / "odds_pages_cache" / "reference_archive_odds.csv"
    odds = pd.read_csv(odds_path) if odds_path.exists() else pd.DataFrame()
    references = pd.read_csv(reference_path) if reference_path.exists() else pd.DataFrame()
    if not odds.empty:
        odds["date"] = pd.to_datetime(odds["date"], errors="coerce")
    rows = []
    for year, team_a, team_b in REFERENCES:
        matched = pd.DataFrame()
        if year == 2022 and not odds.empty:
            matched = odds[
                ((odds["team_a"].eq(team_a)) & (odds["team_b"].eq(team_b)))
                | ((odds["team_a"].eq(team_b)) & (odds["team_b"].eq(team_a)))
            ].copy()
        # The normalized odds file is the imported dataset regardless of whether
        # its source is Kaggle, a local HTML cache, or a public archive table.
        imported_rows = matched
        archive_rows = pd.DataFrame(columns=matched.columns)
        if not references.empty:
            archive_rows = references[
                references["year"].eq(year)
                & (
                    ((references["team_a"].eq(team_a)) & (references["team_b"].eq(team_b)))
                    | ((references["team_a"].eq(team_b)) & (references["team_b"].eq(team_a)))
                )
            ].copy()
        imported = consensus_probabilities(imported_rows)
        archive = consensus_probabilities(archive_rows)
        search_links = [
            f"https://www.google.com/search?q={quote_plus(f'site:{source} {team_a} {team_b} {year} odds')}"
            for source in SOURCES
        ]
        differences = np.abs(imported - archive) if imported is not None and archive is not None else None
        rows.append(
            {
                "year": year,
                "team_a": team_a,
                "team_b": team_b,
                "imported_prob_team_a": imported[0] if imported is not None else np.nan,
                "imported_prob_draw": imported[1] if imported is not None else np.nan,
                "imported_prob_team_b": imported[2] if imported is not None else np.nan,
                "archive_prob_team_a": archive[0] if archive is not None else np.nan,
                "archive_prob_draw": archive[1] if archive is not None else np.nan,
                "archive_prob_team_b": archive[2] if archive is not None else np.nan,
                "max_absolute_probability_difference": differences.max() if differences is not None else np.nan,
                "within_0_05_tolerance": bool(differences.max() <= 0.05) if differences is not None else pd.NA,
                "status": "checked" if differences is not None else "manual_check_required",
                "source_search_links": " | ".join(search_links),
                "notes": "No access controls were bypassed. Save a public archive page locally before parsing." if differences is None else "Compared imported archive consensus with an independently observed bookmaker line from the public match archive.",
            }
        )
    output = pd.DataFrame(rows)
    path = PROJECT_ROOT / "outputs" / "odds_2022_provenance_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
