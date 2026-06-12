"""Create manual discovery links for free 2022 World Cup odds archives."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_odds_2022 import load_2022_backtest  # noqa: E402


SOURCES = {
    "betexplorer": "betexplorer.com",
    "aiscore": "aiscore.com",
    "oddsportal": "oddsportal.com",
}


def main() -> None:
    fixtures = load_2022_backtest(PROJECT_ROOT / "outputs" / "backtest_predictions.csv")
    rows = []
    for match in fixtures.itertuples(index=False):
        for source, domain in SOURCES.items():
            query = quote_plus(f"site:{domain} {match.team_a} {match.team_b} World Cup 2022 odds")
            rows.append(
                {
                    "year": 2022,
                    "match_id": int(match.shadow_match_id),
                    "date": match.date.date().isoformat(),
                    "phase": match.penka_phase,
                    "team_a": match.team_a,
                    "team_b": match.team_b,
                    "source": source,
                    "candidate_search_url": f"https://www.google.com/search?q={query}",
                    "cache_filename": f"2022_{int(match.shadow_match_id):03d}_{source}.html",
                    "status": "manual_check_required",
                    "notes": "Save only a publicly accessible page to data/raw/odds_pages_cache; do not bypass access controls.",
                }
            )
    output = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "odds_2022_source_discovery.csv", index=False)
    with pd.ExcelWriter(out_dir / "odds_2022_source_discovery.xlsx", engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="source_discovery", index=False)
    print(f"Wrote {len(output)} manual source-discovery rows for {output['match_id'].nunique()} fixtures.")


if __name__ == "__main__":
    main()
