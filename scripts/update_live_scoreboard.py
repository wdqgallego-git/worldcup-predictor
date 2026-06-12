"""Refresh the live 2026 scoreboard from hand-entered match results.

First run creates data/raw/live_results_2026.csv pre-filled with the 72 group
fixtures. After each match, type the final 90-minute score into goals_a/goals_b
(and optionally the actually-submitted pick if it differed), then re-run this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_scoreboard import (
    LIVE_RESULTS_COLUMNS,
    LIVE_RESULTS_PATH,
    build_results_template,
    score_live_matches,
    write_scoreboard,
)


RECOMMENDATIONS_PATH = PROJECT_ROOT / "outputs" / "final_match_recommendations.csv"
FINAL_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "final_predictions.csv"


def main() -> None:
    if not RECOMMENDATIONS_PATH.exists() or not FINAL_PREDICTIONS_PATH.exists():
        raise SystemExit("Run scripts/run_final_predictions.py first; the scoreboard scores its outputs.")
    recommendations = pd.read_csv(RECOMMENDATIONS_PATH)
    final_predictions = pd.read_csv(FINAL_PREDICTIONS_PATH)
    live_path = PROJECT_ROOT / LIVE_RESULTS_PATH
    if not live_path.exists():
        template = build_results_template(recommendations, final_predictions)
        live_path.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(live_path, index=False)
        print(f"Created results entry sheet: {live_path}")
        print("Fill goals_a/goals_b after each match (90-minute score) and re-run this script.")
        return
    live_results = pd.read_csv(live_path, dtype={"goals_a": "object", "goals_b": "object", "submitted_prediction": "object"})
    live_results = live_results.reindex(columns=LIVE_RESULTS_COLUMNS)
    board, standings = score_live_matches(live_results, recommendations, final_predictions)
    summary_path = write_scoreboard(board, standings, PROJECT_ROOT / "outputs")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
