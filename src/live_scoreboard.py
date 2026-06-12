"""Live 2026 group-stage scoreboard: real points for the submitted picks and rivals.

Results are entered by hand into data/raw/live_results_2026.csv as matches finish.
The scoreboard scores the submitted picks plus every counterfactual strategy under
real Penka rules, projects final totals, and frames the matchday-3 risk decision.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import orient_favorite_score
from penka_scoring import GROUP, prediction_category, score_prediction


LIVE_RESULTS_PATH = Path("data/raw/live_results_2026.csv")
LIVE_RESULTS_COLUMNS = [
    "match_id", "date", "time", "team_a", "team_b",
    "goals_a", "goals_b", "submitted_prediction",
]
_SCORE_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

# Per-match priors for projecting unplayed matches, from the expanded backtest's
# wc_only_group scope (outputs/expanded_strategy_comparison.csv). Priors, not promises.
PROJECTION_PRIORS = {
    "submitted": 1.75,
    "model_safe_ev": 1.75,
    "model_aggressive": 1.70,
    "favorite_1_0": 1.757,
    "favorite_2_1": 1.799,
    "market_blended": 1.75,
    "market_favorite_2_1": 1.799,
}


def parse_score(text: object) -> tuple[int, int] | None:
    """Parse a '2-1'-style scoreline; None for blanks, error for garbage."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == "":
        return None
    match = _SCORE_PATTERN.match(str(text))
    if match is None:
        raise ValueError(f"Unparseable scoreline: {text!r}. Expected e.g. '2-1'.")
    return int(match.group(1)), int(match.group(2))


def build_results_template(recommendations: pd.DataFrame, final_predictions: pd.DataFrame) -> pd.DataFrame:
    """Pre-filled entry sheet: fill goals_a/goals_b after each match ends."""
    schedule = final_predictions[["match_id", "date", "time", "team_a", "team_b"]].copy()
    template = schedule.merge(recommendations[["match_id"]], on="match_id", how="inner")
    template["goals_a"] = ""
    template["goals_b"] = ""
    template["submitted_prediction"] = ""  # blank = the recommended pick was submitted
    return template[LIVE_RESULTS_COLUMNS].sort_values(["date", "match_id"], kind="stable")


def _strategy_picks(recommendations: pd.DataFrame, final_predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per match with each strategy's pick; market gaps fall back honestly."""
    picks = recommendations[
        ["match_id", "team_a", "team_b", "rank_a", "rank_b", "recommended_prediction",
         "safe_prediction", "favorite_1_0_prediction", "market_blended_prediction",
         "market_favorite_2_1_prediction"]
    ].merge(
        final_predictions[["match_id", "aggressive_prediction", "expected_penka_points"]],
        on="match_id",
        how="left",
    )
    favorite_2_1 = [
        "{}-{}".format(*orient_favorite_score(rank_a, rank_b, 2, 1))
        for rank_a, rank_b in zip(picks["rank_a"], picks["rank_b"])
    ]
    picks["favorite_2_1_prediction"] = favorite_2_1
    # A match without odds has no market pick; the market strategies fall back to
    # their non-market analog, mirroring the backtest fallback rules.
    picks["market_blended_prediction"] = picks["market_blended_prediction"].fillna(picks["safe_prediction"])
    picks["market_favorite_2_1_prediction"] = picks["market_favorite_2_1_prediction"].fillna(
        picks["favorite_2_1_prediction"]
    )
    return picks


STRATEGY_COLUMNS = {
    "submitted": "submitted_prediction_effective",
    "model_safe_ev": "safe_prediction",
    "model_aggressive": "aggressive_prediction",
    "favorite_1_0": "favorite_1_0_prediction",
    "favorite_2_1": "favorite_2_1_prediction",
    "market_blended": "market_blended_prediction",
    "market_favorite_2_1": "market_favorite_2_1_prediction",
}


def score_live_matches(
    live_results: pd.DataFrame,
    recommendations: pd.DataFrame,
    final_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-match points table, per-strategy standings with projection)."""
    picks = _strategy_picks(recommendations, final_predictions)
    board = live_results.merge(picks, on="match_id", how="left", suffixes=("", "_pick"))
    board["submitted_prediction_effective"] = board["submitted_prediction"].replace("", np.nan)
    board["submitted_prediction_effective"] = board["submitted_prediction_effective"].fillna(
        board["recommended_prediction"]
    )
    def _actual_goals(goals_a: object, goals_b: object, match_id: object) -> tuple[int, int] | None:
        blank_a = pd.isna(goals_a) or str(goals_a).strip() == ""
        blank_b = pd.isna(goals_b) or str(goals_b).strip() == ""
        if blank_a and blank_b:
            return None
        if blank_a or blank_b:
            raise ValueError(f"match {match_id}: enter both goals_a and goals_b, or neither.")
        return int(float(goals_a)), int(float(goals_b))

    actuals = [
        _actual_goals(goals_a, goals_b, match_id)
        for goals_a, goals_b, match_id in zip(board["goals_a"], board["goals_b"], board["match_id"])
    ]
    board["played"] = [actual is not None for actual in actuals]
    for strategy, column in STRATEGY_COLUMNS.items():
        points, categories = [], []
        for actual, pick_text in zip(actuals, board[column]):
            pick = parse_score(pick_text)
            if actual is None or pick is None:
                points.append(np.nan)
                categories.append("")
                continue
            points.append(score_prediction(pick[0], pick[1], actual[0], actual[1], GROUP))
            categories.append(prediction_category(pick[0], pick[1], actual[0], actual[1]))
        board[f"{strategy}_points"] = points
        if strategy == "submitted":
            board["submitted_category"] = categories
    played = board[board["played"]]
    remaining = int((~board["played"]).sum())
    standings = []
    for strategy in STRATEGY_COLUMNS:
        scored = played[f"{strategy}_points"].dropna()
        total = float(scored.sum())
        prior = PROJECTION_PRIORS.get(strategy, 1.75)
        standings.append(
            {
                "strategy": strategy,
                "matches_played": int(len(scored)),
                "points_so_far": int(total),
                "average_so_far": float(scored.mean()) if len(scored) else np.nan,
                "remaining_matches": remaining,
                "projection_prior_per_match": prior,
                "projected_final_points": float(total + remaining * prior),
            }
        )
    standings_table = pd.DataFrame(standings).sort_values("points_so_far", ascending=False, kind="stable")
    return board, standings_table


def write_scoreboard(
    board: pd.DataFrame,
    standings: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> Path:
    """Write the per-match board, standings, and a blunt matchday-3 framing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    board.to_csv(output_path / "live_scoreboard.csv", index=False)
    standings.to_csv(output_path / "live_scoreboard_standings.csv", index=False)
    played = board[board["played"]]
    submitted = standings.set_index("strategy").loc["submitted"]
    best = standings.iloc[0]
    lines = [
        "Live 2026 Group-Stage Scoreboard",
        "================================",
        f"matches_played: {len(played)} of {len(board)}",
        f"submitted_points: {int(submitted['points_so_far'])} "
        f"(avg {submitted['average_so_far']:.3f} per match)" if len(played) else "submitted_points: 0 (no results entered yet)",
        "",
        "Standings (points so far | projected final using WC-backtest priors):",
    ]
    for row in standings.itertuples(index=False):
        lines.append(
            f"  {row.strategy:22s} {row.points_so_far:4d} | projected {row.projected_final_points:6.1f}"
        )
    lines += [
        "",
        "Matchday-3 framing: if the submitted total is at or above the best counterfactual,",
        "stay on expected-value picks. If trailing the pool (check the Penka leaderboard,",
        "not just this table), shift remaining picks toward higher-variance exact scores",
        "(the aggressive_prediction column) - trailing entries do not catch up on chalk.",
    ]
    (output_path / "live_scoreboard_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path / "live_scoreboard_summary.txt"
