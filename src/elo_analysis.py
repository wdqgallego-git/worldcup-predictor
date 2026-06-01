"""Leakage-safe causal Elo winning-margin and fixed-score diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prediction_optimizer import points_for_prediction
from rule_b_analysis import add_causal_elo_features


ELO_BUCKETS = {
    "bins": [0, 50, 100, 150, 200, 250, 300, np.inf],
    "labels": ["0_50", "50_100", "100_150", "150_200", "200_250", "250_300", "300_plus"],
}
FIXED_SCORE_ACTIONS = {
    "favorite_1_0": (1, 0),
    "favorite_2_0": (2, 0),
    "favorite_2_1": (2, 1),
    "favorite_3_0": (3, 0),
    "draw_1_1": (1, 1),
    "draw_0_0": (0, 0),
    "underdog_1_0": (0, 1),
}


def add_elo_margin_features(results: pd.DataFrame) -> pd.DataFrame:
    """Return Elo-oriented historical outcomes using pre-match ratings only."""
    matches, _ = add_causal_elo_features(results)
    matches = matches[
        (matches["date"] >= pd.Timestamp("2000-01-01"))
        & matches["goals_a"].notna()
        & matches["goals_b"].notna()
        & np.isfinite(pd.to_numeric(matches["goals_a"], errors="coerce"))
        & np.isfinite(pd.to_numeric(matches["goals_b"], errors="coerce"))
        & matches["pre_elo_a"].ne(matches["pre_elo_b"])
    ].copy()
    matches["favorite_is_a"] = matches["pre_elo_a"] > matches["pre_elo_b"]
    matches["favorite_team"] = np.where(matches["favorite_is_a"], matches["team_a"], matches["team_b"])
    matches["underdog_team"] = np.where(matches["favorite_is_a"], matches["team_b"], matches["team_a"])
    matches["elo_diff"] = (matches["pre_elo_a"] - matches["pre_elo_b"]).abs()
    matches["favorite_goals"] = np.where(matches["favorite_is_a"], matches["goals_a"], matches["goals_b"])
    matches["underdog_goals"] = np.where(matches["favorite_is_a"], matches["goals_b"], matches["goals_a"])
    matches["favorite_margin"] = matches["favorite_goals"] - matches["underdog_goals"]
    matches["favorite_win"] = matches["favorite_margin"].gt(0)
    matches["draw"] = matches["favorite_margin"].eq(0)
    matches["upset"] = matches["favorite_margin"].lt(0)
    matches["favorite_win_by_1"] = matches["favorite_margin"].eq(1)
    matches["favorite_win_by_2"] = matches["favorite_margin"].eq(2)
    matches["favorite_win_by_3_plus"] = matches["favorite_margin"].ge(3)
    matches["total_goals"] = matches["goals_a"] + matches["goals_b"]
    matches["elo_bucket"] = pd.cut(
        matches["elo_diff"],
        bins=ELO_BUCKETS["bins"],
        labels=ELO_BUCKETS["labels"],
        right=False,
        include_lowest=True,
    )
    return matches


def summarize_elo_margin_distribution(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize favorite outcomes by causal Elo severity bucket."""
    rows = []
    for bucket, bucket_rows in matches.groupby("elo_bucket", observed=True):
        rows.append(
            {
                "elo_bucket": str(bucket),
                "match_count": len(bucket_rows),
                "favorite_win_rate": float(bucket_rows["favorite_win"].mean()),
                "draw_rate": float(bucket_rows["draw"].mean()),
                "upset_rate": float(bucket_rows["upset"].mean()),
                "favorite_win_by_1_rate": float(bucket_rows["favorite_win_by_1"].mean()),
                "favorite_win_by_2_rate": float(bucket_rows["favorite_win_by_2"].mean()),
                "favorite_win_by_3_plus_rate": float(bucket_rows["favorite_win_by_3_plus"].mean()),
                "avg_favorite_margin": float(bucket_rows["favorite_margin"].mean()),
                "avg_total_goals": float(bucket_rows["total_goals"].mean()),
            }
        )
    return pd.DataFrame(rows)


def oriented_action_points(rows: pd.DataFrame, favorite_goals: int, underdog_goals: int) -> np.ndarray:
    """Return empirical contest points for one favorite-oriented score action."""
    return np.asarray(
        [
            points_for_prediction(favorite_goals, underdog_goals, int(actual_favorite), int(actual_underdog))
            for actual_favorite, actual_underdog in zip(rows["favorite_goals"], rows["underdog_goals"])
        ],
        dtype=int,
    )


def summarize_fixed_score_ev(matches: pd.DataFrame) -> pd.DataFrame:
    """Report empirical fixed-score EV for each Elo bucket."""
    report_rows = []
    for bucket, bucket_rows in matches.groupby("elo_bucket", observed=True):
        for action, (favorite_goals, underdog_goals) in FIXED_SCORE_ACTIONS.items():
            points = oriented_action_points(bucket_rows, favorite_goals, underdog_goals)
            report_rows.append(
                {
                    "elo_bucket": str(bucket),
                    "action": action,
                    "match_count": len(bucket_rows),
                    "avg_contest_points": float(points.mean()),
                    "exact_score_rate": float(
                        (
                            bucket_rows["favorite_goals"].eq(favorite_goals)
                            & bucket_rows["underdog_goals"].eq(underdog_goals)
                        ).mean()
                    ),
                    "correct_result_rate": float(
                        (np.sign(favorite_goals - underdog_goals) == np.sign(bucket_rows["favorite_margin"])).mean()
                    ),
                    "correct_goal_difference_rate": float(
                        ((favorite_goals - underdog_goals) == bucket_rows["favorite_margin"]).mean()
                    ),
                }
            )
    return pd.DataFrame(report_rows)


def write_elo_summary(distribution: pd.DataFrame, fixed_score_ev: pd.DataFrame, path: str | Path) -> None:
    """Write the causal Elo margin and score-rule decision summary."""
    pivot = fixed_score_ev.pivot_table(
        index=["elo_bucket", "match_count"],
        columns="action",
        values="avg_contest_points",
    ).reset_index()
    pivot["best_action"] = pivot[list(FIXED_SCORE_ACTIONS)].idxmax(axis=1)
    pivot["favorite_2_0_minus_1_0"] = pivot["favorite_2_0"] - pivot["favorite_1_0"]
    stable_two_nil = pivot[(pivot["match_count"] >= 100) & pivot["favorite_2_0_minus_1_0"].gt(0)]
    close_elo = pivot[pivot["elo_bucket"].isin(["0_50", "50_100"])]
    draw_beats_favorite = bool(
        (
            close_elo[["draw_1_1", "draw_0_0"]].max(axis=1)
            > close_elo["favorite_1_0"]
        ).any()
    )
    lines = [
        "Causal Elo Margin Summary",
        "=========================",
        "Scoring note: full-history rows use the real Penka group-stage scoring approximation.",
        "Elo construction: initialize at 1500, store pre-match ratings, then update after each result.",
        f"matches_used_since_2000: {int(distribution['match_count'].sum())}",
        "",
        f"Does 2-0 beat 1-0 in any Elo bucket with at least 100 matches? {not stable_two_nil.empty}",
        f"Do draw predictions beat favorite_1_0 in any close Elo bucket? {draw_beats_favorite}",
        "",
        "Best empirical fixed-score action by Elo bucket:",
        pivot[
            ["elo_bucket", "match_count", "best_action", "favorite_1_0", "favorite_2_0", "favorite_3_0"]
        ].to_string(index=False),
        "",
        "Margin distribution by Elo bucket:",
        distribution.to_string(index=False),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_elo_analysis(results: pd.DataFrame, output_dir: str | Path = "outputs") -> dict[str, pd.DataFrame]:
    """Build causal Elo diagnostics and save CSV and text outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matches = add_elo_margin_features(results)
    distribution = summarize_elo_margin_distribution(matches)
    fixed_score_ev = summarize_fixed_score_ev(matches)
    distribution.to_csv(output_path / "elo_margin_distribution.csv", index=False)
    fixed_score_ev.to_csv(output_path / "elo_fixed_score_ev.csv", index=False)
    write_elo_summary(distribution, fixed_score_ev, output_path / "elo_margin_summary.txt")
    return {
        "elo_matches": matches,
        "elo_margin_distribution": distribution,
        "elo_fixed_score_ev": fixed_score_ev,
    }
