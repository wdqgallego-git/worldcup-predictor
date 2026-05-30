"""Run the full match-level validation and backtesting audit."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtesting import WORLD_CUP_BACKTESTS, run_match_backtest, run_walk_forward_backtests
from data_loader import load_rankings, load_results
from evaluation import challenge_points
from features import build_training_table
from leakage_checks import (
    check_no_future_matches_used,
    check_no_missing_feature_values,
    check_no_placeholder_features,
    check_no_target_columns_in_features,
    check_ranking_dates_before_match,
)


AUDIT_CUTOFFS = {
    2014: "2014-06-01",
    2018: "2018-06-01",
    2022: "2022-11-01",
}
REQUIRED_BASELINES = {
    "always_1_1",
    "favorite_1_0",
    "favorite_2_0_if_strong",
    "most_likely_poisson",
    "expected_points_optimized",
}


def assert_true(condition: bool, message: str) -> None:
    """Raise a clear audit failure."""
    if not condition:
        raise AssertionError(message)


def run_leakage_audit(results: pd.DataFrame, rankings: pd.DataFrame) -> None:
    """Verify historical cutoff and feature-quality invariants."""
    training_df, feature_cols = build_training_table(results, rankings)
    check_no_target_columns_in_features(feature_cols)
    check_no_placeholder_features(feature_cols)
    forbidden_targets = {"home_score", "away_score", "actual_goals_a", "actual_goals_b"}
    assert_true(not forbidden_targets.intersection(feature_cols), "Additional target aliases leaked into feature_cols.")
    check_no_missing_feature_values(training_df, feature_cols)
    check_ranking_dates_before_match(training_df)
    for year, cutoff in AUDIT_CUTOFFS.items():
        historical_train = training_df[training_df["date"] < pd.Timestamp(cutoff)].copy()
        check_no_future_matches_used(historical_train, cutoff)
        assert_true((historical_train["date"] < pd.Timestamp(cutoff)).all(), f"{year} training cutoff failed.")
        print(f"{year} leakage cutoff passed: {len(historical_train):,} rows before {cutoff}")
    print("Feature leakage checks passed.")


def run_scoring_audit() -> None:
    """Verify exact-score and bonus scoring semantics."""
    assert_true(challenge_points(2, 1, 2, 1) == 5, "Exact score must be 5 total points.")
    assert_true(challenge_points(2, 0, 3, 0) == 3, "Correct result without goal-difference bonus must be 3 points.")
    assert_true(challenge_points(2, 1, 3, 2) == 4, "Correct goal difference must add exactly 1 point.")
    print("Challenge-point scoring checks passed.")


def build_summary_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward suite and print the requested summary table."""
    temp_dir = Path(tempfile.mkdtemp(prefix="worldcup_backtest_audit_"))
    outputs = run_walk_forward_backtests(output_dir=temp_dir)
    predictions = outputs["backtest_predictions"]
    metrics = outputs["backtest_metrics"].copy()
    baselines = outputs["backtest_baseline_comparison"].copy()
    assert_true(predictions.groupby("backtest_year").size().eq(64).all(), "Each World Cup must have exactly 64 test matches.")
    assert_true(set(metrics["probability_method"]) == {"independent", "dixon_coles"}, "Both Poisson methods are required.")
    assert_true(REQUIRED_BASELINES.issubset(set(baselines["baseline"])), "Requested baselines are missing.")
    for file_name in ("backtest_predictions.csv", "backtest_metrics.csv", "backtest_baseline_comparison.csv"):
        assert_true((temp_dir / file_name).exists(), f"Missing generated output: {file_name}")
    summary = metrics[
        [
            "year", "model_name", "probability_method", "prediction_method", "matches",
            "total_challenge_points", "average_challenge_points", "exact_score_rate",
            "correct_result_rate", "correct_goal_difference_rate",
        ]
    ].rename(columns={"average_challenge_points": "avg_challenge_points"})
    print(f"Temporary backtest outputs: {temp_dir}")
    print("\nSUMMARY TABLE")
    print(summary.to_string(index=False))
    overall = baselines.groupby("baseline", as_index=False).agg(
        total_challenge_points=("total_challenge_points", "sum"),
        matches=("year", "size"),
    )
    overall["avg_challenge_points"] = overall["total_challenge_points"] / (overall["matches"] * 64)
    winner = overall.sort_values("avg_challenge_points", ascending=False, kind="stable").iloc[0]
    print("\nBASELINE WINNER")
    print(overall.sort_values("avg_challenge_points", ascending=False).to_string(index=False))
    print(f"\nWinning method by avg_challenge_points: {winner['baseline']} ({winner['avg_challenge_points']:.6f})")
    return summary, baselines


def main() -> None:
    results = load_results()
    rankings = load_rankings()
    run_leakage_audit(results, rankings)
    run_scoring_audit()
    build_summary_table()
    print("\nFull validation and backtesting audit passed.")


if __name__ == "__main__":
    main()
