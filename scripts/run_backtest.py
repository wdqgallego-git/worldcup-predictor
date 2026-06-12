"""Run the full match-level validation and backtesting audit."""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtesting import compare_goal_models, run_walk_forward_backtests
from data_loader import load_rankings, load_results
from download_data import download_all
from evaluation import challenge_points
from penka_backtesting import run_penka_backtesting_reports
from penka_scoring import GROUP, QUARTER_FINAL, ROUND_OF_32, validate_penka_scoring_examples
from features import build_training_table
from leakage_checks import (
    check_no_future_matches_used,
    check_no_missing_feature_values,
    check_no_placeholder_features,
    check_no_target_columns_in_features,
    check_ranking_dates_before_match,
)
from margin_analysis import run_margin_distribution_analysis
from elo_analysis import run_elo_analysis
from ml_policy_analysis import run_ml_policy_analysis
from rule_b_analysis import run_rule_b_verification
from score_pmf_backend_analysis import run_score_pmf_backend_analysis
from score_pmf_backends import validate_score_pmf_backends


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
    "expected_points_optimized_with_realistic_aggressive_filter",
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
    """Verify mutually exclusive, phase-aware Penka scoring semantics."""
    validate_penka_scoring_examples()
    validate_score_pmf_backends()
    assert_true(challenge_points(2, 1, 2, 1, GROUP) == 5, "Group exact score must be 5 points.")
    assert_true(challenge_points(3, 1, 2, 0, GROUP) == 3, "Group goal difference must be 3 points.")
    assert_true(challenge_points(1, 0, 2, 0, GROUP) == 2, "Group correct winner must be 2 points.")
    assert_true(challenge_points(2, 0, 2, 0, ROUND_OF_32) == 8, "Round-of-32 exact score must be 8 points.")
    assert_true(challenge_points(1, 0, 2, 0, ROUND_OF_32) == 3, "Round-of-32 winner must be 3 points.")
    assert_true(challenge_points(3, 1, 2, 0, QUARTER_FINAL) == 7, "Quarter-final goal difference must be 7 points.")
    print("Real Penka phase-specific scoring checks passed.")


def build_summary_table(output_dir: str | Path = "outputs") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward suite, save outputs, and print summary tables."""
    output_path = Path(output_dir)
    outputs = run_walk_forward_backtests(output_dir=output_path)
    predictions = outputs["backtest_predictions"]
    metrics = outputs["backtest_metrics"].copy()
    baselines = outputs["backtest_baseline_comparison"].copy()
    hybrid_comparison = outputs["hybrid_strategy_comparison"].copy()
    calibration_report = outputs["calibration_report"].copy()
    significance_report = outputs["strategy_significance_report"].copy()
    assert_true(predictions.groupby("backtest_year").size().eq(64).all(), "Each World Cup must have exactly 64 test matches.")
    assert_true(set(metrics["probability_method"]) == {"independent", "dixon_coles"}, "Both Poisson methods are required.")
    assert_true(REQUIRED_BASELINES.issubset(set(baselines["baseline"])), "Requested baselines are missing.")
    for file_name in (
        "backtest_predictions.csv",
        "backtest_metrics.csv",
        "backtest_baseline_comparison.csv",
        "hybrid_strategy_diagnostics.csv",
        "hybrid_strategy_comparison.csv",
        "calibration_report.csv",
        "calibration_summary.txt",
        "strategy_significance_report.csv",
    ):
        assert_true((output_path / file_name).exists(), f"Missing generated output: {file_name}")
    summary = metrics[
        [
            "year", "model_name", "probability_method", "prediction_method", "matches",
            "total_challenge_points", "average_challenge_points", "exact_score_rate",
            "correct_result_rate", "correct_goal_difference_rate",
        ]
    ].rename(columns={"average_challenge_points": "avg_challenge_points"})
    print(f"Backtest outputs: {output_path}")
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
    overall.to_csv(output_path / "backtest_baseline_overall.csv", index=False)
    best_hybrid = hybrid_comparison.iloc[0]
    print("\nBEST HYBRID STRATEGY")
    print(best_hybrid.to_frame().T.to_string(index=False))
    if bool(best_hybrid["recommended_strategy"]):
        print("\nLegacy hybrid diagnostic: candidate improved in this auxiliary report.")
    else:
        print("\nLegacy hybrid diagnostic: no threshold rule beat favorite_1_0 overall and in at least 2 of 3 World Cups.")
    print("Use the authoritative real-Penka reports below for match recommendations.")
    print("\nCALIBRATION REPORT")
    print(calibration_report.to_string(index=False))
    print("\nPAIRED BOOTSTRAP STRATEGY DIFFERENCES VS FAVORITE_1_0")
    print(significance_report.to_string(index=False))
    return summary, baselines


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run World Cup match-level walk-forward backtests.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh raw data before loading it. Existing raw files are used by default.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for backtest CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        download_all()
    results = load_results()
    rankings = load_rankings()
    run_leakage_audit(results, rankings)
    run_scoring_audit()
    training_df, feature_cols = build_training_table(results, rankings)
    model_comparison = compare_goal_models(training_df=training_df, feature_cols=feature_cols)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_comparison.to_csv(output_path / "backtest_goal_model_comparison.csv", index=False)
    print("\nGOAL MODEL COMPARISON")
    print(model_comparison.to_string(index=False))
    build_summary_table(output_dir=output_path)
    print("\nREAL PENKA STRATEGY AND CALIBRATION REPORTS")
    penka_outputs = run_penka_backtesting_reports(
        pd.read_csv(output_path / "backtest_predictions.csv"),
        output_dir=output_path,
    )
    assert_true((output_path / "penka_strategy_backtest_comparison.csv").exists(), "Missing focused Penka strategy CSV.")
    assert_true((output_path / "penka_strategy_summary.txt").exists(), "Missing focused Penka strategy summary.")
    assert_true((output_path / "market_blend_gate.json").exists(), "Missing market blend gate decision.")
    assert_true((output_path / "market_blend_gate_summary.txt").exists(), "Missing market blend gate summary.")
    assert_true(
        (output_path / "market_blend_backtest_by_year.csv").exists(),
        "Missing scope-labeled market blend by-year report.",
    )
    assert_true(
        (output_path / "market_blend_lambda_audit.csv").exists(),
        "Missing matched-odds market lambda audit.",
    )
    for file_name in (
        "market_strategy_matched_comparison.csv",
        "market_strategy_by_year.csv",
        "market_blend_weight_sensitivity.csv",
        "definitive_market_strategy_summary.txt",
    ):
        assert_true((output_path / file_name).exists(), f"Missing definitive market strategy output: {file_name}")
    print("\nMARKET BLEND PROMOTION GATE")
    print((output_path / "market_blend_gate_summary.txt").read_text(encoding="utf-8"))
    print(penka_outputs["penka_scoring_strategy_comparison"].to_string(index=False))
    print("\nFOCUSED PENKA GROUP STRATEGY REPLACEMENT TEST")
    print(penka_outputs["penka_strategy_backtest_comparison"].to_string(index=False))
    print("\nFOCUSED PENKA GROUP STRATEGY DECISION")
    print((output_path / "penka_strategy_summary.txt").read_text(encoding="utf-8"))
    print("\nREAL PENKA STRATEGY DECISION")
    print((output_path / "penka_scoring_summary.txt").read_text(encoding="utf-8"))
    print("\nSCORE PMF BACKEND DIAGNOSTIC")
    pmf_outputs = run_score_pmf_backend_analysis(
        pd.read_csv(output_path / "backtest_predictions.csv"),
        results,
        output_dir=output_path,
    )
    for file_name in (
        "score_pmf_backend_comparison.csv",
        "score_pmf_backend_by_year.csv",
        "score_pmf_backend_by_bucket.csv",
        "score_pmf_backend_summary.txt",
    ):
        assert_true((output_path / file_name).exists(), f"Missing PMF backend output: {file_name}")
    print(pmf_outputs["score_pmf_backend_comparison"].to_string(index=False))
    print("\nSCORE PMF BACKEND DECISION")
    print((output_path / "score_pmf_backend_summary.txt").read_text(encoding="utf-8"))
    print("\nFULL-HISTORY WINNING-MARGIN ANALYSIS")
    margin_outputs = run_margin_distribution_analysis(results, rankings, output_dir=output_path)
    print(margin_outputs["sample_counts"].to_string(index=False))
    print("\nFIXED-SCORE RULE DECISION")
    print((output_path / "fixed_score_rule_summary.txt").read_text(encoding="utf-8"))
    print("\nCAUSAL ELO DIAGNOSTIC")
    elo_outputs = run_elo_analysis(results, output_dir=output_path)
    print(elo_outputs["elo_margin_distribution"].to_string(index=False))
    print("\nCAUSAL ELO FIXED-SCORE DECISION")
    print((output_path / "elo_margin_summary.txt").read_text(encoding="utf-8"))
    print("\nRULE B ADVISORY VERIFICATION")
    rule_b_outputs = run_rule_b_verification(results, rankings, output_dir=output_path)
    print(rule_b_outputs["two_zero_axis_convergence"].to_string(index=False))
    print("\nRULE B ADVISORY DECISION")
    print((output_path / "rule_b_decision_summary.txt").read_text(encoding="utf-8"))
    print("\nDIRECT CONTEST-SCORE ML POLICY")
    policy_outputs = run_ml_policy_analysis(results, rankings, output_dir=output_path)
    policy_overall = policy_outputs["ml_policy_strategy_comparison"]
    policy_overall = policy_overall[
        policy_overall["scope"].eq("full_history_walk_forward")
        & policy_overall["dimension"].eq("overall")
    ]
    print(policy_overall.to_string(index=False))
    print("\nDIRECT CONTEST-SCORE ML POLICY DECISION")
    print((output_path / "ml_policy_summary.txt").read_text(encoding="utf-8"))
    print("\nFull validation and backtesting audit passed.")


if __name__ == "__main__":
    main()
