"""Run final World Cup 2026 match and tournament predictions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MAX_GOALS_FINAL, N_SIMULATIONS_FINAL
from calibration import (
    add_raw_lambda_columns,
    apply_lambda_calibration,
    estimate_knockout_context_multiplier,
    fit_pre_tournament_calibrator,
)
from data_loader import load_fixtures, load_rankings, load_results
from download_data import download_all
from evaluation import add_baseline_predictions
from features import build_fixture_features, build_training_table
from model import predict_expected_goals, train_goal_models
from poisson import summarize_score_probs
from prediction_optimizer import summarize_match_strategy
from penka_scoring import GROUP
from tournament_simulator import (
    run_monte_carlo_tournament,
    validate_2026_format,
    validate_third_place_assignment_matrix,
)


REQUIRED_GROUP_MATCHES = 72
REQUIRED_GROUP_TEAMS = 48
THIRD_PLACE_MATRIX_PATH = PROJECT_ROOT / "data" / "raw" / "third_place_assignment_matrix.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible final run."""
    parser = argparse.ArgumentParser(description="Generate World Cup 2026 match and tournament predictions.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Refresh raw data before loading it. Existing raw files are used by default.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=N_SIMULATIONS_FINAL,
        help=f"Number of tournament simulations. Defaults to {N_SIMULATIONS_FINAL:,}.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for final prediction outputs.")
    parser.add_argument(
        "--require-official-matrix",
        action="store_true",
        help="Fail unless the official Round-of-32 third-place assignment matrix has been populated.",
    )
    parser.add_argument(
        "--allow-development-fallback",
        action="store_true",
        help="Allow deterministic third-place fallback for development only. Never use for a final submission.",
    )
    return parser.parse_args()


def ensure_raw_data(download: bool) -> None:
    """Download raw data when explicitly requested or when required files are missing."""
    required_paths = [
        PROJECT_ROOT / "data" / "raw" / "results.csv",
        PROJECT_ROOT / "data" / "raw" / "rankings.csv",
        PROJECT_ROOT / "data" / "raw" / "fixtures.csv",
    ]
    if download or any(not path.exists() for path in required_paths):
        download_all()


def validate_group_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Return the resolved 48-team group schedule or raise a clear error."""
    group_fixtures = fixtures[fixtures["stage"].eq("group_stage")].copy()
    if len(group_fixtures) != REQUIRED_GROUP_MATCHES:
        raise ValueError(f"Expected {REQUIRED_GROUP_MATCHES} group matches. Found {len(group_fixtures)}.")
    if group_fixtures["has_placeholder_team"].any():
        unresolved = group_fixtures.loc[
            group_fixtures["has_placeholder_team"], ["match_id", "team_a", "team_b"]
        ]
        raise ValueError(f"Group fixtures still contain unresolved teams:\n{unresolved.to_string(index=False)}")
    teams = set(group_fixtures["team_a"]) | set(group_fixtures["team_b"])
    if len(teams) != REQUIRED_GROUP_TEAMS:
        raise ValueError(f"Expected {REQUIRED_GROUP_TEAMS} group-stage teams. Found {len(teams)}.")
    return group_fixtures


def load_third_place_assignment_matrix(
    require_official_matrix: bool,
    allow_development_fallback: bool,
) -> pd.DataFrame:
    """Load the complete official matrix or require an explicit development opt-in."""
    try:
        matrix = pd.read_csv(THIRD_PLACE_MATRIX_PATH, comment="#")
        return validate_third_place_assignment_matrix(matrix)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
        message = f"Official 2026 third-place assignment matrix is unavailable or invalid: {exc}"
        if require_official_matrix or not allow_development_fallback:
            raise ValueError(message) from exc
        print(f"WARNING: {message}")
        print("WARNING: Using deterministic third-place assignment fallback for development only.")
        return pd.DataFrame()


def load_knockout_lambda_multiplier(output_dir: Path) -> float:
    """Estimate a constrained knockout adjustment from available historical backtests."""
    backtest_path = output_dir / "backtest_predictions.csv"
    if not backtest_path.exists():
        print("WARNING: Backtest predictions are unavailable; using knockout lambda multiplier 1.0.")
        return 1.0
    backtests = pd.read_csv(backtest_path)
    required = {"calibrated_expected_goals_a", "calibrated_expected_goals_b", "goals_a", "goals_b", "backtest_year", "date"}
    if not required.issubset(backtests.columns):
        print("WARNING: Backtests predate calibration outputs; using knockout lambda multiplier 1.0.")
        return 1.0
    return estimate_knockout_context_multiplier(backtests)


def optimize_fixture_predictions(fixture_predictions: pd.DataFrame) -> pd.DataFrame:
    """Add W/D/L probabilities and expected-points optimized score picks."""
    rows = []
    for fixture in fixture_predictions.itertuples(index=False):
        probabilities = summarize_score_probs(
            fixture.expected_goals_a,
            fixture.expected_goals_b,
            max_goals=MAX_GOALS_FINAL,
            method="independent",
        )
        strategy = summarize_match_strategy(probabilities["score_probs"], max_goals=MAX_GOALS_FINAL, phase=GROUP)
        safe = strategy["safe_prediction"]
        aggressive = strategy["aggressive_prediction"]
        most_likely_goals = np.unravel_index(
            np.argmax(probabilities["score_probs"]),
            probabilities["score_probs"].shape,
        )
        rows.append(
            {
                "match_id": fixture.match_id,
                "most_likely_score": f"{most_likely_goals[0]}-{most_likely_goals[1]}",
                "p_team_a_win": probabilities["team_a_win"],
                "p_draw": probabilities["draw"],
                "p_team_b_win": probabilities["team_b_win"],
                "optimized_score_a": safe["pred_a"],
                "optimized_score_b": safe["pred_b"],
                "safe_prediction": safe["prediction"],
                "optimized_result": safe["result"],
                "safe_expected_points": safe["expected_points"],
                "expected_penka_points": safe["expected_points"],
                "exact_score_probability": safe["exact_score_probability"],
                "correct_goal_difference_or_draw_probability": safe["correct_goal_difference_or_draw_probability"],
                "correct_winner_probability": safe["correct_winner_probability"],
                "wrong_probability": safe["wrong_probability"],
                "selected_by": safe["selected_by"],
                "tiebreak_changed_selection": safe["tiebreak_changed_selection"],
                "aggressive_prediction": aggressive["prediction"],
                "aggressive_expected_points": aggressive["expected_points"],
                "aggressive_ev_ratio": aggressive["ev_ratio"],
                "refresh_required_before_match": True,
            }
        )
    return fixture_predictions.merge(pd.DataFrame(rows), on="match_id", how="left", validate="one_to_one")


def select_final_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, submission-friendly match prediction table."""
    columns = [
        "match_id",
        "stage",
        "round",
        "group",
        "date",
        "time",
        "team_a",
        "team_b",
        "raw_expected_goals_a",
        "raw_expected_goals_b",
        "calibrated_expected_goals_a",
        "calibrated_expected_goals_b",
        "calibration_method",
        "expected_goals_a",
        "expected_goals_b",
        "most_likely_score",
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "optimized_score_a",
        "optimized_score_b",
        "safe_prediction",
        "optimized_result",
        "safe_expected_points",
        "expected_penka_points",
        "exact_score_probability",
        "correct_goal_difference_or_draw_probability",
        "correct_winner_probability",
        "wrong_probability",
        "selected_by",
        "tiebreak_changed_selection",
        "aggressive_prediction",
        "aggressive_expected_points",
        "aggressive_ev_ratio",
        "refresh_required_before_match",
    ]
    return predictions[columns].copy()


def save_match_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Save final match predictions in CSV and Excel formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "final_predictions.csv", index=False)
    with pd.ExcelWriter(output_dir / "final_predictions.xlsx", engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="match_predictions", index=False)


def build_final_recommendations(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Write refreshable group recommendations from real Penka expected value."""
    favorite = add_baseline_predictions(predictions, "favorite_1_0", max_goals=MAX_GOALS_FINAL)
    recommended_strategy = "penka_group_expected_value_refreshable"
    reason = (
        "Real group-phase Penka EV with exact-score tie-breaking. Refresh close to kickoff; "
        "the earlier favorite_1_0 conclusion used an invalid points formula."
    )
    recommendations = predictions[
        ["match_id", "team_a", "team_b", "rank_a", "rank_b", "safe_prediction", "expected_penka_points", "selected_by"]
    ].copy()
    recommendations["favorite_1_0_prediction"] = [
        f"{int(pred_a)}-{int(pred_b)}" for pred_a, pred_b in zip(favorite["pred_a"], favorite["pred_b"])
    ]
    recommendations["recommended_strategy"] = recommended_strategy
    recommendations["recommended_prediction"] = recommendations["safe_prediction"]
    recommendations["recommendation_reason"] = reason
    recommendations["refresh_required_before_match"] = True
    recommendations.to_csv(output_dir / "final_match_recommendations.csv", index=False)
    with pd.ExcelWriter(output_dir / "final_match_recommendations.xlsx", engine="openpyxl") as writer:
        recommendations.to_excel(writer, sheet_name="recommendations", index=False)
    return recommendations


def main() -> None:
    args = parse_args()
    if args.simulations < 1:
        raise ValueError("--simulations must be positive.")

    os.chdir(PROJECT_ROOT)
    validate_2026_format()
    ensure_raw_data(args.download)
    results = load_results()
    rankings = load_rankings()
    fixtures = load_fixtures()
    group_fixtures = validate_group_fixtures(fixtures)
    third_place_matrix = load_third_place_assignment_matrix(
        args.require_official_matrix,
        args.allow_development_fallback,
    )

    print("Building leakage-safe training features...")
    training_df, feature_cols = build_training_table(results, rankings)
    print("Selecting and training the best goal model...")
    models = train_goal_models(training_df, feature_cols)
    print(f"Selected goal model: {models['selected_model_name']}")
    calibrator, calibration_selection = fit_pre_tournament_calibrator(
        training_df,
        feature_cols,
        selected_model_name=models["selected_model_name"],
        reference_date="2026-06-01",
        max_goals=MAX_GOALS_FINAL,
    )
    print(f"Selected lambda calibration: {calibrator.method}")
    print(calibration_selection.to_string(index=False))

    fixture_features, _ = build_fixture_features(group_fixtures, results, rankings, feature_cols)
    output_dir = Path(args.output_dir)
    fixture_predictions = add_raw_lambda_columns(predict_expected_goals(models, fixture_features))
    fixture_predictions = apply_lambda_calibration(fixture_predictions, calibrator)
    optimized_predictions = optimize_fixture_predictions(fixture_predictions)
    final_predictions = select_final_prediction_columns(optimized_predictions)
    save_match_predictions(final_predictions, output_dir)
    recommendations = build_final_recommendations(optimized_predictions, output_dir)
    print(f"Saved {len(final_predictions)} optimized group-stage match predictions.")
    print(f"Recommended match strategy: {recommendations['recommended_strategy'].iloc[0]}")
    print(f"Running {args.simulations:,} World Cup 2026 tournament simulations...")
    knockout_lambda_multiplier = load_knockout_lambda_multiplier(output_dir)
    print(f"Constrained knockout lambda multiplier: {knockout_lambda_multiplier:.6f}")
    run_monte_carlo_tournament(
        fixtures=fixtures,
        match_predictions=fixture_predictions,
        n_simulations=args.simulations,
        third_place_assignment_matrix=third_place_matrix,
        output_dir=output_dir,
        allow_development_fallback=args.allow_development_fallback,
        knockout_lambda_multiplier=knockout_lambda_multiplier,
    )
    print(f"Final prediction outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
