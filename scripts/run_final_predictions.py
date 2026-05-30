"""Run final World Cup 2026 match and tournament predictions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MAX_GOALS_FINAL, N_SIMULATIONS_FINAL
from data_loader import load_fixtures, load_rankings, load_results
from download_data import download_all
from features import build_fixture_features, build_training_table
from model import predict_expected_goals, train_goal_models
from poisson import summarize_score_probs
from prediction_optimizer import summarize_match_strategy
from tournament_simulator import run_monte_carlo_tournament, validate_2026_format


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


def load_third_place_assignment_matrix(require_official_matrix: bool) -> pd.DataFrame:
    """Load the official matrix when populated, otherwise allow the documented development fallback."""
    matrix = pd.read_csv(THIRD_PLACE_MATRIX_PATH, comment="#")
    if matrix.empty:
        message = (
            "Official 2026 third-place assignment matrix is not populated. "
            "Using the simulator's deterministic development fallback."
        )
        if require_official_matrix:
            raise ValueError(message)
        print(f"WARNING: {message}")
    return matrix


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
        strategy = summarize_match_strategy(probabilities["score_probs"], max_goals=MAX_GOALS_FINAL)
        safe = strategy["safe_prediction"]
        aggressive = strategy["aggressive_prediction"]
        rows.append(
            {
                "match_id": fixture.match_id,
                "team_a_win_probability": probabilities["team_a_win"],
                "draw_probability": probabilities["draw"],
                "team_b_win_probability": probabilities["team_b_win"],
                "optimized_score_a": safe["pred_a"],
                "optimized_score_b": safe["pred_b"],
                "optimized_score": safe["prediction"],
                "optimized_result": safe["result"],
                "optimized_expected_points": safe["expected_points"],
                "aggressive_score": aggressive["prediction"],
                "aggressive_expected_points": aggressive["expected_points"],
                "aggressive_ev_ratio": aggressive["ev_ratio"],
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
        "expected_goals_a",
        "expected_goals_b",
        "team_a_win_probability",
        "draw_probability",
        "team_b_win_probability",
        "optimized_score_a",
        "optimized_score_b",
        "optimized_score",
        "optimized_result",
        "optimized_expected_points",
        "aggressive_score",
        "aggressive_expected_points",
        "aggressive_ev_ratio",
    ]
    return predictions[columns].copy()


def save_match_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    """Save final match predictions in CSV and Excel formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "final_predictions.csv", index=False)
    with pd.ExcelWriter(output_dir / "final_predictions.xlsx", engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="match_predictions", index=False)


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
    third_place_matrix = load_third_place_assignment_matrix(args.require_official_matrix)

    print("Building leakage-safe training features...")
    training_df, feature_cols = build_training_table(results, rankings)
    print("Selecting and training the best goal model...")
    models = train_goal_models(training_df, feature_cols)
    print(f"Selected goal model: {models['selected_model_name']}")

    fixture_features, _ = build_fixture_features(group_fixtures, results, rankings, feature_cols)
    fixture_predictions = predict_expected_goals(models, fixture_features)
    final_predictions = select_final_prediction_columns(optimize_fixture_predictions(fixture_predictions))

    output_dir = Path(args.output_dir)
    save_match_predictions(final_predictions, output_dir)
    print(f"Saved {len(final_predictions)} optimized group-stage match predictions.")
    print(f"Running {args.simulations:,} World Cup 2026 tournament simulations...")
    run_monte_carlo_tournament(
        fixtures=fixtures,
        match_predictions=fixture_predictions,
        n_simulations=args.simulations,
        third_place_assignment_matrix=third_place_matrix,
        output_dir=output_dir,
    )
    print(f"Final prediction outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
