"""Run a fast end-to-end smoke test of the World Cup predictor."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from awards_model import simulate_awards
from data_loader import (
    load_company_scoring_rules,
    load_fixtures,
    load_rankings,
    load_results,
)
from features import build_fixture_features, build_training_table
from model import predict_expected_goals, train_goal_models
from player_data import (
    build_player_feature_table,
    load_goalkeeper_stats,
    load_injury_adjustments,
    load_penalty_takers,
    load_player_stats,
    load_players,
    load_squad_status,
)
from poisson import score_probability_matrix
from prediction_optimizer import summarize_match_strategy
from tournament_simulator import run_tournament_simulation, validate_2026_format


OUTPUT_PATH = PROJECT_ROOT / "outputs" / "smoke_test_predictions.csv"
SMOKE_TRAINING_ROWS = 1_500


def validate_scoring_rules() -> None:
    """Check the required award categories before running the pipeline."""
    scoring = load_company_scoring_rules()
    required_awards = {"champion", "runner_up", "top_scorer", "mvp", "golden_glove"}
    found_awards = {row["category"] for row in scoring}
    missing_awards = required_awards - found_awards
    if missing_awards:
        raise AssertionError(f"Missing scoring rules: {sorted(missing_awards)}")


def main() -> None:
    validate_2026_format()
    validate_scoring_rules()

    results = load_results()
    rankings = load_rankings()
    fixtures = load_fixtures()
    training_df, feature_cols = build_training_table(results, rankings)
    smoke_training_df = training_df.tail(SMOKE_TRAINING_ROWS).copy()

    models = train_goal_models(smoke_training_df, feature_cols)
    fixture_features, _ = build_fixture_features(fixtures, results, rankings, feature_cols)
    fixture_predictions = predict_expected_goals(models, fixture_features)

    first_group_match = fixture_predictions.loc[fixture_predictions["stage"].eq("group_stage")].iloc[0]
    score_probs = score_probability_matrix(
        first_group_match["expected_goals_a"],
        first_group_match["expected_goals_b"],
    )
    strategy = summarize_match_strategy(score_probs)
    if strategy["safe_prediction"]["expected_points"] <= 0:
        raise AssertionError("Match optimizer returned a non-positive expected-points prediction.")

    tournament_paths = run_tournament_simulation(
        fixtures,
        fixture_predictions.loc[fixture_predictions["stage"].eq("group_stage")],
        rng=np.random.default_rng(2026),
    )
    tournament_paths.insert(0, "simulation_id", 1)
    if len(tournament_paths) != 48 or int(tournament_paths["champion"].sum()) != 1:
        raise AssertionError("Tiny 2026-style tournament simulation failed.")

    players = load_players()
    player_stats = load_player_stats(players=players)
    goalkeepers = load_goalkeeper_stats(players=players)
    penalty_takers = load_penalty_takers(players=players)
    player_features = build_player_feature_table(
        players=players,
        player_stats=player_stats,
        penalty_takers=penalty_takers,
        squad_status=load_squad_status(),
        injury_adjustments=load_injury_adjustments(),
    )
    awards = simulate_awards(
        tournament_paths,
        player_features,
        goalkeepers,
        penalty_takers=penalty_takers,
        random_seed=2026,
    )
    if len(awards) != 1:
        raise AssertionError("Awards simulation did not produce one result for the tiny tournament run.")

    output_columns = [
        "match_id",
        "stage",
        "round",
        "group",
        "date",
        "team_a",
        "team_b",
        "expected_goals_a",
        "expected_goals_b",
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture_predictions[output_columns].to_csv(OUTPUT_PATH, index=False)
    print("Smoke test passed")


if __name__ == "__main__":
    main()
