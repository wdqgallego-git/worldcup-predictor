"""Run a fast end-to-end smoke test of the World Cup predictor."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from awards_model import score_golden_glove_candidates, simulate_awards
from contest_config import AWARD_POINTS
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
from penka_scoring import GROUP, QUARTER_FINAL, ROUND_OF_32, score_prediction, validate_penka_scoring_examples
from tournament_simulator import (
    assign_third_placed_teams_to_bracket,
    run_tournament_simulation,
    validate_2026_format,
    validate_third_place_assignment_matrix,
)


OUTPUT_PATH = PROJECT_ROOT / "outputs" / "smoke_test_predictions.csv"
THIRD_PLACE_MATRIX_PATH = PROJECT_ROOT / "data" / "raw" / "third_place_assignment_matrix.csv"
SMOKE_TRAINING_ROWS = 1_500


def validate_scoring_rules() -> None:
    """Check the required award categories before running the pipeline."""
    scoring = load_company_scoring_rules()
    required_awards = {"champion", "runner_up", "top_scorer", "mvp", "golden_glove"}
    found_awards = {row["category"] for row in scoring}
    missing_awards = required_awards - found_awards
    if missing_awards:
        raise AssertionError(f"Missing scoring rules: {sorted(missing_awards)}")
    loaded = {row["category"]: int(row["points"]) for row in scoring}
    if loaded != AWARD_POINTS:
        raise AssertionError(f"Award points do not match confirmed Penka values: {loaded}")


def validate_penka_match_scoring() -> None:
    """Prove phase scoring, draw handling, and optimizer recommendation caps."""
    validate_penka_scoring_examples()
    if score_prediction(1, 1, 0, 0, GROUP) != 3:
        raise AssertionError("Non-exact group draw must score goal-difference/draw points.")
    if score_prediction(1, 0, 2, 0, ROUND_OF_32) != 3:
        raise AssertionError("Round-of-32 correct-winner points are incorrect.")
    if score_prediction(1, 0, 2, 0, QUARTER_FINAL) != 5:
        raise AssertionError("Quarter-final correct-winner points are incorrect.")
    knockout = summarize_match_strategy(score_probability_matrix(0.30, 0.30, max_goals=5), phase=ROUND_OF_32)
    if knockout["safe_prediction"]["result"] != "draw":
        raise AssertionError("Knockout optimizer must allow a 90-minute draw prediction.")
    for candidate in knockout["top_candidates"]:
        if max(candidate["pred_a"], candidate["pred_b"]) > 4 or candidate["pred_a"] + candidate["pred_b"] > 5:
            raise AssertionError("Optimizer emitted an absurd ordinary recommendation.")


def validate_non_champion_golden_glove_path() -> None:
    """Prove the Golden Glove heuristic can select a non-champion keeper."""
    team_paths = pd.DataFrame(
        [
            {"simulation_id": 1, "team": "Champion", "matches_played": 8, "clean_sheets": 0, "finish_rank": 1},
            {"simulation_id": 1, "team": "Runner-up", "matches_played": 8, "clean_sheets": 8, "finish_rank": 2},
        ]
    )
    keepers = pd.DataFrame(
        [
            {"goalkeeper": "Champion Keeper", "team": "Champion", "minutes": 720, "saves": 0, "goals_conceded": 8, "clean_sheets": 0, "save_percentage": 0.0, "goals_prevented": -5.0, "reputation_prior": 0.0},
            {"goalkeeper": "Runner-up Keeper", "team": "Runner-up", "minutes": 720, "saves": 20, "goals_conceded": 0, "clean_sheets": 8, "save_percentage": 1.0, "goals_prevented": 5.0, "reputation_prior": 1.0},
        ]
    )
    winner = score_golden_glove_candidates(team_paths, keepers).iloc[0]
    if winner["team"] != "Runner-up":
        raise AssertionError("Golden Glove heuristic incorrectly auto-selected the champion keeper.")


def validate_official_matrix_contract() -> pd.DataFrame:
    """Prove final mode rejects fallback while development mode remains explicit."""
    matrix = validate_third_place_assignment_matrix(pd.read_csv(THIRD_PLACE_MATRIX_PATH, comment="#"))
    qualified = pd.DataFrame(
        {"group": list("ABCDEFGH"), "team": [f"Third {group}" for group in "ABCDEFGH"]}
    )
    try:
        assign_third_placed_teams_to_bracket(qualified, pd.DataFrame())
    except ValueError:
        pass
    else:
        raise AssertionError("Final mode silently used the development third-place fallback.")
    fallback = assign_third_placed_teams_to_bracket(
        qualified,
        pd.DataFrame(),
        allow_development_fallback=True,
    )
    if len(fallback) != 8:
        raise AssertionError("Explicit development fallback assignment failed.")
    return matrix


def main() -> None:
    validate_2026_format()
    validate_scoring_rules()
    validate_penka_match_scoring()
    validate_non_champion_golden_glove_path()
    third_place_matrix = validate_official_matrix_contract()

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
    strategy = summarize_match_strategy(score_probs, phase=GROUP)
    if strategy["safe_prediction"]["expected_points"] <= 0:
        raise AssertionError("Match optimizer returned a non-positive expected-points prediction.")

    tournament_paths = run_tournament_simulation(
        fixtures,
        fixture_predictions.loc[fixture_predictions["stage"].eq("group_stage")],
        third_place_assignment_matrix=third_place_matrix,
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
