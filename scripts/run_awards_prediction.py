"""Run path-driven World Cup award predictions and write readiness outputs."""

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

from award_strategy import (
    build_award_picks,
    calculate_award_expected_points,
    load_scoring_rules,
    save_award_strategy_outputs,
)
from awards_model import aggregate_award_probabilities, simulate_awards
from player_data import (
    build_player_feature_table,
    load_goalkeeper_stats,
    load_injury_adjustments,
    load_penalty_takers,
    load_player_stats,
    load_players,
    load_squad_status,
)


SAMPLE_PLAYER_WARNING = "WARNING: Award outputs are based on sample player data and are not valid for final submission."
PLAYER_FILE_NAMES = ("players", "player_stats", "goalkeeper_stats", "penalty_takers")
THIRD_PLACE_MATRIX_PATH = Path("data/raw/third_place_assignment_matrix.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Generate World Cup award probabilities and company-game picks.")
    parser.add_argument("--output-dir", default="outputs", help="Directory containing tournament paths and award outputs.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow a tiny synthetic path table only when team_path_simulations.csv is missing.",
    )
    return parser.parse_args()


def create_synthetic_team_paths() -> pd.DataFrame:
    """Return a tiny development-only path table for explicit synthetic runs."""
    print("WARNING: Using explicit synthetic tournament paths for development only.")
    return pd.DataFrame(
        [
            {
                "simulation_id": 1,
                "team": team,
                "matches_played": 3,
                "goals_for": 0,
                "goals_against": 0,
                "clean_sheets": 0,
                "finish_rank": 0,
                "reached_knockouts": False,
                "reached_round_of_16": False,
                "reached_quarterfinal": False,
                "reached_semifinal": False,
                "reached_final": False,
                "champion": False,
                "runner_up": False,
            }
            for team in ("Argentina", "Brazil", "France")
        ]
    )


def load_team_paths(output_dir: Path, allow_synthetic: bool) -> tuple[pd.DataFrame, bool]:
    """Load real tournament paths or fail unless synthetic paths were explicitly allowed."""
    path = output_dir / "team_path_simulations.csv"
    if path.exists():
        paths = pd.read_csv(path)
        required = {"simulation_id", "team", "matches_played", "clean_sheets", "finish_rank"}
        missing = sorted(required - set(paths.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        print(f"Loaded real tournament paths: {len(paths):,} rows from {path}")
        return paths, False
    if not allow_synthetic:
        raise FileNotFoundError(
            f"Required tournament paths not found: {path}. "
            "Run scripts/run_final_predictions.py first. "
            "Use --allow-synthetic only for an explicit development-only fallback."
        )
    return create_synthetic_team_paths(), True


def using_sample_player_data() -> bool:
    """Return whether any core player input falls back to a sample CSV."""
    return any(
        not Path(f"data/raw/{file_name}.csv").exists()
        and Path(f"data/raw/{file_name}_sample.csv").exists()
        for file_name in PLAYER_FILE_NAMES
    )


def load_player_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and build canonical player, goalkeeper, and penalty-taker inputs."""
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
    return player_features, goalkeepers, penalty_takers


def probability_files(output_dir: Path) -> dict[str, tuple[Path, str]]:
    """Return award-strategy probability paths rooted in output_dir."""
    return {
        "champion": (output_dir / "champion_probabilities.csv", "champion_probability"),
        "runner_up": (output_dir / "runner_up_probabilities.csv", "runner_up_probability"),
        "top_scorer": (output_dir / "top_scorer_probabilities.csv", "probability"),
        "mvp": (output_dir / "mvp_probabilities.csv", "probability"),
        "golden_glove": (output_dir / "golden_glove_probabilities.csv", "probability"),
    }


def official_third_place_matrix_missing() -> bool:
    """Return whether the official third-place assignment matrix is still empty."""
    return not THIRD_PLACE_MATRIX_PATH.exists() or pd.read_csv(THIRD_PLACE_MATRIX_PATH, comment="#").empty


def write_model_readiness_report(output_dir: Path, sample_data: bool, synthetic_paths: bool) -> Path:
    """Write a blunt submission-readiness report from generated artifacts."""
    baseline_path = output_dir / "backtest_baseline_overall.csv"
    final_predictions_path = output_dir / "final_predictions.csv"
    optimized_beats_favorite = False
    if baseline_path.exists():
        baselines = pd.read_csv(baseline_path).set_index("baseline")
        optimized_beats_favorite = (
            "expected_points_optimized" in baselines.index
            and "favorite_1_0" in baselines.index
            and float(baselines.loc["expected_points_optimized", "avg_challenge_points"])
            > float(baselines.loc["favorite_1_0", "avg_challenge_points"])
        )

    aggressive_usable = False
    if final_predictions_path.exists():
        predictions = pd.read_csv(final_predictions_path)
        required = {"safe_prediction", "aggressive_ev_ratio", "aggressive_prediction"}
        if required.issubset(predictions.columns):
            safe_scores = predictions["safe_prediction"].str.split("-", expand=True).astype(int)
            aggressive_scores = predictions["aggressive_prediction"].str.split("-", expand=True).astype(int)
            aggressive_usable = (
                predictions["aggressive_ev_ratio"].ge(0.95).all()
                and aggressive_scores.max(axis=1).le(4).all()
                and aggressive_scores.sum(axis=1).le(safe_scores.sum(axis=1) + 2).all()
                and (aggressive_scores[0] - aggressive_scores[1]).abs().le(
                    (safe_scores[0] - safe_scores[1]).abs() + 1
                ).all()
            )

    matrix_missing = official_third_place_matrix_missing()
    ready = not sample_data and not synthetic_paths and not matrix_missing
    lines = [
        "World Cup Predictor Model Readiness Report",
        "==========================================",
        f"legacy_optimizer_beats_favorite_1_0_under_old_report: {optimized_beats_favorite}",
        f"aggressive_picks_usable: {aggressive_usable}",
        f"awards_used_sample_player_data: {sample_data}",
        f"awards_used_synthetic_team_paths: {synthetic_paths}",
        f"official_third_place_assignment_matrix_missing: {matrix_missing}",
        f"final_predictions_ready_for_submission: {ready}",
        "",
        "Notes:",
        "- Match advice must use the real phase-specific Penka reports; old baseline conclusions are invalid.",
        "- Award outputs based on sample player data are development-only.",
        "- Champion and runner-up probabilities use simulator outcomes; runner-up is not finalist probability.",
    ]
    report_path = output_dir / "model_readiness_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_award_readiness_summary(output_dir: Path, sample_data: bool, synthetic_paths: bool) -> Path:
    """Write a submission marker for award artifacts."""
    ready = not sample_data and not synthetic_paths
    lines = [
        "Award Readiness Summary",
        "=======================",
        f"status: {'SUBMISSION READY' if ready else 'NOT SUBMISSION READY - SAMPLE PLAYER DATA OR SYNTHETIC PATHS'}",
        f"sample_player_data_used: {sample_data}",
        f"synthetic_team_paths_used: {synthetic_paths}",
        "champion_source: champion_probability from simulated champion outcomes",
        "runner_up_source: runner_up_probability from simulated runner_up flag, not finalist probability",
        "top_scorer_model: direct player simulation with separate penalty model and Golden Boot tie-breaks",
        "mvp_model: transparent fixed heuristic weights",
        "golden_glove_model: transparent fixed heuristic weights; not auto-assigned to champion keeper",
    ]
    path = output_dir / "award_readiness_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """Run award probabilities, expected-points strategy, and readiness reporting."""
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    team_paths, synthetic_paths = load_team_paths(output_dir, args.allow_synthetic)
    sample_data = using_sample_player_data()
    player_features, goalkeepers, penalty_takers = load_player_inputs()
    if sample_data:
        print(SAMPLE_PLAYER_WARNING)

    print(f"Running award simulations for {team_paths['simulation_id'].nunique():,} tournament paths...")
    award_results = simulate_awards(
        team_paths,
        player_features,
        goalkeepers,
        penalty_takers=penalty_takers,
        random_seed=2026,
    )
    award_probabilities = aggregate_award_probabilities(award_results, output_dir=output_dir)
    for category, probabilities in award_probabilities.items():
        print(f"Saved {category}: {len(probabilities):,} candidates")

    scoring_rules = load_scoring_rules()
    expected_points = calculate_award_expected_points(
        scoring_rules=scoring_rules,
        probability_files=probability_files(output_dir),
    )
    final_picks = build_award_picks(expected_points)
    strategy_outputs = save_award_strategy_outputs(expected_points, final_picks=final_picks, output_dir=output_dir)
    report_path = write_model_readiness_report(output_dir, sample_data=sample_data, synthetic_paths=synthetic_paths)
    award_readiness_path = write_award_readiness_summary(output_dir, sample_data=sample_data, synthetic_paths=synthetic_paths)
    print(final_picks.to_string(index=False))
    print(f"Saved award strategy outputs: {', '.join(str(path) for path in strategy_outputs.values())}")
    print(f"Saved model readiness report: {report_path}")
    print(f"Saved award readiness summary: {award_readiness_path}")


if __name__ == "__main__":
    main()
