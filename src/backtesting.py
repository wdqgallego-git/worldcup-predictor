"""Walk-forward match backtests for the 2014, 2018, and 2022 World Cups.

Historical World Cups used the old 32-team format. This module validates
per-match models and prediction strategy only; it does not validate the 2026
tournament bracket.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_rankings, load_results
from evaluation import (
    evaluate_baselines,
    evaluate_challenge_points,
    evaluate_goal_metrics,
    evaluate_result_metrics,
)
from features import build_fixture_features, build_training_table, prepare_match_table
from model import evaluate_goal_models, predict_expected_goals, train_goal_models
from poisson import score_probability_matrix
from prediction_optimizer import get_safe_prediction


WORLD_CUP_BACKTESTS = {
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}


def historical_world_cup_matches(results: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return the 64 match rows for one historical World Cup."""
    matches = prepare_match_table(results)
    tournament = matches["tournament"].fillna("").astype(str)
    world_cup = matches[(matches["date"].dt.year == year) & tournament.eq("FIFA World Cup")].copy()
    if world_cup.empty:
        raise ValueError(f"No FIFA World Cup matches found for {year}.")
    return world_cup.sort_values("date", kind="stable").reset_index(drop=True)


def add_optimized_score_predictions(
    predictions: pd.DataFrame,
    method: str = "independent",
    max_goals: int = 6,
) -> pd.DataFrame:
    """Add expected-points optimized scorelines for one Poisson method."""
    scored = predictions.copy()
    optimized = []
    for expected_a, expected_b in zip(scored["expected_goals_a"], scored["expected_goals_b"]):
        score_probs = score_probability_matrix(expected_a, expected_b, max_goals=max_goals, method=method)
        safe_prediction = get_safe_prediction(score_probs, max_goals=max_goals)
        optimized.append((safe_prediction["pred_a"], safe_prediction["pred_b"], safe_prediction["expected_points"]))
    scored[["pred_a", "pred_b", "prediction_expected_points"]] = pd.DataFrame(optimized, index=scored.index)
    scored["poisson_method"] = method
    return scored


def summarize_prediction_metrics(predictions: pd.DataFrame, label: str, year: int) -> dict[str, object]:
    """Summarize goal, result, and challenge-point metrics."""
    return {
        "year": year,
        "method": label,
        "model_name": str(predictions["model_name"].iloc[0]) if "model_name" in predictions.columns else "unknown",
        "probability_method": str(predictions["poisson_method"].iloc[0]) if "poisson_method" in predictions.columns else "not_applicable",
        "prediction_method": "expected_points_optimized",
        "matches": len(predictions),
        **evaluate_goal_metrics(
            predictions["goals_a"],
            predictions["goals_b"],
            predictions["expected_goals_a"],
            predictions["expected_goals_b"],
        ),
        **evaluate_result_metrics(
            predictions["goals_a"],
            predictions["goals_b"],
            predictions["pred_a"],
            predictions["pred_b"],
        ),
        **evaluate_challenge_points(
            predictions["goals_a"],
            predictions["goals_b"],
            predictions["pred_a"],
            predictions["pred_b"],
        ),
    }


def run_match_backtest(
    year: int,
    cutoff_date: str,
    results: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
    max_goals: int = 6,
) -> dict[str, object]:
    """Train before cutoff and backtest score predictions on one World Cup."""
    results = load_results() if results is None else results.copy()
    rankings = load_rankings() if rankings is None else rankings.copy()
    cutoff = pd.Timestamp(cutoff_date)
    tournament_matches = historical_world_cup_matches(results, year)
    if (tournament_matches["date"] < cutoff).any():
        raise ValueError(f"{year} cutoff {cutoff.date()} is after at least one tournament match.")

    training_df, feature_cols = build_training_table(results, rankings)
    models = train_goal_models(training_df, feature_cols, reference_date=str(cutoff.date()))
    fixture_features, _ = build_fixture_features(tournament_matches, results, rankings, feature_cols)
    predictions = predict_expected_goals(models, fixture_features)
    predictions["backtest_year"] = year
    predictions["model_name"] = models["selected_model_name"]
    independent = add_optimized_score_predictions(predictions, method="independent", max_goals=max_goals)
    dixon_coles = add_optimized_score_predictions(predictions, method="dixon_coles", max_goals=max_goals)
    metrics = pd.DataFrame(
        [
            summarize_prediction_metrics(independent, "expected_points_independent", year),
            summarize_prediction_metrics(dixon_coles, "expected_points_dixon_coles", year),
        ]
    )
    baseline_input = predictions.copy()
    baselines = evaluate_baselines(baseline_input, max_goals=max_goals)
    baselines.insert(0, "year", year)
    return {
        "predictions": independent,
        "dixon_coles_predictions": dixon_coles,
        "metrics": metrics,
        "baseline_comparison": baselines,
        "selected_model_name": models["selected_model_name"],
        "feature_cols": feature_cols,
    }


def compare_poisson_methods(
    predictions: pd.DataFrame,
    year: int | None = None,
    max_goals: int = 6,
) -> pd.DataFrame:
    """Compare independent Poisson and Dixon-Coles score optimization."""
    required = {"goals_a", "goals_b", "expected_goals_a", "expected_goals_b"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")
    comparison_year = int(year) if year is not None else int(predictions.get("backtest_year", pd.Series([0])).iloc[0])
    rows = []
    for method in ("independent", "dixon_coles"):
        scored = add_optimized_score_predictions(predictions, method=method, max_goals=max_goals)
        rows.append(summarize_prediction_metrics(scored, method, comparison_year))
    return pd.DataFrame(rows)


def compare_goal_models(
    training_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    reference_date: str = "2026-06-01",
    results: pd.DataFrame | None = None,
    rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare expected-goals candidates on their chronological validation split."""
    if training_df is None or feature_cols is None:
        results = load_results() if results is None else results.copy()
        rankings = load_rankings() if rankings is None else rankings.copy()
        training_df, feature_cols = build_training_table(results, rankings)
    return evaluate_goal_models(training_df, feature_cols, reference_date=reference_date)


def run_walk_forward_backtests(
    output_dir: str | Path = "outputs",
    max_goals: int = 6,
) -> dict[str, pd.DataFrame]:
    """Run 2014, 2018, and 2022 match-only backtests and write CSV outputs."""
    results = load_results()
    rankings = load_rankings()
    prediction_frames = []
    metric_frames = []
    baseline_frames = []
    for year, cutoff_date in WORLD_CUP_BACKTESTS.items():
        backtest = run_match_backtest(year, cutoff_date, results=results, rankings=rankings, max_goals=max_goals)
        prediction_frames.append(backtest["predictions"])
        metric_frames.append(backtest["metrics"])
        baseline_frames.append(backtest["baseline_comparison"])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / "backtest_predictions.csv", index=False)
    metrics.to_csv(output_path / "backtest_metrics.csv", index=False)
    baselines.to_csv(output_path / "backtest_baseline_comparison.csv", index=False)
    return {
        "backtest_predictions": predictions,
        "backtest_metrics": metrics,
        "backtest_baseline_comparison": baselines,
    }
