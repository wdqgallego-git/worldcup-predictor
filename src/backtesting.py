"""Walk-forward match backtests for the 2014, 2018, and 2022 World Cups.

Historical World Cups used the old 32-team format. This module validates
per-match models and prediction strategy only; it does not validate the 2026
tournament bracket.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calibration import (
    add_historical_match_context,
    add_raw_lambda_columns,
    apply_lambda_calibration,
    build_calibration_report,
    estimate_knockout_context_multiplier,
    fit_pre_tournament_calibrator,
    paired_bootstrap_report,
)
from data_loader import load_rankings, load_results
from evaluation import (
    add_baseline_predictions,
    evaluate_baselines,
    evaluate_challenge_points,
    evaluate_goal_metrics,
    evaluate_result_metrics,
)
from features import build_fixture_features, build_training_table, prepare_match_table
from model import evaluate_goal_models, predict_expected_goals, train_goal_models
from poisson import score_probability_matrix, summarize_score_probs
from prediction_optimizer import get_safe_prediction, points_for_prediction


WORLD_CUP_BACKTESTS = {
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

FAVORITE_WIN_PROBABILITY_THRESHOLDS = np.round(np.arange(0.45, 0.751, 0.05), 2)
EXPECTED_GOAL_DIFFERENCE_THRESHOLDS = [0.2, 0.5, 0.8, 1.1, 1.5]
DRAW_PROBABILITY_THRESHOLDS = [0.25, 0.30, 0.35, 0.40]


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


def add_favorite_close_match_fallback(
    calibrated_predictions: pd.DataFrame,
    close_match_goal_difference: float = 0.20,
    max_goals: int = 6,
) -> pd.DataFrame:
    """Use favorite_1_0 for calibrated close matches and optimization otherwise."""
    scored = add_optimized_score_predictions(calibrated_predictions, max_goals=max_goals)
    favorite = add_baseline_predictions(calibrated_predictions, "favorite_1_0", max_goals=max_goals)
    close_match = (
        scored["expected_goals_a"] - scored["expected_goals_b"]
    ).abs().le(close_match_goal_difference)
    scored.loc[close_match, ["pred_a", "pred_b"]] = favorite.loc[close_match, ["pred_a", "pred_b"]]
    return scored


def summarize_baseline_metrics(predictions: pd.DataFrame, baseline: str) -> dict[str, object]:
    """Summarize one already-scored baseline or calibrated strategy."""
    return {
        "baseline": baseline,
        **evaluate_goal_metrics(
            predictions["goals_a"], predictions["goals_b"], predictions["pred_a"], predictions["pred_b"]
        ),
        **evaluate_result_metrics(
            predictions["goals_a"], predictions["goals_b"], predictions["pred_a"], predictions["pred_b"]
        ),
        **evaluate_challenge_points(
            predictions["goals_a"], predictions["goals_b"], predictions["pred_a"], predictions["pred_b"]
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
    predictions = add_raw_lambda_columns(predict_expected_goals(models, fixture_features))
    predictions["backtest_year"] = year
    predictions["model_name"] = models["selected_model_name"]
    independent = add_optimized_score_predictions(predictions, method="independent", max_goals=max_goals)
    dixon_coles = add_optimized_score_predictions(predictions, method="dixon_coles", max_goals=max_goals)
    calibrator, calibration_selection = fit_pre_tournament_calibrator(
        training_df,
        feature_cols,
        selected_model_name=models["selected_model_name"],
        reference_date=str(cutoff.date()),
        max_goals=max_goals,
    )
    calibrated_predictions = apply_lambda_calibration(predictions, calibrator)
    calibrated = add_optimized_score_predictions(calibrated_predictions, max_goals=max_goals)
    calibrated_fallback = add_favorite_close_match_fallback(calibrated_predictions, max_goals=max_goals)
    favorite = add_baseline_predictions(predictions, "favorite_1_0", max_goals=max_goals)
    independent["calibrated_expected_goals_a"] = calibrated["expected_goals_a"]
    independent["calibrated_expected_goals_b"] = calibrated["expected_goals_b"]
    independent["calibration_method"] = calibrator.method
    independent["calibrated_pred_a"] = calibrated["pred_a"]
    independent["calibrated_pred_b"] = calibrated["pred_b"]
    independent["calibrated_fallback_pred_a"] = calibrated_fallback["pred_a"]
    independent["calibrated_fallback_pred_b"] = calibrated_fallback["pred_b"]
    independent["favorite_1_0_points"] = [
        points_for_prediction(pred_a, pred_b, int(goals_a), int(goals_b))
        for pred_a, pred_b, goals_a, goals_b in zip(
            favorite["pred_a"], favorite["pred_b"], predictions["goals_a"], predictions["goals_b"]
        )
    ]
    independent["uncalibrated_optimizer_points"] = [
        points_for_prediction(pred_a, pred_b, int(goals_a), int(goals_b))
        for pred_a, pred_b, goals_a, goals_b in zip(
            independent["pred_a"], independent["pred_b"], predictions["goals_a"], predictions["goals_b"]
        )
    ]
    independent["calibrated_optimizer_points"] = [
        points_for_prediction(pred_a, pred_b, int(goals_a), int(goals_b))
        for pred_a, pred_b, goals_a, goals_b in zip(
            calibrated["pred_a"], calibrated["pred_b"], predictions["goals_a"], predictions["goals_b"]
        )
    ]
    independent["calibrated_optimizer_with_favorite_1_0_close_match_fallback_points"] = [
        points_for_prediction(pred_a, pred_b, int(goals_a), int(goals_b))
        for pred_a, pred_b, goals_a, goals_b in zip(
            calibrated_fallback["pred_a"],
            calibrated_fallback["pred_b"],
            predictions["goals_a"],
            predictions["goals_b"],
        )
    ]
    independent = add_historical_match_context(independent)
    metrics = pd.DataFrame(
        [
            summarize_prediction_metrics(independent, "expected_points_independent", year),
            summarize_prediction_metrics(dixon_coles, "expected_points_dixon_coles", year),
        ]
    )
    baseline_input = predictions.copy()
    baselines = evaluate_baselines(baseline_input, max_goals=max_goals)
    calibrated_rows = pd.DataFrame(
        [
            summarize_baseline_metrics(calibrated, "calibrated_expected_points_optimized"),
            summarize_baseline_metrics(
                calibrated_fallback,
                "calibrated_optimizer_with_favorite_1_0_close_match_fallback",
            ),
        ]
    )
    baselines = pd.concat([baselines, calibrated_rows], ignore_index=True)
    baselines.insert(0, "year", year)
    return {
        "predictions": independent,
        "dixon_coles_predictions": dixon_coles,
        "metrics": metrics,
        "baseline_comparison": baselines,
        "selected_model_name": models["selected_model_name"],
        "feature_cols": feature_cols,
        "calibration_selection": calibration_selection.assign(year=year),
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


def add_hybrid_strategy_context(predictions: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Add transparent match-type diagnostics used by the hybrid search."""
    context = predictions.drop(columns=["favorite_1_0_points", "optimized_points"], errors="ignore").copy()
    favorite_predictions = add_baseline_predictions(context, "favorite_1_0", max_goals=max_goals)
    rows = []
    for match, favorite in zip(context.itertuples(index=False), favorite_predictions.itertuples(index=False)):
        probabilities = summarize_score_probs(
            match.expected_goals_a,
            match.expected_goals_b,
            max_goals=max_goals,
            method="independent",
        )
        rank_diff = float(match.rank_a) - float(match.rank_b)
        if rank_diff < 0:
            favorite_side = "team_a"
            favorite_win_probability = probabilities["team_a_win"]
        elif rank_diff > 0:
            favorite_side = "team_b"
            favorite_win_probability = probabilities["team_b_win"]
        else:
            favorite_side = "tied_rank"
            favorite_win_probability = max(probabilities["team_a_win"], probabilities["team_b_win"])
        rows.append(
            {
                "favorite_side": favorite_side,
                "favorite_win_probability": favorite_win_probability,
                "draw_probability": probabilities["draw"],
                "abs_rank_diff": abs(rank_diff),
                "abs_expected_goal_difference": abs(float(match.expected_goals_a) - float(match.expected_goals_b)),
                "total_expected_goals": float(match.expected_goals_a) + float(match.expected_goals_b),
                "favorite_pred_a": int(favorite.pred_a),
                "favorite_pred_b": int(favorite.pred_b),
                "favorite_1_0_points": points_for_prediction(
                    int(favorite.pred_a), int(favorite.pred_b), int(match.goals_a), int(match.goals_b)
                ),
                "optimized_points": points_for_prediction(
                    int(match.pred_a), int(match.pred_b), int(match.goals_a), int(match.goals_b)
                ),
            }
        )
    context = pd.concat([context.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    context["optimizer_minus_favorite_points"] = context["optimized_points"] - context["favorite_1_0_points"]
    context["favorite_strength_rank_diff_bucket"] = pd.cut(
        context["abs_rank_diff"],
        bins=[-np.inf, 10, 25, 50, np.inf],
        labels=["close_0_10", "medium_10_25", "strong_25_50", "very_strong_50_plus"],
    )
    context["expected_goal_difference_bucket"] = pd.cut(
        context["abs_expected_goal_difference"],
        bins=[-np.inf, 0.2, 0.5, 1.0, np.inf],
        labels=["close_0_0.2", "small_0.2_0.5", "medium_0.5_1.0", "large_1.0_plus"],
    )
    context["total_expected_goals_bucket"] = pd.cut(
        context["total_expected_goals"],
        bins=[-np.inf, 2.0, 2.5, 3.0, np.inf],
        labels=["low_up_to_2.0", "medium_2.0_2.5", "high_2.5_3.0", "very_high_3.0_plus"],
    )
    context["draw_probability_bucket"] = pd.cut(
        context["draw_probability"],
        bins=[-np.inf, 0.25, 0.30, 0.35, np.inf],
        labels=["low_up_to_0.25", "medium_0.25_0.30", "high_0.30_0.35", "very_high_0.35_plus"],
    )
    return context


def build_hybrid_strategy_diagnostics(predictions: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Summarize where optimized picks win and lose against favorite_1_0."""
    context = add_hybrid_strategy_context(predictions, max_goals=max_goals)
    dimensions = {
        "favorite_strength_rank_diff": "favorite_strength_rank_diff_bucket",
        "expected_goal_difference": "expected_goal_difference_bucket",
        "total_expected_goals": "total_expected_goals_bucket",
        "draw_probability": "draw_probability_bucket",
        "world_cup_year": "backtest_year",
    }
    summaries = []
    for dimension, column in dimensions.items():
        for bucket, rows in context.groupby(column, observed=True, dropna=False):
            deltas = rows["optimizer_minus_favorite_points"]
            summaries.append(
                {
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "matches": len(rows),
                    "favorite_1_0_total_points": int(rows["favorite_1_0_points"].sum()),
                    "favorite_1_0_avg_points": float(rows["favorite_1_0_points"].mean()),
                    "optimized_total_points": int(rows["optimized_points"].sum()),
                    "optimized_avg_points": float(rows["optimized_points"].mean()),
                    "optimizer_minus_favorite_points": int(deltas.sum()),
                    "matches_optimizer_wins": int(deltas.gt(0).sum()),
                    "matches_favorite_1_0_wins": int(deltas.lt(0).sum()),
                    "matches_tied": int(deltas.eq(0).sum()),
                }
            )
    return pd.DataFrame(summaries)


def hybrid_score_predictions(
    context: pd.DataFrame,
    favorite_win_probability_threshold: float,
    expected_goal_difference_threshold: float,
    draw_probability_threshold: float,
) -> pd.DataFrame:
    """Apply one small, readable hybrid score-prediction rule."""
    predictions = context.copy()
    scores = []
    for match in predictions.itertuples(index=False):
        close_match = match.abs_expected_goal_difference < expected_goal_difference_threshold
        if match.draw_probability >= draw_probability_threshold and close_match:
            pred_a, pred_b = 1, 1
        elif match.favorite_win_probability >= favorite_win_probability_threshold and match.favorite_side != "tied_rank":
            favorite_goals = 2 if match.abs_expected_goal_difference >= expected_goal_difference_threshold else 1
            pred_a, pred_b = (favorite_goals, 0) if match.favorite_side == "team_a" else (0, favorite_goals)
        else:
            pred_a, pred_b = int(match.pred_a), int(match.pred_b)
        scores.append((pred_a, pred_b))
    predictions[["hybrid_pred_a", "hybrid_pred_b"]] = pd.DataFrame(scores, index=predictions.index)
    predictions["hybrid_points"] = [
        points_for_prediction(pred_a, pred_b, int(goals_a), int(goals_b))
        for pred_a, pred_b, goals_a, goals_b in zip(
            predictions["hybrid_pred_a"],
            predictions["hybrid_pred_b"],
            predictions["goals_a"],
            predictions["goals_b"],
        )
    ]
    return predictions


def search_hybrid_strategies(predictions: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Run a deliberately small threshold search on historical backtests only."""
    context = add_hybrid_strategy_context(predictions, max_goals=max_goals)
    favorite_year_points = context.groupby("backtest_year")["favorite_1_0_points"].sum().to_dict()
    rows = []
    for favorite_threshold in FAVORITE_WIN_PROBABILITY_THRESHOLDS:
        for expected_goal_difference_threshold in EXPECTED_GOAL_DIFFERENCE_THRESHOLDS:
            for draw_threshold in DRAW_PROBABILITY_THRESHOLDS:
                scored = hybrid_score_predictions(
                    context,
                    favorite_win_probability_threshold=float(favorite_threshold),
                    expected_goal_difference_threshold=float(expected_goal_difference_threshold),
                    draw_probability_threshold=float(draw_threshold),
                )
                year_points = scored.groupby("backtest_year")["hybrid_points"].sum().to_dict()
                rows.append(
                    {
                        "strategy": "hybrid_favorite_safe_strong_favorite_draw",
                        "favorite_win_probability_threshold": float(favorite_threshold),
                        "expected_goal_difference_threshold": float(expected_goal_difference_threshold),
                        "draw_probability_threshold": float(draw_threshold),
                        "total_challenge_points": int(scored["hybrid_points"].sum()),
                        "matches": len(scored),
                        "avg_challenge_points": float(scored["hybrid_points"].mean()),
                        "points_2014": int(year_points[2014]),
                        "points_2018": int(year_points[2018]),
                        "points_2022": int(year_points[2022]),
                        "world_cups_beating_favorite_1_0": int(
                            sum(year_points[year] > favorite_year_points[year] for year in favorite_year_points)
                        ),
                    }
                )
    comparison = pd.DataFrame(rows).sort_values(
        ["avg_challenge_points", "world_cups_beating_favorite_1_0"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    favorite_average = float(context["favorite_1_0_points"].mean())
    comparison["beats_favorite_1_0_overall"] = comparison["avg_challenge_points"] > favorite_average
    comparison["recommended_strategy"] = (
        comparison["beats_favorite_1_0_overall"]
        & comparison["world_cups_beating_favorite_1_0"].ge(2)
    )
    return comparison


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
    calibration_selection_frames = []
    for year, cutoff_date in WORLD_CUP_BACKTESTS.items():
        backtest = run_match_backtest(year, cutoff_date, results=results, rankings=rankings, max_goals=max_goals)
        prediction_frames.append(backtest["predictions"])
        metric_frames.append(backtest["metrics"])
        baseline_frames.append(backtest["baseline_comparison"])
        calibration_selection_frames.append(backtest["calibration_selection"])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    calibration_selection = pd.concat(calibration_selection_frames, ignore_index=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / "backtest_predictions.csv", index=False)
    metrics.to_csv(output_path / "backtest_metrics.csv", index=False)
    baselines.to_csv(output_path / "backtest_baseline_comparison.csv", index=False)
    diagnostics = build_hybrid_strategy_diagnostics(predictions, max_goals=max_goals)
    hybrid_comparison = search_hybrid_strategies(predictions, max_goals=max_goals)
    diagnostics.to_csv(output_path / "hybrid_strategy_diagnostics.csv", index=False)
    hybrid_comparison.to_csv(output_path / "hybrid_strategy_comparison.csv", index=False)
    calibration_report = build_calibration_report(predictions)
    calibration_report.to_csv(output_path / "calibration_report.csv", index=False)
    strategy_points = predictions[
        [
            "favorite_1_0_points",
            "uncalibrated_optimizer_points",
            "calibrated_optimizer_points",
            "calibrated_optimizer_with_favorite_1_0_close_match_fallback_points",
        ]
    ]
    significance_report = paired_bootstrap_report(strategy_points)
    significance_report.to_csv(output_path / "strategy_significance_report.csv", index=False)
    knockout_multiplier = estimate_knockout_context_multiplier(predictions)
    selected_methods = calibration_selection[calibration_selection["selected"]].groupby("method").size().to_dict()
    summary_lines = [
        "Expected-Goals Calibration Summary",
        "==================================",
        f"selected_methods_by_backtest_year: {selected_methods}",
        f"constrained_knockout_lambda_multiplier: {knockout_multiplier:.6f}",
        "default_calibration: bounded linear lambda_adj = a + b * lambda_raw",
        "isotonic_policy: select only after an out-of-sample challenge-points win with stable high-lambda bias",
        "",
        "Group context is evaluated first. Knockout adjustment is constrained to [0.95, 1.05].",
    ]
    (output_path / "calibration_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "backtest_predictions": predictions,
        "backtest_metrics": metrics,
        "backtest_baseline_comparison": baselines,
        "hybrid_strategy_diagnostics": diagnostics,
        "hybrid_strategy_comparison": hybrid_comparison,
        "calibration_report": calibration_report,
        "strategy_significance_report": significance_report,
    }
