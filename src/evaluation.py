"""Evaluation metrics and transparent score-prediction baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from poisson import score_probability_matrix
from prediction_optimizer import get_result, get_safe_prediction, points_for_prediction


ACTUAL_SCORE_COLUMNS = ["goals_a", "goals_b"]
PREDICTED_SCORE_COLUMNS = ["pred_a", "pred_b"]
EXPECTED_GOALS_COLUMNS = ["expected_goals_a", "expected_goals_b"]
RANKING_COLUMNS = ["rank_a", "rank_b"]
STRONG_FAVORITE_RANK_GAP = 20


def validate_columns(df: pd.DataFrame, required_columns, dataset_name: str) -> None:
    """Raise a clear error for missing evaluation inputs."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def challenge_points(pred_a: int, pred_b: int, actual_a: int, actual_b: int) -> int:
    """Return match challenge points using the shared scoring rule."""
    return points_for_prediction(pred_a, pred_b, actual_a, actual_b)


def poisson_negative_log_likelihood(actual_goals, expected_goals) -> float:
    """Return mean Poisson negative log likelihood."""
    actual = np.asarray(actual_goals, dtype=float)
    expected = np.clip(np.asarray(expected_goals, dtype=float), 0.05, 6.0)
    log_factorial = np.asarray([math.lgamma(goal + 1.0) for goal in actual])
    return float(np.mean(expected - actual * np.log(expected) + log_factorial))


def evaluate_goal_metrics(
    actual_a,
    actual_b,
    predicted_a,
    predicted_b,
) -> dict[str, float]:
    """Evaluate Team A and Team B goal predictions."""
    actual_a = np.asarray(actual_a, dtype=float)
    actual_b = np.asarray(actual_b, dtype=float)
    predicted_a = np.asarray(predicted_a, dtype=float)
    predicted_b = np.asarray(predicted_b, dtype=float)
    return {
        "mae_team_a": float(mean_absolute_error(actual_a, predicted_a)),
        "mae_team_b": float(mean_absolute_error(actual_b, predicted_b)),
        "rmse_team_a": float(mean_squared_error(actual_a, predicted_a) ** 0.5),
        "rmse_team_b": float(mean_squared_error(actual_b, predicted_b) ** 0.5),
        "poisson_nll": poisson_negative_log_likelihood(
            np.concatenate([actual_a, actual_b]),
            np.concatenate([predicted_a, predicted_b]),
        ),
    }


def evaluate_result_metrics(
    actual_a,
    actual_b,
    predicted_a,
    predicted_b,
) -> dict[str, float]:
    """Evaluate predicted win/draw/loss results."""
    actual_results = [get_result(a, b) for a, b in zip(actual_a, actual_b)]
    predicted_results = [get_result(a, b) for a, b in zip(predicted_a, predicted_b)]
    return {
        "result_accuracy": float(accuracy_score(actual_results, predicted_results)),
        "team_a_win_accuracy": float(np.mean([pred == actual for pred, actual in zip(predicted_results, actual_results) if actual == "team_a_win"]))
        if "team_a_win" in actual_results else np.nan,
        "draw_accuracy": float(np.mean([pred == actual for pred, actual in zip(predicted_results, actual_results) if actual == "draw"]))
        if "draw" in actual_results else np.nan,
        "team_b_win_accuracy": float(np.mean([pred == actual for pred, actual in zip(predicted_results, actual_results) if actual == "team_b_win"]))
        if "team_b_win" in actual_results else np.nan,
    }


def evaluate_challenge_points(
    actual_a,
    actual_b,
    predicted_a,
    predicted_b,
) -> dict[str, float]:
    """Evaluate total and average challenge points."""
    points = [
        challenge_points(pred_a, pred_b, actual_goals_a, actual_goals_b)
        for actual_goals_a, actual_goals_b, pred_a, pred_b in zip(actual_a, actual_b, predicted_a, predicted_b)
    ]
    return {
        "total_challenge_points": int(sum(points)),
        "average_challenge_points": float(np.mean(points)),
        "exact_score_rate": float(np.mean([pred_a == actual_goals_a and pred_b == actual_goals_b for actual_goals_a, actual_goals_b, pred_a, pred_b in zip(actual_a, actual_b, predicted_a, predicted_b)])),
        "correct_result_rate": float(np.mean([get_result(pred_a, pred_b) == get_result(actual_goals_a, actual_goals_b) for actual_goals_a, actual_goals_b, pred_a, pred_b in zip(actual_a, actual_b, predicted_a, predicted_b)])),
        "correct_goal_difference_rate": float(np.mean([pred_a - pred_b == actual_goals_a - actual_goals_b for actual_goals_a, actual_goals_b, pred_a, pred_b in zip(actual_a, actual_b, predicted_a, predicted_b)])),
    }


def most_likely_poisson_score(expected_goals_a: float, expected_goals_b: float, max_goals: int = 6) -> tuple[int, int]:
    """Return the scoreline with the highest independent Poisson probability."""
    probabilities = score_probability_matrix(expected_goals_a, expected_goals_b, max_goals=max_goals)
    pred_a, pred_b = np.unravel_index(np.argmax(probabilities), probabilities.shape)
    return int(pred_a), int(pred_b)


def ranking_favorite_score(rank_a: float, rank_b: float, strong_favorite_rank_gap: float = STRONG_FAVORITE_RANK_GAP) -> tuple[int, int]:
    """Return a 1-0 or 2-0 score for the ranking favorite."""
    if pd.isna(rank_a) or pd.isna(rank_b) or rank_a == rank_b:
        return 1, 1
    rank_gap = abs(float(rank_a) - float(rank_b))
    favorite_score = 2 if rank_gap >= strong_favorite_rank_gap else 1
    return (favorite_score, 0) if rank_a < rank_b else (0, favorite_score)


def add_baseline_predictions(
    matches: pd.DataFrame,
    baseline: str,
    max_goals: int = 6,
) -> pd.DataFrame:
    """Return matches with pred_a/pred_b for one transparent baseline."""
    predictions = matches.copy()
    if baseline == "always_1_1":
        predictions["pred_a"], predictions["pred_b"] = 1, 1
    elif baseline == "favorite_1_0":
        validate_columns(predictions, RANKING_COLUMNS, baseline)
        scores = [
            (1, 1) if pd.isna(rank_a) or pd.isna(rank_b) or rank_a == rank_b else (1, 0) if rank_a < rank_b else (0, 1)
            for rank_a, rank_b in zip(predictions["rank_a"], predictions["rank_b"])
        ]
        predictions[["pred_a", "pred_b"]] = pd.DataFrame(scores, index=predictions.index)
    elif baseline == "favorite_2_0_if_strong":
        validate_columns(predictions, RANKING_COLUMNS, baseline)
        scores = [ranking_favorite_score(rank_a, rank_b) for rank_a, rank_b in zip(predictions["rank_a"], predictions["rank_b"])]
        predictions[["pred_a", "pred_b"]] = pd.DataFrame(scores, index=predictions.index)
    elif baseline in {"most_likely_poisson", "expected_points_optimized"}:
        validate_columns(predictions, EXPECTED_GOALS_COLUMNS, baseline)
        scores = []
        for expected_a, expected_b in zip(predictions["expected_goals_a"], predictions["expected_goals_b"]):
            probabilities = score_probability_matrix(expected_a, expected_b, max_goals=max_goals)
            if baseline == "most_likely_poisson":
                scores.append(most_likely_poisson_score(expected_a, expected_b, max_goals=max_goals))
            else:
                safe = get_safe_prediction(probabilities, max_goals=max_goals)
                scores.append((safe["pred_a"], safe["pred_b"]))
        predictions[["pred_a", "pred_b"]] = pd.DataFrame(scores, index=predictions.index)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return predictions


def evaluate_baselines(matches: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Compare all requested score-prediction baselines."""
    validate_columns(matches, ACTUAL_SCORE_COLUMNS, "matches")
    baselines = [
        "always_1_1",
        "favorite_1_0",
        "favorite_2_0_if_strong",
        "most_likely_poisson",
        "expected_points_optimized",
    ]
    rows = []
    for baseline in baselines:
        predictions = add_baseline_predictions(matches, baseline, max_goals=max_goals)
        metrics = {
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
        rows.append({"baseline": baseline, **metrics})
    return pd.DataFrame(rows).sort_values(
        ["average_challenge_points", "result_accuracy"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
