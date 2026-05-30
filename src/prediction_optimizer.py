"""Optimize score predictions for expected company challenge points."""

import numpy as np

from poisson import normalize_score_probs


EXACT_SCORE_POINTS = 5
CORRECT_RESULT_POINTS = 3
CORRECT_GOAL_DIFFERENCE_BONUS = 1
DEFAULT_MAX_GOALS = 6
DEFAULT_AGGRESSIVE_MIN_EV_RATIO = 0.95
DEFAULT_AGGRESSIVE_MIN_SCORE_PROBABILITY = 0.02
DEFAULT_AGGRESSIVE_MAX_ADDITIONAL_TOTAL_GOALS = 2
DEFAULT_AGGRESSIVE_MAX_GOALS_PER_TEAM = 4


def get_result(goals_a: int, goals_b: int) -> str:
    """Return a stable result label for a scoreline."""
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def points_for_prediction(pred_a: int, pred_b: int, actual_a: int, actual_b: int) -> int:
    """Return challenge points without double-counting an exact score."""
    if pred_a == actual_a and pred_b == actual_b:
        return EXACT_SCORE_POINTS

    points = 0
    if get_result(pred_a, pred_b) == get_result(actual_a, actual_b):
        points += CORRECT_RESULT_POINTS
    if pred_a - pred_b == actual_a - actual_b:
        points += CORRECT_GOAL_DIFFERENCE_BONUS
    return points


def validate_score_probs(score_probs: np.ndarray, max_goals: int) -> np.ndarray:
    """Normalize and validate a score matrix for the requested candidate range."""
    probabilities = normalize_score_probs(score_probs)
    required_size = max_goals + 1
    if probabilities.shape[0] < required_size or probabilities.shape[1] < required_size:
        raise ValueError(
            f"score_probs must include goals 0 through {max_goals} for both teams. "
            f"Received shape: {probabilities.shape}"
        )
    return probabilities


def expected_points_for_prediction(pred_a: int, pred_b: int, score_probs: np.ndarray) -> float:
    """Return expected challenge points for one predicted scoreline."""
    probabilities = normalize_score_probs(score_probs)
    expected_points = 0.0
    for actual_a in range(probabilities.shape[0]):
        for actual_b in range(probabilities.shape[1]):
            expected_points += probabilities[actual_a, actual_b] * points_for_prediction(
                pred_a,
                pred_b,
                actual_a,
                actual_b,
            )
    return float(expected_points)


def rank_prediction_candidates(score_probs: np.ndarray, max_goals: int = DEFAULT_MAX_GOALS) -> list[dict[str, object]]:
    """Rank every prediction candidate from 0-0 through max_goals-max_goals."""
    probabilities = validate_score_probs(score_probs, max_goals)
    candidates = []
    for pred_a in range(max_goals + 1):
        for pred_b in range(max_goals + 1):
            predicted_result = get_result(pred_a, pred_b)
            predicted_goal_difference = pred_a - pred_b
            candidates.append(
                {
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "prediction": f"{pred_a}-{pred_b}",
                    "result": predicted_result,
                    "goal_difference": predicted_goal_difference,
                    "expected_points": expected_points_for_prediction(pred_a, pred_b, probabilities),
                    "exact_score_probability": float(probabilities[pred_a, pred_b]),
                    "result_probability": float(
                        sum(
                            probabilities[actual_a, actual_b]
                            for actual_a in range(probabilities.shape[0])
                            for actual_b in range(probabilities.shape[1])
                            if get_result(actual_a, actual_b) == predicted_result
                        )
                    ),
                    "goal_difference_probability": float(
                        sum(
                            probabilities[actual_a, actual_b]
                            for actual_a in range(probabilities.shape[0])
                            for actual_b in range(probabilities.shape[1])
                            if actual_a - actual_b == predicted_goal_difference
                        )
                    ),
                }
            )
    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate["expected_points"],
            -candidate["exact_score_probability"],
            candidate["pred_a"] + candidate["pred_b"],
            candidate["pred_a"],
            candidate["pred_b"],
        ),
    )


def get_safe_prediction(score_probs: np.ndarray, max_goals: int = DEFAULT_MAX_GOALS) -> dict[str, object]:
    """Return the scoreline with the highest expected challenge points."""
    return rank_prediction_candidates(score_probs, max_goals=max_goals)[0]


def get_aggressive_prediction(
    score_probs: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    min_ev_ratio: float = DEFAULT_AGGRESSIVE_MIN_EV_RATIO,
    min_score_probability: float = DEFAULT_AGGRESSIVE_MIN_SCORE_PROBABILITY,
) -> dict[str, object]:
    """Return a realistic differentiated alternative while preserving safe EV."""
    if not 0 < min_ev_ratio <= 1:
        raise ValueError("min_ev_ratio must be greater than 0 and no greater than 1.")
    if not 0 <= min_score_probability <= 1:
        raise ValueError("min_score_probability must be between 0 and 1.")

    candidates = rank_prediction_candidates(score_probs, max_goals=max_goals)
    safe_prediction = candidates[0]
    minimum_expected_points = safe_prediction["expected_points"] * min_ev_ratio
    safe_total_goals = safe_prediction["pred_a"] + safe_prediction["pred_b"]
    safe_goal_difference = abs(safe_prediction["goal_difference"])
    eligible = [
        candidate
        for candidate in candidates
        if candidate["prediction"] != safe_prediction["prediction"]
        and candidate["expected_points"] >= minimum_expected_points
        and candidate["exact_score_probability"] >= min_score_probability
        and candidate["pred_a"] + candidate["pred_b"]
        <= safe_total_goals + DEFAULT_AGGRESSIVE_MAX_ADDITIONAL_TOTAL_GOALS
        and abs(candidate["goal_difference"]) <= safe_goal_difference + 1
        and max(candidate["pred_a"], candidate["pred_b"]) <= DEFAULT_AGGRESSIVE_MAX_GOALS_PER_TEAM
    ]
    aggressive = (eligible[0] if eligible else safe_prediction).copy()
    aggressive["ev_ratio"] = aggressive["expected_points"] / safe_prediction["expected_points"]
    return aggressive


def summarize_match_strategy(
    score_probs: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    top_n: int = 5,
) -> dict[str, object]:
    """Return safe, aggressive, and top-ranked score prediction candidates."""
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    candidates = rank_prediction_candidates(score_probs, max_goals=max_goals)
    safe_prediction = candidates[0].copy()
    safe_prediction["ev_ratio"] = 1.0
    aggressive_prediction = get_aggressive_prediction(score_probs, max_goals=max_goals)
    top_candidates = [candidate.copy() for candidate in candidates[:top_n]]
    for candidate in top_candidates:
        candidate["ev_ratio"] = candidate["expected_points"] / safe_prediction["expected_points"]
    return {
        "safe_prediction": safe_prediction,
        "aggressive_prediction": aggressive_prediction,
        "top_candidates": top_candidates,
    }
