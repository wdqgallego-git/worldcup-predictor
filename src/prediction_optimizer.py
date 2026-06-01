"""Optimize score predictions for expected Colombia Tech Fest Penka points."""

from __future__ import annotations

import numpy as np

from penka_scoring import GROUP, prediction_category, score_prediction
from poisson import normalize_score_probs


DEFAULT_MAX_GOALS = 5
DEFAULT_EV_TIE_THRESHOLD = 0.03
DEFAULT_MAX_RECOMMENDED_TEAM_GOALS = 4
DEFAULT_MAX_RECOMMENDED_TOTAL_GOALS = 5
DEFAULT_AGGRESSIVE_MIN_EV_RATIO = 0.95
DEFAULT_AGGRESSIVE_MIN_SCORE_PROBABILITY = 0.02
DEFAULT_AGGRESSIVE_MAX_ADDITIONAL_TOTAL_GOALS = 2


def get_result(goals_a: int, goals_b: int) -> str:
    """Return a stable 90-minute result label for a scoreline."""
    if goals_a > goals_b:
        return "team_a_win"
    if goals_b > goals_a:
        return "team_b_win"
    return "draw"


def points_for_prediction(
    pred_a: int,
    pred_b: int,
    actual_a: int,
    actual_b: int,
    phase: str = GROUP,
) -> int:
    """Compatibility wrapper around the centralized mutually exclusive Penka scorer."""
    return score_prediction(pred_a, pred_b, actual_a, actual_b, phase)


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


def prediction_category_probabilities(
    pred_a: int,
    pred_b: int,
    score_probs: np.ndarray,
) -> dict[str, float]:
    """Return mutually exclusive category probabilities for one score prediction."""
    probabilities = normalize_score_probs(score_probs)
    category_probs = {
        "exact_score": 0.0,
        "correct_goal_difference_or_draw": 0.0,
        "correct_winner": 0.0,
        "wrong": 0.0,
    }
    for actual_a in range(probabilities.shape[0]):
        for actual_b in range(probabilities.shape[1]):
            category = prediction_category(pred_a, pred_b, actual_a, actual_b)
            category_probs[category] += float(probabilities[actual_a, actual_b])
    return category_probs


def expected_points_for_prediction(
    pred_a: int,
    pred_b: int,
    score_probs: np.ndarray,
    phase: str = GROUP,
) -> float:
    """Return expected Penka points for one predicted 90-minute scoreline."""
    probabilities = normalize_score_probs(score_probs)
    expected_points = 0.0
    for actual_a in range(probabilities.shape[0]):
        for actual_b in range(probabilities.shape[1]):
            expected_points += probabilities[actual_a, actual_b] * score_prediction(
                pred_a,
                pred_b,
                actual_a,
                actual_b,
                phase,
            )
    return float(expected_points)


def _candidate_allowed(pred_a: int, pred_b: int, allow_diagnostic_scores: bool) -> bool:
    """Keep ordinary advice conservative while allowing explicit diagnostics."""
    return allow_diagnostic_scores or (
        max(pred_a, pred_b) <= DEFAULT_MAX_RECOMMENDED_TEAM_GOALS
        and pred_a + pred_b <= DEFAULT_MAX_RECOMMENDED_TOTAL_GOALS
    )


def rank_prediction_candidates(
    score_probs: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    phase: str = GROUP,
    allow_diagnostic_scores: bool = False,
) -> list[dict[str, object]]:
    """Rank feasible prediction candidates by real phase-specific Penka EV."""
    probabilities = validate_score_probs(score_probs, max_goals)
    candidates = []
    for pred_a in range(max_goals + 1):
        for pred_b in range(max_goals + 1):
            if not _candidate_allowed(pred_a, pred_b, allow_diagnostic_scores):
                continue
            category_probs = prediction_category_probabilities(pred_a, pred_b, probabilities)
            predicted_result = get_result(pred_a, pred_b)
            candidates.append(
                {
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "prediction": f"{pred_a}-{pred_b}",
                    "phase": phase,
                    "result": predicted_result,
                    "goal_difference": pred_a - pred_b,
                    "expected_points": expected_points_for_prediction(pred_a, pred_b, probabilities, phase=phase),
                    "exact_score_probability": category_probs["exact_score"],
                    "correct_goal_difference_or_draw_probability": category_probs[
                        "correct_goal_difference_or_draw"
                    ],
                    "correct_winner_probability": category_probs["correct_winner"],
                    "wrong_probability": category_probs["wrong"],
                    "result_probability": float(
                        category_probs["exact_score"]
                        + category_probs["correct_goal_difference_or_draw"]
                        + category_probs["correct_winner"]
                    ),
                    "goal_difference_probability": float(
                        category_probs["exact_score"] + category_probs["correct_goal_difference_or_draw"]
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


def select_prediction_candidate(
    candidates: list[dict[str, object]],
    ev_tie_threshold: float = DEFAULT_EV_TIE_THRESHOLD,
) -> dict[str, object]:
    """Use max EV, then prefer exact-score probability within the Penka tie window."""
    if not candidates:
        raise ValueError("At least one prediction candidate is required.")
    if ev_tie_threshold < 0:
        raise ValueError("ev_tie_threshold must be non-negative.")
    max_ev = float(candidates[0]["expected_points"])
    near_ties = [
        candidate
        for candidate in candidates
        if max_ev - float(candidate["expected_points"]) <= ev_tie_threshold + 1e-12
    ]
    selected = sorted(
        near_ties,
        key=lambda candidate: (
            -candidate["exact_score_probability"],
            candidate["pred_a"] + candidate["pred_b"],
            -candidate["expected_points"],
            candidate["pred_a"],
            candidate["pred_b"],
        ),
    )[0].copy()
    selected["selected_by"] = "max_ev" if selected["prediction"] == candidates[0]["prediction"] else "exact_score_tiebreak"
    selected["tiebreak_changed_selection"] = selected["selected_by"] == "exact_score_tiebreak"
    return selected


def get_safe_prediction(
    score_probs: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    phase: str = GROUP,
    ev_tie_threshold: float = DEFAULT_EV_TIE_THRESHOLD,
    allow_diagnostic_scores: bool = False,
) -> dict[str, object]:
    """Return the phase-aware Penka-EV scoreline with exact-score tie-breaking."""
    candidates = rank_prediction_candidates(
        score_probs,
        max_goals=max_goals,
        phase=phase,
        allow_diagnostic_scores=allow_diagnostic_scores,
    )
    return select_prediction_candidate(candidates, ev_tie_threshold=ev_tie_threshold)


def get_aggressive_prediction(
    score_probs: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    phase: str = GROUP,
    min_ev_ratio: float = DEFAULT_AGGRESSIVE_MIN_EV_RATIO,
    min_score_probability: float = DEFAULT_AGGRESSIVE_MIN_SCORE_PROBABILITY,
) -> dict[str, object]:
    """Return a nearby advisory alternative while preserving Penka EV."""
    if not 0 < min_ev_ratio <= 1:
        raise ValueError("min_ev_ratio must be greater than 0 and no greater than 1.")
    if not 0 <= min_score_probability <= 1:
        raise ValueError("min_score_probability must be between 0 and 1.")

    candidates = rank_prediction_candidates(score_probs, max_goals=max_goals, phase=phase)
    safe_prediction = get_safe_prediction(score_probs, max_goals=max_goals, phase=phase)
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
        <= min(DEFAULT_MAX_RECOMMENDED_TOTAL_GOALS, safe_total_goals + DEFAULT_AGGRESSIVE_MAX_ADDITIONAL_TOTAL_GOALS)
        and abs(candidate["goal_difference"]) <= safe_goal_difference + 1
    ]
    aggressive = (eligible[0] if eligible else safe_prediction).copy()
    aggressive["ev_ratio"] = aggressive["expected_points"] / safe_prediction["expected_points"]
    return aggressive


def summarize_match_strategy(
    score_probs: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
    top_n: int = 5,
    phase: str = GROUP,
) -> dict[str, object]:
    """Return safe, aggressive, and top-ranked phase-aware Penka candidates."""
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    candidates = rank_prediction_candidates(score_probs, max_goals=max_goals, phase=phase)
    safe_prediction = get_safe_prediction(score_probs, max_goals=max_goals, phase=phase)
    safe_prediction["ev_ratio"] = 1.0
    aggressive_prediction = get_aggressive_prediction(score_probs, max_goals=max_goals, phase=phase)
    top_candidates = [candidate.copy() for candidate in candidates[:top_n]]
    for candidate in top_candidates:
        candidate["ev_ratio"] = candidate["expected_points"] / safe_prediction["expected_points"]
    return {
        "safe_prediction": safe_prediction,
        "aggressive_prediction": aggressive_prediction,
        "top_candidates": top_candidates,
    }
