"""Poisson score matrices with optional Dixon-Coles adjustment."""

import math

import numpy as np


SUPPORTED_METHODS = {"independent", "dixon_coles"}


def poisson_pmf(k: int, rate: float) -> float:
    """Return the Poisson probability of k goals at the given rate."""
    if k < 0:
        return 0.0
    if rate < 0:
        raise ValueError("rate must be non-negative")
    return math.exp(-rate) * rate**k / math.factorial(k)


def normalize_score_probs(score_probs: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a score-probability matrix."""
    probabilities = np.asarray(score_probs, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("score_probs must be a two-dimensional matrix.")
    if not np.isfinite(probabilities).all():
        raise ValueError("score_probs must contain only finite values.")
    if (probabilities < 0).any():
        raise ValueError("score_probs must not contain negative probabilities.")
    probability_sum = probabilities.sum()
    if probability_sum <= 0:
        raise ValueError("score_probs must have a positive sum.")
    return probabilities / probability_sum


def apply_dixon_coles_adjustment(
    score_probs: np.ndarray,
    lambda_a: float,
    lambda_b: float,
    rho: float = 0.05,
) -> np.ndarray:
    """Adjust only 0-0, 1-0, 0-1, and 1-1 score probabilities."""
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("Expected goals must be non-negative.")

    adjusted = np.asarray(score_probs, dtype=float).copy()
    if adjusted.ndim != 2 or adjusted.shape[0] < 2 or adjusted.shape[1] < 2:
        raise ValueError("score_probs must include at least goals 0 and 1 for both teams.")

    adjusted[0, 0] *= 1 - lambda_a * lambda_b * rho
    adjusted[1, 0] *= 1 + lambda_b * rho
    adjusted[0, 1] *= 1 + lambda_a * rho
    adjusted[1, 1] *= 1 - rho

    if (adjusted < 0).any():
        raise ValueError("Dixon-Coles adjustment produced a negative probability. Reduce rho.")
    return normalize_score_probs(adjusted)


def score_probability_matrix(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 6,
    method: str = "independent",
    dc_rho: float = 0.05,
) -> np.ndarray:
    """Return a normalized score matrix indexed as [team_a_goals, team_b_goals]."""
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("Expected goals must be non-negative.")
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}. Expected one of {sorted(SUPPORTED_METHODS)}.")

    probabilities_a = np.asarray([poisson_pmf(goals, lambda_a) for goals in range(max_goals + 1)])
    probabilities_b = np.asarray([poisson_pmf(goals, lambda_b) for goals in range(max_goals + 1)])
    score_probs = normalize_score_probs(np.outer(probabilities_a, probabilities_b))
    if method == "dixon_coles":
        return apply_dixon_coles_adjustment(score_probs, lambda_a, lambda_b, rho=dc_rho)
    return score_probs


def summarize_score_probs(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 6,
    method: str = "independent",
) -> dict[str, object]:
    """Return the score matrix and W/D/L probabilities for one match."""
    score_probs = score_probability_matrix(lambda_a, lambda_b, max_goals=max_goals, method=method)
    team_a_win = float(np.tril(score_probs, k=-1).sum())
    draw = float(np.trace(score_probs))
    team_b_win = float(np.triu(score_probs, k=1).sum())
    return {
        "score_probs": score_probs,
        "team_a_win": team_a_win,
        "draw": draw,
        "team_b_win": team_b_win,
        "wdl_sum": team_a_win + draw + team_b_win,
        "method": method,
    }
