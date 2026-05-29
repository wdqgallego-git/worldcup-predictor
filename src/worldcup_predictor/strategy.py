"""Expected-points strategy helpers."""

from .config import AWARD_POINTS, STRATEGIC_EV_FLOOR, STRATEGIC_EV_TARGET


def expected_award_points(probability: float, award: str) -> float:
    """Convert an award probability into company-game expected points."""
    if award not in AWARD_POINTS:
        raise KeyError(f"Unknown award: {award}")
    return probability * AWARD_POINTS[award]


def is_strategic_pick_allowed(candidate_ev: float, safe_ev: float) -> bool:
    """Return whether a strategic pick keeps enough of the safe expected value."""
    if safe_ev <= 0:
        return candidate_ev >= safe_ev
    ev_ratio = candidate_ev / safe_ev
    return ev_ratio >= STRATEGIC_EV_FLOOR

