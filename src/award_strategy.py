"""Company-game award strategy helpers."""

from config import COMPANY_SCORING, STRATEGIC_EV_MIN_RATIO


def expected_award_points(probability: float, award: str) -> float:
    """Convert an award probability into company-game expected points."""
    return probability * COMPANY_SCORING[award]


def strategic_pick_allowed(candidate_ev: float, safe_ev: float) -> bool:
    """Check whether a strategic pick keeps enough of the safe expected value."""
    if safe_ev <= 0:
        return candidate_ev >= safe_ev
    return candidate_ev / safe_ev >= STRATEGIC_EV_MIN_RATIO

