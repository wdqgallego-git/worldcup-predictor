"""Match score and result prediction skeleton."""

from .schemas import Match, ScorePrediction


def predict_match_score(match: Match) -> ScorePrediction:
    """Predict expected goals, result, and goal difference for one match."""
    raise NotImplementedError("Match prediction will be implemented in a later step.")

