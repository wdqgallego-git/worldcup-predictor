"""Phase-aware, mutually exclusive Colombia Tech Fest Penka match scoring."""

from __future__ import annotations

from numbers import Real
from typing import Mapping

import pandas as pd

from contest_config import MATCH_SCORING


GROUP = "group"
ROUND_OF_32 = "round_of_32"
ROUND_OF_16 = "round_of_16"
QUARTER_FINAL = "quarter_final"
SEMI_FINAL = "semi_final"
THIRD_PLACE = "third_place"
FINAL = "final"
UNKNOWN_PHASE = "unknown_phase"
VALID_PHASES = tuple(MATCH_SCORING)


def normalize_phase(phase: object) -> str:
    """Normalize one supported phase name without guessing unknown values."""
    value = str(phase).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "group_stage": GROUP,
        "groups": GROUP,
        "round_of_32": ROUND_OF_32,
        "r32": ROUND_OF_32,
        "round_of_16": ROUND_OF_16,
        "last_16": ROUND_OF_16,
        "r16": ROUND_OF_16,
        "quarter_finals": QUARTER_FINAL,
        "quarterfinal": QUARTER_FINAL,
        "quarterfinals": QUARTER_FINAL,
        "semi_finals": SEMI_FINAL,
        "semifinal": SEMI_FINAL,
        "semifinals": SEMI_FINAL,
        "third_place_play_off": THIRD_PLACE,
        "third_place_playoff": THIRD_PLACE,
        "match_for_third_place": THIRD_PLACE,
    }
    return aliases.get(value, value)


def penka_phase_points(phase: str) -> dict[str, int]:
    """Return the mutually exclusive points for one Penka phase."""
    normalized = normalize_phase(phase)
    if normalized not in MATCH_SCORING:
        raise ValueError(f"Unknown Penka phase: {phase!r}. Expected one of {sorted(MATCH_SCORING)}.")
    return dict(MATCH_SCORING[normalized])


def _validate_goal(name: str, value: object) -> int:
    """Return a validated non-negative integer goal count."""
    if isinstance(value, bool) or not isinstance(value, Real) or not float(value).is_integer():
        raise TypeError(f"{name} must be a non-negative integer. Received: {value!r}")
    if float(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer. Received: {value!r}")
    return int(value)


def prediction_category(
    pred_home_goals: int,
    pred_away_goals: int,
    actual_home_goals: int,
    actual_away_goals: int,
) -> str:
    """Return the single mutually exclusive Penka outcome category."""
    pred_home = _validate_goal("pred_home_goals", pred_home_goals)
    pred_away = _validate_goal("pred_away_goals", pred_away_goals)
    actual_home = _validate_goal("actual_home_goals", actual_home_goals)
    actual_away = _validate_goal("actual_away_goals", actual_away_goals)
    if pred_home == actual_home and pred_away == actual_away:
        return "exact_score"
    pred_diff = pred_home - pred_away
    actual_diff = actual_home - actual_away
    if pred_diff == actual_diff:
        return "correct_goal_difference_or_draw"
    if (pred_diff > 0 and actual_diff > 0) or (pred_diff < 0 and actual_diff < 0) or (pred_diff == actual_diff == 0):
        return "correct_winner"
    return "wrong"


def score_prediction(
    pred_home_goals: int,
    pred_away_goals: int,
    actual_home_goals: int,
    actual_away_goals: int,
    phase: str,
) -> int:
    """Score one 90-minute score prediction under the real Penka rules."""
    points = penka_phase_points(phase)
    category = prediction_category(
        pred_home_goals,
        pred_away_goals,
        actual_home_goals,
        actual_away_goals,
    )
    return {
        "exact_score": points["exact_points"],
        "correct_goal_difference_or_draw": points["goal_difference_points"],
        "correct_winner": points["winner_points"],
        "wrong": points["wrong_points"],
    }[category]


def infer_world_cup_phase(row: Mapping[str, object] | pd.Series | object) -> str:
    """Infer a World Cup phase from source labels, preserving uncertainty."""
    if hasattr(row, "_asdict"):
        values = row._asdict()
    elif isinstance(row, Mapping):
        values = row
    elif isinstance(row, pd.Series):
        values = row.to_dict()
    else:
        values = vars(row)
    text = " ".join(
        str(values.get(column, ""))
        for column in ("penka_phase", "phase", "round", "stage", "group", "tournament")
        if values.get(column) is not None
    ).strip().lower()
    if "third" in text:
        return THIRD_PLACE
    if "semi" in text:
        return SEMI_FINAL
    if "quarter" in text:
        return QUARTER_FINAL
    if "round of 32" in text or "round_of_32" in text or "r32" in text:
        return ROUND_OF_32
    if "round of 16" in text or "round_of_16" in text or "last 16" in text or "r16" in text:
        return ROUND_OF_16
    if "final" in text:
        return FINAL
    if "group" in text:
        return GROUP
    return UNKNOWN_PHASE


def validate_penka_scoring_examples() -> None:
    """Run the confirmed Penka examples as lightweight internal tests."""
    examples = [
        (GROUP, 2, 0, 2, 0, 5),
        (GROUP, 3, 1, 2, 0, 3),
        (GROUP, 1, 0, 2, 0, 2),
        (GROUP, 1, 2, 2, 0, 0),
        (GROUP, 1, 1, 0, 0, 3),
        (GROUP, 1, 1, 1, 1, 5),
        (ROUND_OF_32, 2, 0, 2, 0, 8),
        (ROUND_OF_32, 3, 1, 2, 0, 5),
        (ROUND_OF_32, 1, 0, 2, 0, 3),
        (QUARTER_FINAL, 2, 0, 2, 0, 11),
        (QUARTER_FINAL, 3, 1, 2, 0, 7),
        (QUARTER_FINAL, 1, 0, 2, 0, 5),
    ]
    for phase, pred_home, pred_away, actual_home, actual_away, expected in examples:
        actual = score_prediction(pred_home, pred_away, actual_home, actual_away, phase)
        assert actual == expected, (phase, pred_home, pred_away, actual_home, actual_away, actual, expected)


if __name__ == "__main__":
    validate_penka_scoring_examples()
    print("Penka scoring examples passed.")
