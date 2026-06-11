from __future__ import annotations

import pytest

from penka_scoring import (
    FINAL,
    GROUP,
    QUARTER_FINAL,
    ROUND_OF_32,
    normalize_phase,
    penka_phase_points,
    prediction_category,
    score_prediction,
)


class TestGroupStageRules:
    """The confirmed Penka group rules: 5 exact, 3 goal-difference/draw, 2 winner, 0 wrong."""

    def test_exact_score(self):
        assert score_prediction(2, 1, 2, 1, GROUP) == 5

    def test_exact_draw(self):
        assert score_prediction(1, 1, 1, 1, GROUP) == 5

    def test_correct_goal_difference(self):
        assert score_prediction(2, 1, 1, 0, GROUP) == 3

    def test_correct_draw_wrong_score(self):
        assert score_prediction(1, 1, 0, 0, GROUP) == 3

    def test_correct_winner_only(self):
        assert score_prediction(1, 0, 3, 1, GROUP) == 2

    def test_wrong_direction(self):
        assert score_prediction(1, 2, 2, 0, GROUP) == 0

    def test_predicted_draw_actual_winner_is_wrong(self):
        assert score_prediction(1, 1, 2, 0, GROUP) == 0


class TestCategoryPriority:
    """Categories are mutually exclusive and checked exact -> goal difference -> winner."""

    def test_exact_beats_goal_difference(self):
        assert prediction_category(2, 1, 2, 1) == "exact_score"

    def test_goal_difference_beats_winner(self):
        assert prediction_category(3, 2, 1, 0) == "correct_goal_difference_or_draw"

    def test_winner_only(self):
        assert prediction_category(1, 0, 4, 2) == "correct_winner"

    def test_wrong(self):
        assert prediction_category(0, 1, 1, 0) == "wrong"


class TestPhaseAwareness:
    def test_round_of_32_points_differ_from_group(self):
        assert score_prediction(2, 0, 2, 0, ROUND_OF_32) == 8
        assert score_prediction(1, 0, 2, 0, ROUND_OF_32) == 3

    def test_quarter_final_points(self):
        assert score_prediction(3, 1, 2, 0, QUARTER_FINAL) == 7

    def test_phase_aliases_normalize(self):
        assert normalize_phase("Round of 32") == ROUND_OF_32
        assert normalize_phase("group-stage") == GROUP

    def test_unknown_phase_raises(self):
        with pytest.raises(ValueError):
            penka_phase_points("preliminary_round")

    def test_final_phase_exists(self):
        assert penka_phase_points(FINAL)["exact_points"] > penka_phase_points(GROUP)["exact_points"]


class TestInputValidation:
    def test_negative_goals_rejected(self):
        with pytest.raises(ValueError):
            score_prediction(-1, 0, 1, 0, GROUP)

    def test_non_integer_goals_rejected(self):
        with pytest.raises(TypeError):
            score_prediction(1.5, 0, 1, 0, GROUP)
