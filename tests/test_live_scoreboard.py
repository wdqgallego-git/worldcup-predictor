from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live_scoreboard import (
    build_results_template,
    parse_score,
    score_live_matches,
    write_scoreboard,
)


@pytest.fixture
def artifacts():
    recommendations = pd.DataFrame(
        [
            {
                "match_id": 1, "team_a": "Mexico", "team_b": "South Africa",
                "rank_a": 17.0, "rank_b": 59.0,
                "recommended_prediction": "1-0", "safe_prediction": "1-0",
                "favorite_1_0_prediction": "1-0",
                "market_blended_prediction": "2-1", "market_favorite_2_1_prediction": "2-1",
            },
            {
                "match_id": 2, "team_a": "South Korea", "team_b": "Czechia",
                "rank_a": 23.0, "rank_b": 40.0,
                "recommended_prediction": "1-0", "safe_prediction": "1-0",
                "favorite_1_0_prediction": "1-0",
                "market_blended_prediction": np.nan, "market_favorite_2_1_prediction": np.nan,
            },
        ]
    )
    final_predictions = pd.DataFrame(
        [
            {"match_id": 1, "date": "2026-06-11", "time": "13:00 UTC-6", "team_a": "Mexico",
             "team_b": "South Africa", "aggressive_prediction": "2-0", "expected_penka_points": 1.8},
            {"match_id": 2, "date": "2026-06-11", "time": "20:00 UTC-6", "team_a": "South Korea",
             "team_b": "Czechia", "aggressive_prediction": "2-1", "expected_penka_points": 1.5},
        ]
    )
    return recommendations, final_predictions


class TestParsing:
    def test_parse_score(self):
        assert parse_score("2-1") == (2, 1)
        assert parse_score("") is None
        assert parse_score(np.nan) is None
        with pytest.raises(ValueError):
            parse_score("two-one")


class TestTemplate:
    def test_template_has_all_fixtures_and_blank_goals(self, artifacts):
        recommendations, final_predictions = artifacts
        template = build_results_template(recommendations, final_predictions)
        assert len(template) == 2
        assert (template["goals_a"] == "").all()
        assert list(template.columns)[:5] == ["match_id", "date", "time", "team_a", "team_b"]


class TestScoring:
    def make_live(self, goals=("2", "1", "", "")):
        return pd.DataFrame(
            {
                "match_id": [1, 2],
                "date": ["2026-06-11"] * 2,
                "time": ["13:00 UTC-6", "20:00 UTC-6"],
                "team_a": ["Mexico", "South Korea"],
                "team_b": ["South Africa", "Czechia"],
                "goals_a": [goals[0], goals[2]],
                "goals_b": [goals[1], goals[3]],
                "submitted_prediction": ["", ""],
            }
        )

    def test_scores_played_and_skips_unplayed(self, artifacts):
        recommendations, final_predictions = artifacts
        board, standings = score_live_matches(self.make_live(), recommendations, final_predictions)
        # actual 2-1: submitted 1-0 -> correct goal difference = 3; market_blended 2-1 -> exact = 5
        row = board[board["match_id"].eq(1)].iloc[0]
        assert row["submitted_points"] == 3
        assert row["market_blended_points"] == 5
        assert bool(board[board["match_id"].eq(2)]["played"].iloc[0]) is False
        table = standings.set_index("strategy")
        assert table.loc["submitted", "points_so_far"] == 3
        assert table.loc["market_blended", "points_so_far"] == 5
        assert table.loc["submitted", "remaining_matches"] == 1

    def test_market_fallback_uses_non_market_analog(self, artifacts):
        recommendations, final_predictions = artifacts
        live = self.make_live(goals=("2", "1", "2", "1"))
        board, _ = score_live_matches(live, recommendations, final_predictions)
        row = board[board["match_id"].eq(2)].iloc[0]
        # no odds for match 2: market_favorite_2_1 falls back to rank-oriented 2-1 -> exact
        assert row["market_favorite_2_1_points"] == 5

    def test_submitted_override_wins(self, artifacts):
        recommendations, final_predictions = artifacts
        live = self.make_live()
        live.loc[0, "submitted_prediction"] = "2-1"
        board, _ = score_live_matches(live, recommendations, final_predictions)
        assert board[board["match_id"].eq(1)]["submitted_points"].iloc[0] == 5

    def test_half_entered_result_fails_loudly(self, artifacts):
        recommendations, final_predictions = artifacts
        live = self.make_live(goals=("2", "", "", ""))
        with pytest.raises(ValueError, match="both goals_a and goals_b"):
            score_live_matches(live, recommendations, final_predictions)

    def test_writes_artifacts(self, tmp_path, artifacts):
        recommendations, final_predictions = artifacts
        board, standings = score_live_matches(self.make_live(), recommendations, final_predictions)
        summary = write_scoreboard(board, standings, tmp_path)
        assert summary.exists()
        assert (tmp_path / "live_scoreboard.csv").exists()
        text = summary.read_text(encoding="utf-8")
        assert "matches_played: 1 of 2" in text
