from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting import EXPANDED_TOURNAMENT_BACKTESTS, historical_tournament_matches
from expanded_backtest_analysis import (
    _bucket_policy_predictions,
    _lambda_favorite_fixed,
    compute_totals_scale_factors,
)
from penka_scoring import FINAL, GROUP, QUARTER_FINAL, ROUND_OF_16, THIRD_PLACE


def _synthetic_tournament(name: str, year: int, n_matches: int) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-06-10", periods=n_matches, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "home_team": [f"Team {i}" for i in range(n_matches)],
            "away_team": [f"Team {i + 100}" for i in range(n_matches)],
            "home_score": 1,
            "away_score": 0,
            "tournament": name,
            "city": "City",
            "country": "Country",
            "neutral": True,
        }
    )


class TestTournamentSpecs:
    def test_specs_cover_eight_tournaments(self):
        assert len(EXPANDED_TOURNAMENT_BACKTESTS) == 8
        labels = {spec["label"] for spec in EXPANDED_TOURNAMENT_BACKTESTS}
        assert {"euro_2016", "euro_2021", "euro_2024", "copa_2015", "copa_2024"} <= labels

    def test_phase_sequence_assignment(self):
        spec = next(s for s in EXPANDED_TOURNAMENT_BACKTESTS if s["label"] == "euro_2016")
        results = _synthetic_tournament("UEFA Euro", 2016, 51)
        matches = historical_tournament_matches(results, spec)
        phases = matches["penka_phase"].tolist()
        assert phases[:36] == [GROUP] * 36
        assert phases[36:44] == [ROUND_OF_16] * 8
        assert phases[-1] == FINAL
        assert THIRD_PLACE not in phases  # Euros have no third-place match

    def test_copa_includes_third_place(self):
        spec = next(s for s in EXPANDED_TOURNAMENT_BACKTESTS if s["label"] == "copa_2019")
        results = _synthetic_tournament("Copa América", 2019, 26)
        matches = historical_tournament_matches(results, spec)
        phases = matches["penka_phase"].tolist()
        assert phases[:18] == [GROUP] * 18
        assert phases[18:22] == [QUARTER_FINAL] * 4
        assert phases[-2] == THIRD_PLACE and phases[-1] == FINAL

    def test_wrong_match_count_fails_loudly(self):
        spec = next(s for s in EXPANDED_TOURNAMENT_BACKTESTS if s["label"] == "euro_2024")
        results = _synthetic_tournament("UEFA Euro", 2024, 50)  # one short
        with pytest.raises(ValueError, match="expected 51"):
            historical_tournament_matches(results, spec)


class TestTotalsScale:
    def test_uses_only_pre_cutoff_history(self):
        predictions = pd.DataFrame(
            {
                "backtest_label": ["t1"] * 4,
                "training_cutoff": ["2018-06-14"] * 4,
                "calibrated_expected_goals_a": [1.0, 1.2, 0.9, 1.1],
                "calibrated_expected_goals_b": [0.8, 0.7, 1.0, 0.9],
            }
        )
        history = pd.DataFrame(
            {
                "date": ["2014-06-15", "2016-06-15", "2018-07-01"],  # last is post-cutoff
                "tournament": ["FIFA World Cup", "UEFA Euro", "FIFA World Cup"],
                "home_score": [3, 2, 9],  # post-cutoff 9-9 must be ignored
                "away_score": [1, 1, 9],
            }
        )
        scale = compute_totals_scale_factors(predictions, history)
        # pre-cutoff mean total = (4 + 3) / 2 = 3.5; predicted mean total = 1.9
        assert scale.iloc[0] == pytest.approx(min(3.5 / 1.9, 1.30))
        assert (scale == scale.iloc[0]).all()


class TestPolicies:
    def make_rows(self):
        return pd.DataFrame(
            {
                "backtest_year": [2018, 2018],
                "penka_phase": [GROUP, GROUP],
                "goals_a": [2, 0],
                "goals_b": [1, 0],
                "calibrated_expected_goals_a": [1.8, 1.0],
                "calibrated_expected_goals_b": [0.8, 1.0],
            }
        )

    def test_lambda_favorite_orientation(self):
        picks = _lambda_favorite_fixed(self.make_rows(), 2, 1)
        assert (picks.loc[0, "pred_a"], picks.loc[0, "pred_b"]) == (2, 1)
        # exactly tied lambdas fall back to 1-1
        assert (picks.loc[1, "pred_a"], picks.loc[1, "pred_b"]) == (1, 1)

    def test_bucket_policy_branches(self):
        featured = pd.DataFrame(
            {
                "policy_favorite_probability": [0.80, 0.50, 0.40],
                "policy_draw_probability": [0.15, 0.22, 0.35],
                "policy_lambda_difference": [1.2, 0.5, 0.05],
                "policy_favorite_is_team_a": [True, False, True],
            }
        )
        picks = _bucket_policy_predictions(featured, 0.65, 0.15, 0.30)
        assert (picks.loc[0, "pred_a"], picks.loc[0, "pred_b"]) == (2, 0)  # strong favorite
        assert (picks.loc[1, "pred_a"], picks.loc[1, "pred_b"]) == (1, 2)  # default 2-1, team_b favorite
        assert (picks.loc[2, "pred_a"], picks.loc[2, "pred_b"]) == (1, 1)  # coin flip
