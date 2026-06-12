from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from historical_odds_2022 import (
    assign_backtest_identity,
    expand_source_rows,
    odds_to_decimal,
)


def test_odds_formats_convert_to_decimal():
    assert odds_to_decimal("2.50") == pytest.approx(2.5)
    assert odds_to_decimal("+150") == pytest.approx(2.5)
    assert odds_to_decimal("-200") == pytest.approx(1.5)
    assert odds_to_decimal("3/2") == pytest.approx(2.5)


def test_backtest_identity_matches_reversed_team_orientation():
    backtests = pd.DataFrame(
        {
            "shadow_match_id": [1],
            "date": [pd.Timestamp("2022-11-20")],
            "team_a": ["Qatar"],
            "team_b": ["Ecuador"],
            "penka_phase": ["group"],
        }
    )
    odds = pd.DataFrame(
        {
            "date": ["2022-11-20"],
            "team_a": ["Ecuador"],
            "team_b": ["Qatar"],
            "penka_phase": [pd.NA],
        }
    )
    matched, unmatched = assign_backtest_identity(odds, backtests)
    assert unmatched.empty
    assert int(matched.iloc[0]["match_id"]) == 1
    assert matched.iloc[0]["penka_phase"] == "group"


def test_backtest_identity_adds_phase_when_source_has_no_phase_column():
    backtests = pd.DataFrame(
        {
            "shadow_match_id": [1],
            "date": [pd.Timestamp("2022-11-20")],
            "team_a": ["Qatar"],
            "team_b": ["Ecuador"],
            "penka_phase": ["group"],
        }
    )
    odds = pd.DataFrame({"date": ["2022-11-20"], "team_a": ["Qatar"], "team_b": ["Ecuador"]})
    matched, unmatched = assign_backtest_identity(odds, backtests)
    assert unmatched.empty
    assert matched.iloc[0]["penka_phase"] == "group"


def test_model_probabilities_are_not_silently_treated_as_odds():
    frame = pd.DataFrame(
        {
            "date": ["2022-11-20"],
            "team1": ["Qatar"],
            "team2": ["Ecuador"],
            "prob1": [0.24],
            "probtie": [0.26],
            "prob2": [0.50],
        }
    )
    with pytest.raises(ValueError, match="No complete match-level 1X2 odds"):
        expand_source_rows(frame, {}, "auto", "kaggle_model_probabilities")
