from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from odds_priors import (
    ODDS_STATUS_BLANK,
    ODDS_STATUS_OK,
    add_market_lambdas_to_consensus,
    attach_market_consensus,
    blend_market_lambdas,
    build_match_odds_consensus,
    calculate_overround,
    derive_market_lambdas,
    implied_total_goals_from_over_under,
    load_match_odds,
    normalized_implied_probabilities,
    shin_adjusted_probabilities,
    simple_implied_probabilities,
)
from poisson import summarize_score_probs


ODDS_HEADER = (
    "match_id,date,team_a,team_b,bookmaker,odds_team_a,odds_draw,odds_team_b,"
    "market_type,source_timestamp,over_under_line,odds_over,odds_under,spread,hours_before_kickoff"
)


def write_odds(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "match_odds.csv"
    path.write_text("\n".join([ODDS_HEADER, *rows]) + "\n", encoding="utf-8")
    return path


class TestDeVig:
    def test_simple_implied_probabilities_exceed_one(self):
        assert simple_implied_probabilities([1.95, 3.50, 4.20]).sum() > 1.0

    def test_normalization_sums_to_one(self):
        assert np.isclose(normalized_implied_probabilities([1.95, 3.50, 4.20]).sum(), 1.0)

    def test_overround_positive_for_real_book(self):
        assert calculate_overround([1.95, 3.50, 4.20]) > 0.0

    def test_shin_sums_to_one_and_shrinks_longshots(self):
        shin, insider_fraction = shin_adjusted_probabilities([1.30, 5.50, 11.00])
        proportional = normalized_implied_probabilities([1.30, 5.50, 11.00])
        assert np.isclose(shin.sum(), 1.0)
        assert insider_fraction > 0.0
        # Shin reallocates the vig mostly off the favorite, so the longshot
        # probability comes out below the proportional de-vig estimate.
        assert shin[2] < proportional[2]

    def test_odds_at_or_below_one_rejected(self):
        with pytest.raises(ValueError):
            simple_implied_probabilities([1.0, 3.0, 4.0])


class TestMarketLambdaDerivation:
    def test_roundtrip_recovers_known_lambdas(self):
        summary = summarize_score_probs(1.7, 1.1, max_goals=10)
        derived = derive_market_lambdas(summary["team_a_win"], summary["draw"], summary["team_b_win"])
        assert derived["market_lambda_a"] == pytest.approx(1.7, abs=0.1)
        assert derived["market_lambda_b"] == pytest.approx(1.1, abs=0.1)

    def test_total_goals_anchor_pins_the_sum(self):
        summary = summarize_score_probs(1.7, 1.1, max_goals=10)
        derived = derive_market_lambdas(
            summary["team_a_win"], summary["draw"], summary["team_b_win"], total_goals_target=3.4
        )
        total = derived["market_lambda_a"] + derived["market_lambda_b"]
        assert 2.8 < total < 3.5
        assert total > 1.7 + 1.1

    def test_supremacy_anchor_pins_the_difference(self):
        summary = summarize_score_probs(1.4, 1.4, max_goals=10)
        derived = derive_market_lambdas(
            summary["team_a_win"], summary["draw"], summary["team_b_win"], supremacy_target=0.8
        )
        assert derived["market_lambda_a"] - derived["market_lambda_b"] > 0.3

    def test_invalid_probabilities_rejected(self):
        with pytest.raises(ValueError):
            derive_market_lambdas(0.5, 0.0, 0.5)

    def test_over_under_total_with_even_odds_near_line(self):
        total = implied_total_goals_from_over_under(2.5, 0.5)
        assert 2.5 < total < 3.2  # Poisson median below mean, so T sits above the line

    def test_over_under_without_probability_returns_line(self):
        assert implied_total_goals_from_over_under(3.5) == 3.5


class TestBlend:
    def test_weights_interpolate(self):
        assert blend_market_lambdas(1.0, 2.0, 3.0, 0.0, 0.0) == (1.0, 2.0)
        assert blend_market_lambdas(1.0, 2.0, 3.0, 0.0, 1.0) == (3.0, 0.0)
        assert blend_market_lambdas(1.0, 2.0, 3.0, 0.0, 0.5) == (2.0, 1.0)

    def test_invalid_weight_rejected(self):
        with pytest.raises(ValueError):
            blend_market_lambdas(1.0, 1.0, 1.0, 1.0, 1.5)


class TestLoaderValidation:
    def test_blank_rows_fall_back_instead_of_crashing(self, tmp_path):
        path = write_odds(
            tmp_path,
            [
                "1,2026-06-11,Mexico,South Africa,Bet365,1.55,4.0,6.5,1x2,2026-06-11T10:00:00Z,,,,,",
                "2,2026-06-11,South Korea,Czechia,Bet365,,,,1x2,,,,,,",
            ],
        )
        odds = load_match_odds(path)
        assert list(odds["odds_status"]) == [ODDS_STATUS_OK, ODDS_STATUS_BLANK]
        consensus = build_match_odds_consensus(odds)
        assert list(consensus["match_id"]) == [1]

    def test_odds_not_above_one_flagged_by_row(self, tmp_path):
        path = write_odds(
            tmp_path,
            ["1,2026-06-11,Mexico,South Africa,Bet365,0.95,4.0,6.5,1x2,2026-06-11T10:00:00Z,,,,,"],
        )
        odds = load_match_odds(path)
        assert odds["odds_status"].iloc[0] == "invalid_odds_not_above_1"
        assert build_match_odds_consensus(odds).empty

    def test_fat_finger_overround_flagged(self, tmp_path):
        # 12.0 entered where 1.20 was meant: the book sums to far less than 1.
        path = write_odds(
            tmp_path,
            ["1,2026-06-11,Brazil,Qatar,Bet365,12.0,8.0,21.0,1x2,2026-06-11T10:00:00Z,,,,,"],
        )
        odds = load_match_odds(path)
        assert odds["odds_status"].iloc[0].startswith("suspect_arbitrage_overround")

    def test_strict_mode_raises_on_flagged_rows(self, tmp_path):
        path = write_odds(
            tmp_path,
            ["1,2026-06-11,Brazil,Qatar,Bet365,12.0,8.0,21.0,1x2,2026-06-11T10:00:00Z,,,,,"],
        )
        with pytest.raises(ValueError):
            load_match_odds(path, strict=True)

    def test_bookmaker_disagreement_flagged(self, tmp_path):
        path = write_odds(
            tmp_path,
            [
                "1,2026-06-11,Mexico,South Africa,GoodBook,1.55,4.0,6.5,1x2,2026-06-11T10:00:00Z,,,,,",
                "1,2026-06-11,Mexico,South Africa,AlsoGood,1.57,4.1,6.4,1x2,2026-06-11T10:00:00Z,,,,,",
                "1,2026-06-11,Mexico,South Africa,FatFinger,5.15,4.0,1.65,1x2,2026-06-11T10:00:00Z,,,,,",
            ],
        )
        odds = load_match_odds(path)
        by_book = odds.set_index("bookmaker")["odds_status"]
        assert by_book["GoodBook"] == ODDS_STATUS_OK
        assert by_book["FatFinger"] == "suspect_bookmaker_disagreement"

    def test_post_kickoff_snapshot_flagged(self, tmp_path):
        path = write_odds(
            tmp_path,
            ["1,2026-06-11,Mexico,South Africa,Bet365,1.55,4.0,6.5,1x2,2026-06-11T22:00:00Z,,,,,-2.0"],
        )
        odds = load_match_odds(path)
        assert "invalid_post_kickoff_snapshot" in odds["odds_status"].iloc[0]

    def test_partial_1x2_flagged(self, tmp_path):
        path = write_odds(
            tmp_path,
            ["1,2026-06-11,Mexico,South Africa,Bet365,1.55,,6.5,1x2,2026-06-11T10:00:00Z,,,,,"],
        )
        odds = load_match_odds(path)
        assert odds["odds_status"].iloc[0] == "invalid_incomplete_1x2"

    def test_missing_file_returns_empty(self, tmp_path):
        odds = load_match_odds(tmp_path / "does_not_exist.csv")
        assert odds.empty
        assert build_match_odds_consensus(odds).empty


class TestConsensusAndOrientation:
    def make_consensus(self, tmp_path):
        path = write_odds(
            tmp_path,
            [
                "1,2026-06-11,Mexico,South Africa,Bet365,1.55,4.0,6.5,1x2,2026-06-11T10:00:00Z,2.5,1.9,1.9,-1.0,4.0",
                "1,2026-06-11,Mexico,South Africa,Pinnacle,1.57,4.1,6.4,1x2,2026-06-11T11:00:00Z,2.5,1.95,1.85,-1.0,3.0",
            ],
        )
        return add_market_lambdas_to_consensus(build_match_odds_consensus(load_match_odds(path)))

    def test_consensus_averages_books(self, tmp_path):
        consensus = self.make_consensus(tmp_path)
        assert len(consensus) == 1
        assert consensus["bookmaker_count"].iloc[0] == 2
        assert consensus["hours_before_kickoff"].iloc[0] == 3.0  # freshest snapshot wins
        probs = consensus[["prob_team_a", "prob_draw", "prob_team_b"]].iloc[0]
        assert np.isclose(probs.sum(), 1.0)
        assert consensus["market_lambda_a"].iloc[0] > consensus["market_lambda_b"].iloc[0]

    def test_reverse_orientation_swaps_probabilities_and_spread(self, tmp_path):
        consensus = self.make_consensus(tmp_path)
        fixtures = pd.DataFrame({"date": ["2026-06-11"], "team_a": ["South Africa"], "team_b": ["Mexico"]})
        attached = attach_market_consensus(fixtures, consensus)
        assert len(attached) == 1
        assert attached["prob_team_b"].iloc[0] > attached["prob_team_a"].iloc[0]
        assert attached["spread"].iloc[0] == 1.0
        assert attached["market_lambda_b"].iloc[0] > attached["market_lambda_a"].iloc[0]
