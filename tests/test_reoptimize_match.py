from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_blend import (
    apply_market_prior,
    load_market_consensus,
    parse_kickoff_utc,
    reoptimize_single_match,
    stamp_match_odds_rows,
)
from pick_sheet import build_pick_sheet, provenance_badge


ODDS_HEADER = (
    "match_id,date,team_a,team_b,bookmaker,odds_team_a,odds_draw,odds_team_b,"
    "market_type,source_timestamp,over_under_line,odds_over,odds_under,spread,"
    "hours_before_kickoff,implied_prob_team_a,implied_prob_draw,implied_prob_team_b"
)
FAKE_NOW = pd.Timestamp("2026-06-11T17:00:00Z")  # 2h before the 13:00 UTC-6 kickoff


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    odds_path = tmp_path / "match_odds.csv"
    odds_path.write_text(
        "\n".join(
            [
                ODDS_HEADER,
                "1,2026-06-11,Mexico,South Africa,Bet365,1.50,4.2,7.0,1x2,,,,,,,,,",
                "2,2026-06-11,South Korea,Czechia,Bet365,,,,1x2,,,,,,,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        [
            {
                "match_id": 1, "date": "2026-06-11", "time": "13:00 UTC-6", "group": "Group A",
                "team_a": "Mexico", "team_b": "South Africa",
                "expected_goals_a": 1.4, "expected_goals_b": 0.9,
                "safe_prediction": "1-0", "expected_penka_points": 1.6,
            },
            {
                "match_id": 2, "date": "2026-06-11", "time": "20:00 UTC-6", "group": "Group A",
                "team_a": "South Korea", "team_b": "Czechia",
                "expected_goals_a": 1.1, "expected_goals_b": 1.0,
                "safe_prediction": "1-0", "expected_penka_points": 1.4,
            },
        ]
    )
    predictions_path = tmp_path / "final_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    return {"odds": odds_path, "predictions": predictions_path, "dir": tmp_path}


class TestKickoffParsing:
    def test_utc_minus_six(self):
        kickoff = parse_kickoff_utc("2026-06-11", "13:00 UTC-6")
        assert kickoff == pd.Timestamp("2026-06-11T19:00:00Z")

    def test_plain_utc(self):
        assert parse_kickoff_utc("2026-06-11", "13:00") == pd.Timestamp("2026-06-11T13:00:00Z")

    def test_garbage_time_raises(self):
        with pytest.raises(ValueError):
            parse_kickoff_utc("2026-06-11", "afternoon")


class TestStamping:
    def test_stamp_records_now_and_hours_before_kickoff(self, workspace):
        kickoff = parse_kickoff_utc("2026-06-11", "13:00 UTC-6")
        stamped = stamp_match_odds_rows(workspace["odds"], 1, kickoff, now=FAKE_NOW)
        assert len(stamped) == 1
        raw = pd.read_csv(workspace["odds"])
        row = raw[raw["match_id"].eq(1)].iloc[0]
        assert row["source_timestamp"] == FAKE_NOW.isoformat()
        assert row["hours_before_kickoff"] == 2.0
        # implied probabilities are written back for provenance
        assert 0.5 < row["implied_prob_team_a"] < 0.75

    def test_stamp_unknown_match_raises(self, workspace):
        with pytest.raises(ValueError):
            stamp_match_odds_rows(workspace["odds"], 99, FAKE_NOW, now=FAKE_NOW)


class TestSingleMatchReoptimization:
    def test_reoptimize_blends_market_and_keeps_baseline_without_gate(self, workspace):
        result = reoptimize_single_match(
            1,
            odds_path=workspace["odds"],
            predictions_path=workspace["predictions"],
            gate_path=workspace["dir"] / "missing_gate.json",
            now=FAKE_NOW,
        )
        assert result["has_market_odds"] is True
        assert result["market_blended_prediction"] is not None
        # the gate has not passed, so the validated baseline pick stands
        assert result["market_blend_promoted"] is False
        assert result["recommended_prediction"] == "1-0"

    def test_reoptimize_promotes_when_gate_passed(self, workspace):
        gate_path = workspace["dir"] / "market_blend_gate.json"
        gate_path.write_text(json.dumps({"promote": True}), encoding="utf-8")
        result = reoptimize_single_match(
            1,
            odds_path=workspace["odds"],
            predictions_path=workspace["predictions"],
            gate_path=gate_path,
            now=FAKE_NOW,
        )
        assert result["market_blend_promoted"] is True
        assert result["recommended_prediction"] == result["market_blended_prediction"]

    def test_match_without_odds_falls_back_to_model_pick(self, workspace):
        result = reoptimize_single_match(
            2,
            odds_path=workspace["odds"],
            predictions_path=workspace["predictions"],
            gate_path=workspace["dir"] / "missing_gate.json",
            now=FAKE_NOW,
        )
        assert result["has_market_odds"] is False
        assert result["market_blended_prediction"] is None
        assert result["recommended_prediction"] == "1-0"

    def test_missing_predictions_file_raises(self, workspace):
        with pytest.raises(FileNotFoundError):
            reoptimize_single_match(
                1,
                odds_path=workspace["odds"],
                predictions_path=workspace["dir"] / "nope.csv",
                now=FAKE_NOW,
            )


class TestPickSheetBadges:
    def test_fresh_capture_is_green(self):
        badge, freshness = provenance_badge("Bet365", 2.0, True)
        assert badge == "Bet365 · 2h pre-KO"
        assert freshness == "green"

    def test_stale_capture_is_amber_in_days(self):
        badge, freshness = provenance_badge("Bet365", 120.0, True)
        assert badge == "Bet365 · captured 5d pre-KO"
        assert freshness == "amber"

    def test_no_odds_badge(self):
        badge, freshness = provenance_badge(None, None, False)
        assert badge == "no odds — model-only pick"
        assert freshness == "none"

    def test_pick_sheet_shows_candidate_without_promotion(self, workspace):
        kickoff = parse_kickoff_utc("2026-06-11", "13:00 UTC-6")
        stamp_match_odds_rows(workspace["odds"], 1, kickoff, now=FAKE_NOW)
        predictions = pd.read_csv(workspace["predictions"])
        consensus = load_market_consensus(workspace["odds"])
        annotated = apply_market_prior(predictions, consensus)
        sheet = build_pick_sheet(annotated, {"promote": False})
        with_odds = sheet[sheet["match_id"].eq(1)].iloc[0]
        without_odds = sheet[sheet["match_id"].eq(2)].iloc[0]
        assert with_odds["recommended_prediction"] == "1-0"
        assert "candidate" in with_odds["recommendation_source"]
        assert with_odds["odds_freshness"] == "green"
        assert pd.notna(with_odds["market_prob_team_a"])
        assert without_odds["odds_provenance_badge"] == "no odds — model-only pick"
