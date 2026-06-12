from __future__ import annotations

import pandas as pd

from penka_backtesting import (
    MARKET_BLEND_BY_YEAR_COLUMNS,
    MARKET_BLEND_LAMBDA_AUDIT_COLUMNS,
    MARKET_BLENDED_STRATEGY,
    build_market_blend_backtest_by_year,
    build_market_blend_lambda_audit,
)
from penka_scoring import GROUP, ROUND_OF_16


def synthetic_market_context_and_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    context = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-06-12", "2014-06-28", "2018-06-14", "2018-06-30"]),
            "team_a": ["A", "C", "E", "G"],
            "team_b": ["B", "D", "F", "H"],
            "backtest_year": [2014, 2014, 2018, 2018],
            "penka_phase": [GROUP, ROUND_OF_16, GROUP, ROUND_OF_16],
            "market_odds_matched": [True, True, True, True],
            "market_match_id": [101, 102, 201, 202],
            "calibrated_expected_goals_a": [1.4, 1.2, 1.8, 1.1],
            "calibrated_expected_goals_b": [0.8, 1.0, 0.6, 1.0],
            "market_lambda_a": [1.5, 1.1, 2.0, 1.0],
            "market_lambda_b": [0.7, 1.1, 0.5, 1.1],
            "blended_lambda_a": [1.45, 1.15, 1.9, 1.05],
            "blended_lambda_b": [0.75, 1.05, 0.55, 1.05],
            "market_lambda_fit_error": [0.01, 0.02, 0.01, 0.03],
            "goals_a": [2, 1, 3, 0],
            "goals_b": [0, 1, 0, 1],
        }
    )
    strategy_points = {
        MARKET_BLENDED_STRATEGY: ([3, 2, 4, 1], ["2-0", "1-1", "2-0", "0-1"]),
        "favorite_2_1": ([2, 2, 3, 0], ["2-1", "2-1", "2-1", "2-1"]),
        "favorite_1_0": ([1, 3, 2, 1], ["1-0", "1-0", "1-0", "0-1"]),
    }
    rows = []
    for strategy, (scores, picks) in strategy_points.items():
        for match_index, (score, pick) in enumerate(zip(scores, picks)):
            rows.append(
                {
                    "match_index": match_index,
                    "strategy": strategy,
                    "penka_points": score,
                    "prediction": pick,
                }
            )
    return context, pd.DataFrame(rows)


def test_market_blend_by_year_has_both_labeled_scopes() -> None:
    context, points = synthetic_market_context_and_points()
    report = build_market_blend_backtest_by_year(context, points)

    assert list(report.columns) == MARKET_BLEND_BY_YEAR_COLUMNS
    assert set(report["scope"]) == {"matched_odds_only", "matched_odds_group_stage_only"}
    assert set(report["backtest_year"]) == {2014, 2018}
    assert set(report["baseline"]) == {"favorite_2_1", "favorite_1_0"}
    assert len(report) == 8
    group_rows = report[report["scope"].eq("matched_odds_group_stage_only")]
    assert group_rows["matches"].eq(1).all()


def test_market_blend_lambda_audit_is_one_row_per_matched_match() -> None:
    context, points = synthetic_market_context_and_points()
    audit = build_market_blend_lambda_audit(context, points)

    assert list(audit.columns) == MARKET_BLEND_LAMBDA_AUDIT_COLUMNS
    assert len(audit) == len(context)
    assert audit["match_id"].tolist() == [101, 102, 201, 202]
    assert audit["scope"].eq("matched_odds_only").all()
    assert audit["blended_pick"].tolist() == ["2-0", "1-1", "2-0", "0-1"]
