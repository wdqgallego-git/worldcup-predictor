from __future__ import annotations

import numpy as np
import pandas as pd

from market_blend import orient_market_fixed_score
from penka_backtesting import (
    MARKET_BLENDED_STRATEGY,
    MARKET_FAVORITE_1_0_STRATEGY,
    MARKET_FAVORITE_2_1_STRATEGY,
    build_market_blend_gate_report,
)


def test_market_fixed_score_uses_market_then_rank_fallback() -> None:
    assert orient_market_fixed_score(0.60, 0.20, 20, 5, 2, 1) == "2-1"
    assert orient_market_fixed_score(0.20, 0.60, 5, 20, 1, 0) == "0-1"
    assert orient_market_fixed_score(0.49, 0.47, 12, 4, 2, 1) == "1-2"


def test_multi_candidate_gate_promotes_highest_passing_candidate() -> None:
    matches_per_year = 12
    years = np.repeat([2014, 2018, 2022], matches_per_year)
    context = pd.DataFrame(
        {
            "backtest_year": years,
            "market_odds_matched": True,
        }
    )
    rows = []
    strategy_values = {
        "favorite_2_1": np.full(len(context), 2),
        "favorite_1_0": np.full(len(context), 2),
        MARKET_BLENDED_STRATEGY: np.full(len(context), 2),
        MARKET_FAVORITE_2_1_STRATEGY: np.full(len(context), 4),
        MARKET_FAVORITE_1_0_STRATEGY: np.full(len(context), 3),
    }
    for strategy, values in strategy_values.items():
        for match_index, value in enumerate(values):
            rows.append({"match_index": match_index, "strategy": strategy, "penka_points": value})
    report, decision = build_market_blend_gate_report(context, pd.DataFrame(rows), 0.5)

    assert set(report["candidate_strategy"]) == {
        MARKET_BLENDED_STRATEGY,
        MARKET_FAVORITE_2_1_STRATEGY,
        MARKET_FAVORITE_1_0_STRATEGY,
    }
    assert decision["promote"] is True
    assert decision["promoted_strategy"] == MARKET_FAVORITE_2_1_STRATEGY
    assert decision["candidate_decisions"][MARKET_BLENDED_STRATEGY]["promote"] is False

