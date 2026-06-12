"""Run an isolated 2022 market-odds shadow test without touching the formal gate."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MARKET_BLEND_WEIGHT  # noqa: E402
from historical_odds_2022 import load_2022_backtest  # noqa: E402
from odds_priors import add_market_lambdas_to_consensus, attach_market_consensus, build_match_odds_consensus, load_match_odds  # noqa: E402
from penka_backtesting import MARKET_BLENDED_STRATEGY, build_strategy_points  # noqa: E402
from penka_scoring import prediction_category  # noqa: E402
from poisson import summarize_score_probs  # noqa: E402


ODDS_PATH = PROJECT_ROOT / "data" / "raw" / "historical_match_odds_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STRATEGIES = ["favorite_1_0", "favorite_2_1", "calibrated_optimizer", MARKET_BLENDED_STRATEGY]


def paired_bootstrap(values: np.ndarray, samples: int = 5000, seed: int = 2026) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def empty_outputs(reason: str) -> None:
    comparison_columns = [
        "scope", "strategy", "matches", "total_penka_points", "avg_penka_points",
        "exact_score_count", "exact_score_rate", "correct_winner_count", "correct_winner_rate",
        "correct_goal_difference_or_draw_count", "correct_goal_difference_or_draw_rate",
        "wrong_count", "wrong_rate", "difference_vs_favorite_1_0", "difference_vs_favorite_2_1",
        "difference_vs_model_only", "ci_vs_favorite_2_1_lower", "ci_vs_favorite_2_1_upper",
    ]
    change_columns = [
        "match_id", "date", "phase", "team_a", "team_b", "actual_score", "favorite_1_0_pick",
        "favorite_2_1_pick", "model_only_pick", "market_blended_pick", "market_favorite", "model_favorite",
        "market_implied_prob_team_a", "market_implied_prob_draw", "market_implied_prob_team_b",
        "model_prob_team_a", "model_prob_draw", "model_prob_team_b", "market_lambda_a", "market_lambda_b",
        "blended_lambda_a", "blended_lambda_b", "market_lambda_fit_error", "changed_from_model_only_flag",
        "changed_from_favorite_1_0_flag", "market_blended_points", "favorite_1_0_points",
        "favorite_2_1_points", "model_only_points", "explanation",
    ]
    pd.DataFrame(columns=comparison_columns).to_csv(OUTPUT_DIR / "market_blend_2022_shadow_backtest.csv", index=False)
    pd.DataFrame(columns=change_columns).to_csv(OUTPUT_DIR / "market_blend_2022_pick_changes.csv", index=False)
    summary = [
        "2022 Market Blend Shadow Backtest", "=================================",
        "track: none", "dataset_source: none", "rows_loaded: 0", "odds_status_flags: no_usable_odds_data=1",
        "matched_matches: 0 of 64", "group_stage_matched_matches: 0 of 48",
        "reference_cross_validation: manual_check_required", "market_blended_vs_favorite_1_0: unavailable",
        "market_blended_vs_favorite_2_1: unavailable", "market_blended_vs_model_only: unavailable",
        "picks_changed_by_market_odds: 0", "largest_gains: unavailable", "largest_losses: unavailable",
        "market_change_mechanism: unavailable", f"reason: {reason}",
        "should_influence_formal_gate: no, diagnostic only.",
    ]
    (OUTPUT_DIR / "market_blend_2022_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def favorite_label(row: pd.Series, prefix: str) -> str:
    probabilities = [row[f"{prefix}_prob_team_a"], row[f"{prefix}_prob_draw"], row[f"{prefix}_prob_team_b"]]
    index = int(np.argmax(probabilities))
    return row["team_a"] if index == 0 else "draw" if index == 1 else row["team_b"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ODDS_PATH.exists():
        empty_outputs("data/raw/historical_match_odds_2022.csv is missing")
        print((OUTPUT_DIR / "market_blend_2022_summary.txt").read_text(encoding="utf-8"))
        return
    raw_odds = pd.read_csv(ODDS_PATH)
    if raw_odds.empty:
        empty_outputs("historical_match_odds_2022.csv contains zero rows")
        print((OUTPUT_DIR / "market_blend_2022_summary.txt").read_text(encoding="utf-8"))
        return
    match_odds = load_match_odds(ODDS_PATH)
    consensus = add_market_lambdas_to_consensus(build_match_odds_consensus(match_odds))
    backtests = load_2022_backtest(PROJECT_ROOT / "outputs" / "backtest_predictions.csv")
    context, points = build_strategy_points(backtests, odds_consensus=consensus, market_blend_weight=MARKET_BLEND_WEIGHT)
    matched = context["market_odds_matched"].astype(bool)
    if not matched.any():
        empty_outputs("odds rows did not match 2022 backtest fixtures")
        print((OUTPUT_DIR / "market_blend_2022_summary.txt").read_text(encoding="utf-8"))
        return
    pivot_points = points.pivot(index="match_index", columns="strategy", values="penka_points")
    pivot_picks = points.pivot(index="match_index", columns="strategy", values="prediction")
    pivot_categories = points.pivot(index="match_index", columns="strategy", values="penka_category")
    scopes = {
        "2022_group_stage_matched": context.index[matched & context["penka_phase"].eq("group")],
        "2022_all_matched": context.index[matched],
    }
    comparison_rows = []
    for scope, indexes in scopes.items():
        for strategy in STRATEGIES:
            values = pivot_points.loc[indexes, strategy].to_numpy(float)
            categories = pivot_categories.loc[indexes, strategy]
            differences = {
                baseline: float(values.mean() - pivot_points.loc[indexes, baseline].mean())
                for baseline in ("favorite_1_0", "favorite_2_1", "calibrated_optimizer")
            }
            ci = paired_bootstrap(values - pivot_points.loc[indexes, "favorite_2_1"].to_numpy(float))
            comparison_rows.append(
                {
                    "scope": scope, "strategy": strategy, "matches": len(indexes),
                    "total_penka_points": int(values.sum()), "avg_penka_points": float(values.mean()),
                    "exact_score_count": int(categories.eq("exact_score").sum()), "exact_score_rate": float(categories.eq("exact_score").mean()),
                    "correct_winner_count": int(categories.eq("correct_winner").sum()), "correct_winner_rate": float(categories.eq("correct_winner").mean()),
                    "correct_goal_difference_or_draw_count": int(categories.eq("correct_goal_difference_or_draw").sum()),
                    "correct_goal_difference_or_draw_rate": float(categories.eq("correct_goal_difference_or_draw").mean()),
                    "wrong_count": int(categories.eq("wrong").sum()), "wrong_rate": float(categories.eq("wrong").mean()),
                    "difference_vs_favorite_1_0": differences["favorite_1_0"],
                    "difference_vs_favorite_2_1": differences["favorite_2_1"],
                    "difference_vs_model_only": differences["calibrated_optimizer"],
                    "ci_vs_favorite_2_1_lower": ci[0], "ci_vs_favorite_2_1_upper": ci[1],
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "market_blend_2022_shadow_backtest.csv", index=False)

    matched_context = context.loc[matched].copy()
    attached = attach_market_consensus(matched_context, consensus).set_index("context_index") if "context_index" in matched_context else attach_market_consensus(matched_context.assign(context_index=matched_context.index), consensus).set_index("context_index")
    changes = []
    for index, row in matched_context.iterrows():
        model_summary = summarize_score_probs(row.calibrated_expected_goals_a, row.calibrated_expected_goals_b, max_goals=6)
        market = attached.loc[index]
        model_probs = [model_summary["team_a_win"], model_summary["draw"], model_summary["team_b_win"]]
        model_favorite = row.team_a if np.argmax(model_probs) == 0 else "draw" if np.argmax(model_probs) == 1 else row.team_b
        market_favorite = market.team_a if market.prob_team_a >= max(market.prob_draw, market.prob_team_b) else "draw" if market.prob_draw >= market.prob_team_b else market.team_b
        model_pick = pivot_picks.loc[index, "calibrated_optimizer"]
        blend_pick = pivot_picks.loc[index, MARKET_BLENDED_STRATEGY]
        favorite_pick = pivot_picks.loc[index, "favorite_1_0"]
        if blend_pick == model_pick:
            explanation = "market odds did not change the model-only scoreline"
        elif ("-" in blend_pick and blend_pick.split("-")[0] == blend_pick.split("-")[1]):
            explanation = "market odds changed the recommendation to a draw"
        elif market_favorite != model_favorite:
            explanation = "market and model disagreed on winner direction"
        else:
            explanation = "market odds changed the scoreline total/margin while preserving winner direction"
        changes.append(
            {
                "match_id": int(row.shadow_match_id), "date": row.date.date().isoformat(), "phase": row.penka_phase,
                "team_a": row.team_a, "team_b": row.team_b, "actual_score": f"{int(row.goals_a)}-{int(row.goals_b)}",
                "favorite_1_0_pick": favorite_pick, "favorite_2_1_pick": pivot_picks.loc[index, "favorite_2_1"],
                "model_only_pick": model_pick, "market_blended_pick": blend_pick, "market_favorite": market_favorite,
                "model_favorite": model_favorite, "market_implied_prob_team_a": market.prob_team_a,
                "market_implied_prob_draw": market.prob_draw, "market_implied_prob_team_b": market.prob_team_b,
                "model_prob_team_a": model_probs[0], "model_prob_draw": model_probs[1], "model_prob_team_b": model_probs[2],
                "market_lambda_a": row.market_lambda_a, "market_lambda_b": row.market_lambda_b,
                "blended_lambda_a": row.blended_lambda_a, "blended_lambda_b": row.blended_lambda_b,
                "market_lambda_fit_error": row.market_lambda_fit_error, "changed_from_model_only_flag": blend_pick != model_pick,
                "changed_from_favorite_1_0_flag": blend_pick != favorite_pick,
                "market_blended_points": pivot_points.loc[index, MARKET_BLENDED_STRATEGY],
                "favorite_1_0_points": pivot_points.loc[index, "favorite_1_0"],
                "favorite_2_1_points": pivot_points.loc[index, "favorite_2_1"],
                "model_only_points": pivot_points.loc[index, "calibrated_optimizer"], "explanation": explanation,
            }
        )
    changes = pd.DataFrame(changes)
    changes.to_csv(OUTPUT_DIR / "market_blend_2022_pick_changes.csv", index=False)
    overall = comparison[comparison["scope"].eq("2022_all_matched")].set_index("strategy")
    blend = overall.loc[MARKET_BLENDED_STRATEGY]
    changed = changes[changes["changed_from_model_only_flag"]]
    gains = changed.assign(gain=changed["market_blended_points"] - changed["model_only_points"]).nlargest(5, "gain")
    losses = changed.assign(loss=changed["market_blended_points"] - changed["model_only_points"]).nsmallest(5, "loss")
    source_identity = ", ".join(sorted(raw_odds["source_name"].dropna().astype(str).unique()))
    summary = [
        "2022 Market Blend Shadow Backtest", "=================================", "track: Kaggle" if raw_odds["source_name"].astype(str).str.contains("kaggle", case=False).all() else "track: local HTML or mixed",
        f"dataset_source: {source_identity}", f"rows_loaded: {len(raw_odds)}", "odds_status_flags: see historical_odds_2022_qa.txt",
        f"matched_matches: {int(matched.sum())} of 64", f"group_stage_matched_matches: {int((matched & context['penka_phase'].eq('group')).sum())} of 48",
        "reference_cross_validation: see odds_2022_provenance_check.csv",
        f"market_blended_vs_favorite_1_0: {blend.avg_penka_points:.4f} vs {overall.loc['favorite_1_0','avg_penka_points']:.4f}",
        f"market_blended_vs_favorite_2_1: {blend.avg_penka_points:.4f} vs {overall.loc['favorite_2_1','avg_penka_points']:.4f}",
        f"market_blended_vs_model_only: {blend.avg_penka_points:.4f} vs {overall.loc['calibrated_optimizer','avg_penka_points']:.4f}",
        f"paired_bootstrap_ci_vs_favorite_2_1: [{blend.ci_vs_favorite_2_1_lower:.4f}, {blend.ci_vs_favorite_2_1_upper:.4f}] (single-tournament diagnostic only)",
        f"picks_changed_by_market_odds: {len(changed)}", "largest_gains:",
        *(f"  {row.team_a} vs {row.team_b}: {row.gain:+.0f}" for row in gains.itertuples(index=False)),
        "largest_losses:", *(f"  {row.team_a} vs {row.team_b}: {row.loss:+.0f}" for row in losses.itertuples(index=False)),
        "market_change_mechanism:", *(f"  {label}: {count}" for label, count in changed["explanation"].value_counts().items()),
        "should_influence_formal_gate: no, diagnostic only.",
    ]
    (OUTPUT_DIR / "market_blend_2022_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print((OUTPUT_DIR / "market_blend_2022_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
