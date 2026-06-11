"""Re-optimize one fixture's pick from its freshly entered closing odds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MARKET_BLEND_WEIGHT, MATCH_ODDS_PATH, SCORE_PMF_METHOD
from market_blend import reoptimize_single_match
from odds_priors import load_market_blend_gate
from pick_sheet import provenance_badge, update_pick_sheet_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overwrite one match's odds row in data/raw/match_odds.csv with the fresh closing line, "
            "then run this script: it stamps source_timestamp = now(), computes hours-before-kickoff, "
            "and re-optimizes only that fixture's pick."
        )
    )
    parser.add_argument("--match", required=True, help="match_id of the fixture to re-optimize.")
    parser.add_argument("--odds-path", default=MATCH_ODDS_PATH, help="Path to the manual odds CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Directory holding the prediction artifacts.")
    parser.add_argument(
        "--market-blend-weight",
        type=float,
        default=MARKET_BLEND_WEIGHT,
        help=f"Weight on market-implied lambdas. Defaults to {MARKET_BLEND_WEIGHT}.",
    )
    parser.add_argument("--score-method", default=SCORE_PMF_METHOD, help="Score-probability backend.")
    return parser.parse_args()


def update_recommendations_row(result: dict[str, object], output_dir: Path) -> None:
    """Refresh one row of final_match_recommendations.csv with the re-optimized pick."""
    path = output_dir / "final_match_recommendations.csv"
    if not path.exists():
        print(f"NOTE: {path} not found; skipped recommendations update.")
        return
    recommendations = pd.read_csv(path)
    mask = recommendations["match_id"].astype(str).eq(str(result["match_id"]))
    if not mask.any():
        print(f"NOTE: match_id {result['match_id']!r} not in {path}; skipped recommendations update.")
        return
    promoted = bool(result["market_blend_promoted"])
    for column in ("market_blended_prediction", "has_market_odds", "market_blend_promoted"):
        if column not in recommendations.columns:
            recommendations[column] = np.nan
    recommendations.loc[mask, "market_blended_prediction"] = result["market_blended_prediction"]
    recommendations.loc[mask, "has_market_odds"] = bool(result["has_market_odds"])
    recommendations.loc[mask, "market_blend_promoted"] = promoted
    recommendations.loc[mask, "recommended_prediction"] = result["recommended_prediction"]
    if promoted:
        recommendations.loc[mask, "recommended_strategy"] = "market_blended_prior_promoted_by_backtest_gate"
        recommendations.loc[mask, "recommendation_reason"] = (
            "Market-blended pick from a fresh pre-kickoff line; the blend passed the backtest gate."
        )
    recommendations.to_csv(path, index=False)
    with pd.ExcelWriter(output_dir / "final_match_recommendations.xlsx", engine="openpyxl") as writer:
        recommendations.to_excel(writer, sheet_name="recommendations", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    gate = load_market_blend_gate(output_dir / "market_blend_gate.json")
    result = reoptimize_single_match(
        args.match,
        odds_path=args.odds_path,
        predictions_path=output_dir / "final_predictions.csv",
        market_weight=args.market_blend_weight,
        method=args.score_method,
        gate_path=output_dir / "market_blend_gate.json",
    )
    blended_row = result["blended_row"]
    update_recommendations_row(result, output_dir)
    try:
        update_pick_sheet_row(blended_row, gate, output_dir)
    except FileNotFoundError as exc:
        print(f"NOTE: {exc} Skipped pick-sheet update.")
    badge, freshness = provenance_badge(
        blended_row.get("market_bookmakers"),
        blended_row.get("market_hours_before_kickoff"),
        bool(result["has_market_odds"]),
    )
    print(f"Re-optimized match {result['match_id']}: {result['team_a']} vs {result['team_b']}")
    print(f"  kickoff (UTC): {result['kickoff_utc']}")
    print(f"  odds rows stamped: {result['odds_rows_stamped']}")
    print(f"  provenance: {badge} [{freshness}]")
    if result["has_market_odds"]:
        print(
            f"  implied probabilities: {blended_row['market_prob_team_a']:.3f} / "
            f"{blended_row['market_prob_draw']:.3f} / {blended_row['market_prob_team_b']:.3f}"
        )
        print(f"  market-blended candidate pick: {result['market_blended_prediction']}")
    else:
        print("  no usable odds for this match; the model-only pick stands.")
    print(f"  model pick: {result['model_prediction']}")
    print(f"  recommended pick ({'blend promoted' if result['market_blend_promoted'] else 'validated baseline'}): {result['recommended_prediction']}")


if __name__ == "__main__":
    main()
