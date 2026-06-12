"""Apply a user-selected submission strategy override to the recommendation artifacts.

Rewrites recommended_prediction in final_match_recommendations.csv and the pick
sheet without touching model outputs. The model's EV pick stays visible in
safe_prediction/model_prediction columns, so the override is reversible:
re-run with --strategy model_ev (or re-run run_final_predictions.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import orient_favorite_score

STRATEGIES = ("favorite_2_1", "favorite_1_0", "model_ev")
REASONS = {
    "favorite_2_1": (
        "User-selected: rank-oriented 2-1. Best tested strategy on the World Cup scope "
        "(1.799 pts/match over 2014/2018/2022; CI vs alternatives includes zero). "
        "WC group matches average ~2.6 goals; 2-1 sits on the most common winning margin."
    ),
    "favorite_1_0": "User-selected: rank-oriented 1-0 (best on low-scoring Euro/Copa environments).",
    "model_ev": "Model EV pick restored (penka_group_expected_value_refreshable).",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Override the recommended submission strategy.")
    parser.add_argument("--strategy", choices=STRATEGIES, default="favorite_2_1")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def override_predictions(recommendations: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "model_ev":
        return recommendations["safe_prediction"]
    goals = (2, 1) if strategy == "favorite_2_1" else (1, 0)
    return pd.Series(
        [
            "{}-{}".format(*orient_favorite_score(rank_a, rank_b, *goals))
            for rank_a, rank_b in zip(recommendations["rank_a"], recommendations["rank_b"])
        ],
        index=recommendations.index,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rec_path = output_dir / "final_match_recommendations.csv"
    recommendations = pd.read_csv(rec_path)
    promoted = recommendations.get("market_blend_promoted", pd.Series(False, index=recommendations.index))
    if bool(promoted.any()):
        raise SystemExit("Market gate has promoted rows; resolve that conflict before overriding manually.")
    recommendations["recommended_prediction"] = override_predictions(recommendations, args.strategy)
    recommendations["recommended_strategy"] = (
        f"{args.strategy}_user_selected" if args.strategy != "model_ev" else "penka_group_expected_value_refreshable"
    )
    recommendations["recommendation_reason"] = REASONS[args.strategy]
    recommendations.to_csv(rec_path, index=False)
    with pd.ExcelWriter(output_dir / "final_match_recommendations.xlsx", engine="openpyxl") as writer:
        recommendations.to_excel(writer, sheet_name="recommendations", index=False)
    sheet_path = output_dir / "pick_sheet.csv"
    if sheet_path.exists():
        sheet = pd.read_csv(sheet_path)
        merged = sheet.merge(
            recommendations[["match_id", "recommended_prediction", "recommended_strategy"]],
            on="match_id", how="left", suffixes=("_old", ""),
        )
        sheet["recommended_prediction"] = merged["recommended_prediction"]
        sheet["recommendation_source"] = merged["recommended_strategy"]
        sheet.to_csv(sheet_path, index=False)
        with pd.ExcelWriter(output_dir / "pick_sheet.xlsx", engine="openpyxl") as writer:
            sheet.to_excel(writer, sheet_name="pick_sheet", index=False)
    counts = recommendations["recommended_prediction"].value_counts().to_dict()
    print(f"Applied {args.strategy}: {counts}")
    changed = (recommendations["recommended_prediction"] != recommendations["safe_prediction"]).sum()
    print(f"{changed} of {len(recommendations)} picks differ from the model EV pick.")


if __name__ == "__main__":
    main()
