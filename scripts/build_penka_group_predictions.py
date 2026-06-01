"""Build refreshable 2026 group-stage score recommendations under real Penka rules."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import add_raw_lambda_columns, apply_lambda_calibration, fit_pre_tournament_calibrator
from config import PREDICTION_REFERENCE_DATE
from data_loader import load_fixtures, load_rankings, load_results
from features import build_fixture_features, build_training_table
from model import predict_expected_goals, train_goal_models
from penka_predictions import build_penka_prediction_table, load_manual_adjustments, summarize_penka_predictions
from penka_scoring import GROUP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build refreshable Penka group-stage recommendations.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--manual-adjustments", default="data/manual/match_adjustments.csv")
    parser.add_argument("--rankings", default="data/raw/rankings.csv", help="Optional refreshed FIFA rankings CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results()
    rankings = load_rankings(args.rankings)
    fixtures = load_fixtures()
    group = fixtures[fixtures["stage"].eq("group_stage") & ~fixtures["has_placeholder_team"]].copy()
    if len(group) != 72:
        raise ValueError(f"Expected 72 resolved group-stage fixtures. Found {len(group)}.")
    training_df, feature_cols = build_training_table(results, rankings)
    models = train_goal_models(training_df, feature_cols, reference_date=PREDICTION_REFERENCE_DATE)
    calibrator, _ = fit_pre_tournament_calibrator(
        training_df,
        feature_cols,
        selected_model_name=models["selected_model_name"],
        reference_date=PREDICTION_REFERENCE_DATE,
        max_goals=5,
    )
    features, _ = build_fixture_features(group, results, rankings, feature_cols)
    predictions = add_raw_lambda_columns(predict_expected_goals(models, features))
    predictions = apply_lambda_calibration(predictions, calibrator)
    adjustments = load_manual_adjustments(args.manual_adjustments)
    table = build_penka_prediction_table(
        predictions,
        historical_results=results,
        phase_default=GROUP,
        max_goals=5,
        manual_adjustments=adjustments,
    )
    table.to_csv(output_dir / "penka_group_predictions.csv", index=False)
    with pd.ExcelWriter(output_dir / "penka_group_predictions.xlsx", engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="refreshable_group_predictions", index=False)
    summary = summarize_penka_predictions(table, "2026 Group Stage")
    (output_dir / "penka_group_predictions_summary.txt").write_text(summary, encoding="utf-8")
    print(f"Manual adjustment rows loaded: {len(adjustments)}")
    print(summary)
    print(f"Saved refreshable Penka group predictions to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
