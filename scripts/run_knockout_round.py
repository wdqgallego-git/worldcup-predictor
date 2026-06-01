"""Build one confirmed knockout round of 90-minute Penka predictions."""

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
from data_loader import load_rankings, load_results
from features import build_fixture_features, build_training_table
from model import predict_expected_goals, train_goal_models
from penka_predictions import build_penka_prediction_table, load_manual_adjustments, summarize_penka_predictions
from penka_scoring import VALID_PHASES


REQUIRED_COLUMNS = ["match_id", "phase", "date", "home_team", "away_team"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a confirmed Penka knockout round.")
    parser.add_argument("--fixtures", default="data/manual/knockout_round_fixtures.csv")
    parser.add_argument("--manual-adjustments", default="data/manual/match_adjustments.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--in-tournament-results", default="data/manual/in_tournament_results.csv")
    parser.add_argument("--rankings", default="data/raw/rankings.csv", help="Optional refreshed FIFA rankings CSV.")
    return parser.parse_args()


def load_confirmed_knockout_fixtures(path: str | Path) -> pd.DataFrame | None:
    fixture_path = Path(path)
    example = "match_id,phase,date,home_team,away_team\n73,round_of_32,2026-06-28,Mexico,Canada"
    if not fixture_path.exists():
        print(f"No confirmed knockout fixture file found: {fixture_path}")
        print("Create it once Penka publishes a confirmed round. Example:")
        print(example)
        return None
    fixtures = pd.read_csv(fixture_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(fixtures.columns))
    if missing:
        raise ValueError(f"{fixture_path} is missing required columns: {missing}")
    if fixtures.empty:
        print(f"No confirmed knockout fixtures are present in {fixture_path}.")
        print("Populate the template once Penka publishes a confirmed round. Example:")
        print(example)
        return None
    unknown = sorted(set(fixtures["phase"]) - set(VALID_PHASES))
    if unknown:
        raise ValueError(f"Unsupported knockout phases: {unknown}. Expected {sorted(VALID_PHASES)}")
    if fixtures["phase"].eq("group").any():
        raise ValueError("run_knockout_round.py accepts knockout phases only.")
    fixtures = fixtures.rename(columns={"home_team": "team_a", "away_team": "team_b"})
    fixtures["stage"] = "knockout"
    fixtures["round"] = fixtures["phase"]
    fixtures["group"] = ""
    fixtures["neutral"] = True
    fixtures["tournament"] = "FIFA World Cup"
    return fixtures


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    fixtures = load_confirmed_knockout_fixtures(args.fixtures)
    if fixtures is None:
        return
    results = load_results()
    in_tournament_path = Path(args.in_tournament_results)
    if in_tournament_path.exists():
        in_tournament = load_results(in_tournament_path)
        results = pd.concat([results, in_tournament], ignore_index=True).drop_duplicates()
        print(f"Loaded light in-tournament refresh rows: {len(in_tournament)}")
    rankings = load_rankings(args.rankings)
    training_df, feature_cols = build_training_table(results, rankings)
    models = train_goal_models(training_df, feature_cols, reference_date=PREDICTION_REFERENCE_DATE)
    calibrator, _ = fit_pre_tournament_calibrator(
        training_df,
        feature_cols,
        selected_model_name=models["selected_model_name"],
        reference_date=PREDICTION_REFERENCE_DATE,
        max_goals=5,
    )
    features, _ = build_fixture_features(fixtures, results, rankings, feature_cols)
    predictions = add_raw_lambda_columns(predict_expected_goals(models, features))
    predictions = apply_lambda_calibration(predictions, calibrator)
    adjustments = load_manual_adjustments(args.manual_adjustments)
    table = build_penka_prediction_table(
        predictions,
        historical_results=results,
        max_goals=5,
        manual_adjustments=adjustments,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "penka_knockout_round_predictions.csv", index=False)
    with pd.ExcelWriter(output_dir / "penka_knockout_round_predictions.xlsx", engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="confirmed_round_predictions", index=False)
    summary = summarize_penka_predictions(table, "Confirmed Knockout Round")
    summary += "\nKnockout historical results are diagnostic only. Final advice is phase-aware analytical Penka EV from calibrated 90-minute score probabilities.\n"
    (output_dir / "penka_knockout_round_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
