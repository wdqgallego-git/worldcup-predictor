"""Run the expanded walk-forward backtest (World Cups + Euros + Copas) and decide strategy.

Each tournament is checkpointed to outputs/expanded/<label>_predictions.csv as it
finishes, so the run can be resumed safely. World Cup rows are reused from the
existing audited outputs/backtest_predictions.csv rather than retrained.
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

from backtesting import (
    EXPANDED_TOURNAMENT_BACKTESTS,
    WORLD_CUP_BACKTESTS,
    historical_tournament_matches,
    run_match_backtest,
)
from data_loader import load_rankings, load_results
from expanded_backtest_analysis import run_expanded_analysis


CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "expanded"
WC_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "backtest_predictions.csv"
EXPANDED_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "expanded_backtest_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expanded tournament backtest with strategy decision.")
    parser.add_argument("--analysis-only", action="store_true", help="Skip training; analyze existing checkpoints.")
    parser.add_argument(
        "--max-tournaments",
        type=int,
        default=len(EXPANDED_TOURNAMENT_BACKTESTS),
        help="Maximum new tournaments to train in this invocation (resume-friendly).",
    )
    return parser.parse_args()


def load_world_cup_rows() -> pd.DataFrame:
    """Reuse the audited World Cup walk-forward predictions with expanded labels."""
    if not WC_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"{WC_PREDICTIONS_PATH} missing; run scripts/run_backtest.py first.")
    rows = pd.read_csv(WC_PREDICTIONS_PATH)
    rows["backtest_label"] = "wc_" + rows["backtest_year"].astype(int).astype(str)
    rows["tournament_type"] = "world_cup"
    rows["training_cutoff"] = rows["backtest_year"].astype(int).map(WORLD_CUP_BACKTESTS)
    return rows


def run_pending_tournaments(max_tournaments: int) -> int:
    """Train and checkpoint tournaments that have no checkpoint yet."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [
        spec for spec in EXPANDED_TOURNAMENT_BACKTESTS
        if not (CHECKPOINT_DIR / f"{spec['label']}_predictions.csv").exists()
    ]
    if not pending:
        return 0
    results = load_results()
    rankings = load_rankings()
    completed = 0
    for spec in pending[:max_tournaments]:
        print(f"Backtesting {spec['label']} (cutoff {spec['cutoff']})...")
        matches = historical_tournament_matches(results, spec)
        backtest = run_match_backtest(
            int(spec["year"]),
            str(spec["cutoff"]),
            results=results,
            rankings=rankings,
            tournament_matches=matches,
            backtest_label=str(spec["label"]),
            tournament_type=str(spec["tournament_type"]),
        )
        backtest["predictions"].to_csv(CHECKPOINT_DIR / f"{spec['label']}_predictions.csv", index=False)
        completed += 1
        print(f"  checkpointed {spec['label']}: {len(backtest['predictions'])} matches")
    return completed


def main() -> None:
    args = parse_args()
    if not args.analysis_only:
        run_pending_tournaments(args.max_tournaments)
    missing = [
        spec["label"] for spec in EXPANDED_TOURNAMENT_BACKTESTS
        if not (CHECKPOINT_DIR / f"{spec['label']}_predictions.csv").exists()
    ]
    if missing:
        print(f"PENDING tournaments (re-run to continue): {missing}")
        return
    frames = [load_world_cup_rows()] + [
        pd.read_csv(CHECKPOINT_DIR / f"{spec['label']}_predictions.csv")
        for spec in EXPANDED_TOURNAMENT_BACKTESTS
    ]
    expanded = pd.concat(frames, ignore_index=True)
    expanded.to_csv(EXPANDED_OUTPUT_PATH, index=False)
    print(f"Wrote {len(expanded)} expanded backtest rows ({expanded['backtest_label'].nunique()} tournaments).")
    outputs = run_expanded_analysis(expanded, load_results())
    print("\nEXPANDED STRATEGY COMPARISON (group stage, main scopes)")
    comparison = outputs["comparison"]
    main_scopes = comparison[comparison["scope"].isin(["expanded_all_group", "wc_only_group"])]
    print(main_scopes.to_string(index=False))
    print("\nDECISION")
    print((PROJECT_ROOT / "outputs" / "expanded_strategy_decision.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
