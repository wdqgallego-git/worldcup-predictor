"""Submission pick sheet with per-match odds provenance badges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_blend import select_promoted_market_prediction


PICK_SHEET_COLUMNS = [
    "match_id", "group", "date", "time", "team_a", "team_b",
    "model_prediction", "model_expected_points",
    "market_blended_prediction", "market_blended_expected_points",
    "recommended_prediction", "recommendation_source",
    "odds_provenance_badge", "odds_freshness",
    "market_prob_team_a", "market_prob_draw", "market_prob_team_b",
    "market_overround", "market_bookmaker_count", "market_bookmakers",
    "market_hours_before_kickoff", "market_odds_captured_at",
]
FRESH_HOURS_THRESHOLD = 6.0
NO_ODDS_BADGE = "no odds — model-only pick"


def provenance_badge(bookmakers: object, hours_before_kickoff: object, has_odds: bool) -> tuple[str, str]:
    """Return (badge, freshness) for one match, e.g. ('Bet365 · 2h pre-KO', 'green')."""
    if not has_odds:
        return NO_ODDS_BADGE, "none"
    books = str(bookmakers) if pd.notna(bookmakers) and str(bookmakers).strip() else "unknown book"
    if pd.isna(hours_before_kickoff):
        return f"{books} · freshness unknown", "amber"
    hours = float(hours_before_kickoff)
    if hours <= FRESH_HOURS_THRESHOLD:
        return f"{books} · {hours:.0f}h pre-KO", "green"
    if hours <= 48.0:
        return f"{books} · captured {hours:.0f}h pre-KO", "amber"
    return f"{books} · captured {hours / 24.0:.0f}d pre-KO", "amber"


def build_pick_sheet(predictions: pd.DataFrame, gate: dict[str, object]) -> pd.DataFrame:
    """Build the per-match pick sheet from market-annotated fixture predictions.

    The recommended pick only switches to the market blend when the backtest
    promotion gate passed AND the match actually has usable odds; everything else
    keeps the validated baseline pick.
    """
    sheet = predictions.copy()
    if "has_market_odds" not in sheet.columns:
        sheet["has_market_odds"] = False
    badges = [
        provenance_badge(
            row.get("market_bookmakers"),
            row.get("market_hours_before_kickoff"),
            bool(row.get("has_market_odds")),
        )
        for _, row in sheet.iterrows()
    ]
    sheet["odds_provenance_badge"] = [badge for badge, _ in badges]
    sheet["odds_freshness"] = [freshness for _, freshness in badges]
    sheet["model_prediction"] = sheet["safe_prediction"]
    sheet["model_expected_points"] = sheet["expected_penka_points"]
    use_blend, promoted_prediction, promoted_strategy = select_promoted_market_prediction(sheet, gate)
    sheet["recommended_prediction"] = np.where(
        use_blend, promoted_prediction, sheet["model_prediction"]
    )
    sheet["recommendation_source"] = np.where(
        use_blend,
        f"{promoted_strategy}_promoted_by_backtest_gate" if promoted_strategy else "market_strategy_promoted",
        np.where(
            sheet["has_market_odds"].astype(bool),
            "validated_baseline (market blend shown as candidate; gate not passed)",
            "validated_baseline (no odds entered)",
        ),
    )
    for column in PICK_SHEET_COLUMNS:
        if column not in sheet.columns:
            sheet[column] = np.nan
    return sheet[PICK_SHEET_COLUMNS].copy()


def write_pick_sheet(sheet: pd.DataFrame, output_dir: str | Path = "outputs") -> Path:
    """Write the pick sheet artifact as CSV and Excel."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "pick_sheet.csv"
    sheet.to_csv(csv_path, index=False)
    with pd.ExcelWriter(output_path / "pick_sheet.xlsx", engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="pick_sheet", index=False)
    return csv_path


def update_pick_sheet_row(
    blended_row: pd.Series,
    gate: dict[str, object],
    output_dir: str | Path = "outputs",
) -> pd.DataFrame:
    """Replace one match's pick-sheet row after a single-match re-optimization."""
    output_path = Path(output_dir)
    sheet_path = output_path / "pick_sheet.csv"
    if not sheet_path.exists():
        raise FileNotFoundError(f"{sheet_path} not found. Run scripts/run_final_predictions.py first.")
    sheet = pd.read_csv(sheet_path)
    updated = build_pick_sheet(blended_row.to_frame().T, gate)
    mask = sheet["match_id"].astype(str).eq(str(blended_row["match_id"]))
    if not mask.any():
        raise ValueError(f"match_id {blended_row['match_id']!r} not present in {sheet_path}.")
    for column in PICK_SHEET_COLUMNS:
        sheet.loc[mask, column] = updated.iloc[0][column]
    write_pick_sheet(sheet, output_path)
    return sheet[mask]
