"""Refreshable group and knockout score recommendations under real Penka rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from penka_scoring import GROUP, VALID_PHASES, infer_world_cup_phase, penka_phase_points
from poisson import summarize_score_probs
from prediction_optimizer import expected_points_for_prediction, summarize_match_strategy
from rule_b_analysis import ELO_INITIAL_RATING, latest_causal_elo_before_date


MANUAL_ADJUSTMENT_COLUMNS = [
    "match_id",
    "team",
    "lambda_attack_multiplier",
    "lambda_defense_multiplier",
    "rotation_risk",
    "key_absence_note",
    "must_win_status",
    "motivation_note",
    "lineup_confidence",
    "manual_note",
]
VALID_LINEUP_CONFIDENCE = {"unknown", "projected", "confirmed"}


def load_manual_adjustments(path: str | Path = "data/manual/match_adjustments.csv") -> pd.DataFrame:
    """Load optional human-maintained lambda adjustments without inventing news."""
    adjustment_path = Path(path)
    if not adjustment_path.exists():
        return pd.DataFrame(columns=MANUAL_ADJUSTMENT_COLUMNS)
    adjustments = pd.read_csv(adjustment_path)
    missing = sorted(set(MANUAL_ADJUSTMENT_COLUMNS) - set(adjustments.columns))
    if missing:
        raise ValueError(f"{adjustment_path} is missing required columns: {missing}")
    adjustments = adjustments.copy()
    adjustments["match_id"] = pd.to_numeric(adjustments["match_id"], errors="raise").astype(int)
    adjustments["lambda_attack_multiplier"] = pd.to_numeric(
        adjustments["lambda_attack_multiplier"], errors="coerce"
    ).fillna(1.0)
    adjustments["lambda_defense_multiplier"] = pd.to_numeric(
        adjustments["lambda_defense_multiplier"], errors="coerce"
    ).fillna(1.0)
    if (adjustments[["lambda_attack_multiplier", "lambda_defense_multiplier"]] <= 0).any().any():
        raise ValueError("Manual lambda multipliers must be positive.")
    adjustments["lineup_confidence"] = adjustments["lineup_confidence"].fillna("unknown").astype(str).str.lower()
    unknown = sorted(set(adjustments["lineup_confidence"]) - VALID_LINEUP_CONFIDENCE)
    if unknown:
        raise ValueError(f"Unknown lineup_confidence values: {unknown}. Expected {sorted(VALID_LINEUP_CONFIDENCE)}")
    return adjustments


def _team_adjustment(adjustments: pd.DataFrame, match_id: int, team: str) -> pd.Series | None:
    rows = adjustments[adjustments["match_id"].eq(int(match_id)) & adjustments["team"].eq(team)]
    if len(rows) > 1:
        raise ValueError(f"Duplicate manual adjustment rows for match_id={match_id}, team={team}")
    return None if rows.empty else rows.iloc[0]


def _diagnostic_text(rows: list[pd.Series | None]) -> str:
    fields = ("rotation_risk", "key_absence_note", "must_win_status", "motivation_note", "manual_note")
    notes = []
    for row in rows:
        if row is None:
            continue
        values = [str(row[field]).strip() for field in fields if pd.notna(row[field]) and str(row[field]).strip()]
        if values:
            notes.append(f"{row['team']}: " + " | ".join(values))
    return "; ".join(notes)


def apply_manual_match_adjustments(
    predictions: pd.DataFrame,
    adjustments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Scale attack lambdas and opponent concession factors using explicit manual rows."""
    adjusted = predictions.copy()
    adjustments = load_manual_adjustments() if adjustments is None else adjustments.copy()
    rows = []
    for match in adjusted.itertuples(index=False):
        team_a = _team_adjustment(adjustments, int(match.match_id), str(match.team_a))
        team_b = _team_adjustment(adjustments, int(match.match_id), str(match.team_b))
        attack_a = 1.0 if team_a is None else float(team_a["lambda_attack_multiplier"])
        defense_a = 1.0 if team_a is None else float(team_a["lambda_defense_multiplier"])
        attack_b = 1.0 if team_b is None else float(team_b["lambda_attack_multiplier"])
        defense_b = 1.0 if team_b is None else float(team_b["lambda_defense_multiplier"])
        before_a = float(match.expected_goals_a)
        before_b = float(match.expected_goals_b)
        after_a = float(np.clip(before_a * attack_a * defense_b, 0.05, 6.0))
        after_b = float(np.clip(before_b * attack_b * defense_a, 0.05, 6.0))
        confidences = [
            "unknown" if team_a is None else str(team_a["lineup_confidence"]),
            "unknown" if team_b is None else str(team_b["lineup_confidence"]),
        ]
        rows.append(
            {
                "manual_adjustment_applied": team_a is not None or team_b is not None,
                "lambda_before_manual_adjustment_home": before_a,
                "lambda_before_manual_adjustment_away": before_b,
                "lambda_after_manual_adjustment_home": after_a,
                "lambda_after_manual_adjustment_away": after_b,
                "adjustment_notes": _diagnostic_text([team_a, team_b]),
                "lineup_confidence": "/".join(confidences),
            }
        )
    manual = pd.DataFrame(rows, index=adjusted.index)
    adjusted = pd.concat([adjusted, manual], axis=1)
    adjusted["expected_goals_a"] = adjusted["lambda_after_manual_adjustment_home"]
    adjusted["expected_goals_b"] = adjusted["lambda_after_manual_adjustment_away"]
    return adjusted


def _favorite_score(team_a_is_favorite: bool, favorite_goals: int, underdog_goals: int = 0) -> tuple[int, int]:
    return (favorite_goals, underdog_goals) if team_a_is_favorite else (underdog_goals, favorite_goals)


def _score_ev(score: tuple[int, int], score_probs: np.ndarray, phase: str) -> float:
    return expected_points_for_prediction(score[0], score[1], score_probs, phase=phase)


def build_penka_prediction_table(
    fixture_predictions: pd.DataFrame,
    historical_results: pd.DataFrame,
    phase_default: str = GROUP,
    max_goals: int = 5,
    manual_adjustments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build refreshable Penka-EV rows from calibrated fixture lambdas."""
    adjusted = apply_manual_match_adjustments(fixture_predictions, adjustments=manual_adjustments)
    cutoff = pd.to_datetime(adjusted["date"], errors="raise").min()
    latest_elo = latest_causal_elo_before_date(historical_results, cutoff)
    output_rows = []
    for match in adjusted.itertuples(index=False):
        inferred = infer_world_cup_phase(match)
        phase = inferred if inferred in VALID_PHASES else phase_default
        penka_phase_points(phase)
        probabilities = summarize_score_probs(
            float(match.expected_goals_a),
            float(match.expected_goals_b),
            max_goals=max_goals,
        )
        matrix = probabilities["score_probs"]
        strategy = summarize_match_strategy(matrix, max_goals=max_goals, phase=phase)
        safe = strategy["safe_prediction"]
        team_a_is_favorite = (
            float(match.rank_a) < float(match.rank_b)
            if np.isfinite(float(match.rank_a)) and np.isfinite(float(match.rank_b)) and float(match.rank_a) != float(match.rank_b)
            else float(match.expected_goals_a) >= float(match.expected_goals_b)
        )
        favorite_team = match.team_a if team_a_is_favorite else match.team_b
        favorite_1_0 = _favorite_score(team_a_is_favorite, 1, 0)
        favorite_2_0 = _favorite_score(team_a_is_favorite, 2, 0)
        most_likely = np.unravel_index(np.argmax(matrix), matrix.shape)
        confidence = max(probabilities["team_a_win"], probabilities["draw"], probabilities["team_b_win"])
        output_rows.append(
            {
                "match_id": match.match_id,
                "phase": phase,
                "date": match.date,
                "group": getattr(match, "group", ""),
                "home_team": match.team_a,
                "away_team": match.team_b,
                "model_lambda_home": float(getattr(match, "raw_expected_goals_a", match.expected_goals_a)),
                "model_lambda_away": float(getattr(match, "raw_expected_goals_b", match.expected_goals_b)),
                "calibrated_lambda_home": float(getattr(match, "calibrated_expected_goals_a", match.expected_goals_a)),
                "calibrated_lambda_away": float(getattr(match, "calibrated_expected_goals_b", match.expected_goals_b)),
                "lambda_before_manual_adjustment": f"{match.lambda_before_manual_adjustment_home:.6f}/{match.lambda_before_manual_adjustment_away:.6f}",
                "lambda_after_manual_adjustment": f"{match.lambda_after_manual_adjustment_home:.6f}/{match.lambda_after_manual_adjustment_away:.6f}",
                "manual_adjustment_applied": bool(match.manual_adjustment_applied),
                "adjustment_notes": match.adjustment_notes,
                "lineup_confidence": match.lineup_confidence,
                "favorite_team": favorite_team,
                "rank_gap": abs(float(match.rank_a) - float(match.rank_b)),
                "elo_gap": abs(float(latest_elo.get(str(match.team_a), ELO_INITIAL_RATING)) - float(latest_elo.get(str(match.team_b), ELO_INITIAL_RATING))),
                "recommended_home_goals": safe["pred_a"],
                "recommended_away_goals": safe["pred_b"],
                "recommended_score": safe["prediction"],
                "expected_penka_points": safe["expected_points"],
                "exact_score_probability": safe["exact_score_probability"],
                "correct_goal_difference_or_draw_probability": safe["correct_goal_difference_or_draw_probability"],
                "correct_winner_probability": safe["correct_winner_probability"],
                "wrong_probability": safe["wrong_probability"],
                "baseline_favorite_1_0_score": f"{favorite_1_0[0]}-{favorite_1_0[1]}",
                "baseline_favorite_1_0_ev": _score_ev(favorite_1_0, matrix, phase),
                "baseline_favorite_2_0_score": f"{favorite_2_0[0]}-{favorite_2_0[1]}",
                "baseline_favorite_2_0_ev": _score_ev(favorite_2_0, matrix, phase),
                "draw_1_1_ev": _score_ev((1, 1), matrix, phase),
                "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
                "most_likely_score_ev": _score_ev((int(most_likely[0]), int(most_likely[1])), matrix, phase),
                "recommendation_method": "penka_expected_value_with_exact_score_tiebreak",
                "selected_by": safe["selected_by"],
                "tiebreak_changed_selection": safe["tiebreak_changed_selection"],
                "confidence_bucket": "high" if confidence >= 0.65 else "medium" if confidence >= 0.50 else "low",
                "refresh_required_before_match": True,
            }
        )
    return pd.DataFrame(output_rows)


def summarize_penka_predictions(predictions: pd.DataFrame, label: str) -> str:
    """Return an explicit refresh warning and compact score-count summary."""
    scores = predictions["recommended_score"]
    lines = [
        f"{label} Penka Predictions",
        "=" * (len(label) + 18),
        f"matches: {len(predictions)}",
        f"manual_adjustments_applied: {int(predictions['manual_adjustment_applied'].sum())}",
        f"exact_score_tiebreak_changes: {int(predictions['tiebreak_changed_selection'].sum())}",
        "",
        "IMPORTANT: These are refreshable baseline recommendations, not final submissions.",
        "Penka permits match predictions until 1 minute before kickoff. Refresh rankings, injuries, suspensions,",
        "confirmed lineups, rotation, goalkeeper changes, and matchday-3 incentives before each match.",
        "",
        "recommended_score_counts:",
        scores.value_counts().sort_index().to_string(),
    ]
    return "\n".join(lines) + "\n"

