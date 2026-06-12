"""Blend manual market odds into fixture predictions as a gated candidate strategy."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import MARKET_BLEND_WEIGHT, MATCH_ODDS_PATH, MAX_GOALS_FINAL, SCORE_PMF_METHOD
from odds_priors import (
    ODDS_STATUS_OK,
    add_market_lambdas_to_consensus,
    attach_market_consensus,
    blend_market_lambdas,
    build_match_odds_consensus,
    load_market_blend_gate,
    load_match_odds,
)
from penka_scoring import GROUP
from poisson import score_probability_matrix
from prediction_optimizer import get_safe_prediction


MARKET_PREDICTION_COLUMNS = [
    "has_market_odds", "market_bookmaker_count", "market_bookmakers",
    "market_odds_captured_at", "market_hours_before_kickoff",
    "market_prob_team_a", "market_prob_draw", "market_prob_team_b", "market_overround",
    "market_lambda_a", "market_lambda_b",
    "blended_expected_goals_a", "blended_expected_goals_b",
    "market_blended_prediction", "market_blended_expected_points",
    "market_favorite_2_1_prediction", "market_favorite_1_0_prediction",
]
MARKET_FAVORITE_MIN_PROBABILITY_GAP = 0.03
PROMOTED_MARKET_COLUMNS = {
    "market_blended_optimizer": "market_blended_prediction",
    "market_favorite_2_1": "market_favorite_2_1_prediction",
    "market_favorite_1_0": "market_favorite_1_0_prediction",
}
_KICKOFF_TIME_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*(?:UTC\s*([+-]\d{1,2}(?:\.\d+)?))?\s*$")


def orient_market_fixed_score(
    prob_team_a: float,
    prob_team_b: float,
    rank_a: float | None,
    rank_b: float | None,
    favorite_goals: int,
    underdog_goals: int,
    min_probability_gap: float = MARKET_FAVORITE_MIN_PROBABILITY_GAP,
) -> str:
    """Orient a fixed score using the market, with ranking fallback for near-ties."""
    probabilities_available = np.isfinite(prob_team_a) and np.isfinite(prob_team_b)
    decisive_market = probabilities_available and abs(float(prob_team_a) - float(prob_team_b)) >= min_probability_gap
    if decisive_market:
        team_a_favorite = float(prob_team_a) > float(prob_team_b)
    elif rank_a is not None and rank_b is not None and np.isfinite(rank_a) and np.isfinite(rank_b):
        team_a_favorite = float(rank_a) < float(rank_b)
    elif probabilities_available:
        team_a_favorite = float(prob_team_a) >= float(prob_team_b)
    else:
        team_a_favorite = True
    return f"{favorite_goals}-{underdog_goals}" if team_a_favorite else f"{underdog_goals}-{favorite_goals}"


def select_promoted_market_prediction(
    predictions: pd.DataFrame,
    gate: dict[str, object],
) -> tuple[pd.Series, pd.Series, str | None]:
    """Return the gated market-pick mask and selected candidate, failing closed."""
    promoted_strategy = gate.get("promoted_strategy")
    if promoted_strategy is None and bool(gate.get("promote")):
        promoted_strategy = "market_blended_optimizer"
    candidate_column = PROMOTED_MARKET_COLUMNS.get(str(promoted_strategy))
    has_odds = predictions.get("has_market_odds", pd.Series(False, index=predictions.index)).astype(bool)
    if not bool(gate.get("promote")) or candidate_column is None or candidate_column not in predictions:
        return pd.Series(False, index=predictions.index), pd.Series(np.nan, index=predictions.index), None
    selected = predictions[candidate_column]
    return has_odds & selected.notna(), selected, str(promoted_strategy)


def parse_kickoff_utc(date: object, time_label: object) -> pd.Timestamp:
    """Parse a fixture date plus a 'HH:MM UTC-6'-style time label into a UTC timestamp."""
    day = pd.Timestamp(date)
    if pd.isna(day):
        raise ValueError(f"Fixture date is missing or unparseable: {date!r}")
    match = _KICKOFF_TIME_PATTERN.match(str(time_label)) if pd.notna(time_label) else None
    if match is None:
        raise ValueError(f"Fixture kickoff time is unparseable: {time_label!r}. Expected 'HH:MM UTC-6'.")
    hour, minute = int(match.group(1)), int(match.group(2))
    offset_hours = float(match.group(3)) if match.group(3) else 0.0
    local = day.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return (local - pd.Timedelta(hours=offset_hours)).tz_localize("UTC")


def load_market_consensus(
    odds_path: str | Path = MATCH_ODDS_PATH,
    method: str = SCORE_PMF_METHOD,
    strict: bool = False,
) -> pd.DataFrame:
    """Load entered odds, de-vig, average books, and derive market lambdas."""
    odds = load_match_odds(odds_path, strict=strict)
    return add_market_lambdas_to_consensus(build_match_odds_consensus(odds), method=method)


def apply_market_prior(
    fixture_predictions: pd.DataFrame,
    odds_consensus: pd.DataFrame,
    market_weight: float = MARKET_BLEND_WEIGHT,
    max_goals: int = MAX_GOALS_FINAL,
    method: str = SCORE_PMF_METHOD,
    phase: str = GROUP,
    lambda_a_column: str = "expected_goals_a",
    lambda_b_column: str = "expected_goals_b",
) -> pd.DataFrame:
    """Add market-blended lambdas and candidate picks; matches without odds stay model-only."""
    predictions = fixture_predictions.copy()
    text_columns = {
        "market_bookmakers", "market_odds_captured_at", "market_blended_prediction",
        "market_favorite_2_1_prediction", "market_favorite_1_0_prediction",
    }
    predictions["has_market_odds"] = False
    for column in MARKET_PREDICTION_COLUMNS[1:]:
        predictions[column] = pd.Series(
            np.nan, index=predictions.index, dtype="object" if column in text_columns else "float64"
        )
    if odds_consensus is None or odds_consensus.empty:
        return predictions
    keyed_columns = ["date", "team_a", "team_b", lambda_a_column, lambda_b_column]
    keyed_columns.extend(column for column in ("rank_a", "rank_b") if column in predictions.columns)
    keyed = predictions[keyed_columns].copy()
    keyed["prediction_index"] = predictions.index
    matched = attach_market_consensus(keyed, odds_consensus)
    if matched.empty:
        return predictions
    for row in matched.itertuples(index=False):
        if not np.isfinite(row.market_lambda_a) or not np.isfinite(row.market_lambda_b):
            continue
        model_lambda_a = float(getattr(row, lambda_a_column))
        model_lambda_b = float(getattr(row, lambda_b_column))
        blended_a, blended_b = blend_market_lambdas(
            model_lambda_a, model_lambda_b, float(row.market_lambda_a), float(row.market_lambda_b), market_weight
        )
        probabilities = score_probability_matrix(blended_a, blended_b, max_goals=max_goals, method=method)
        safe = get_safe_prediction(probabilities, max_goals=max_goals, phase=phase)
        index = row.prediction_index
        predictions.loc[index, "has_market_odds"] = True
        predictions.loc[index, "market_bookmaker_count"] = row.bookmaker_count
        predictions.loc[index, "market_bookmakers"] = row.bookmakers
        predictions.loc[index, "market_odds_captured_at"] = (
            row.source_timestamp.isoformat() if pd.notna(row.source_timestamp) else np.nan
        )
        predictions.loc[index, "market_hours_before_kickoff"] = row.hours_before_kickoff
        predictions.loc[index, "market_prob_team_a"] = row.prob_team_a
        predictions.loc[index, "market_prob_draw"] = row.prob_draw
        predictions.loc[index, "market_prob_team_b"] = row.prob_team_b
        predictions.loc[index, "market_overround"] = row.mean_overround
        predictions.loc[index, "market_lambda_a"] = row.market_lambda_a
        predictions.loc[index, "market_lambda_b"] = row.market_lambda_b
        predictions.loc[index, "blended_expected_goals_a"] = blended_a
        predictions.loc[index, "blended_expected_goals_b"] = blended_b
        predictions.loc[index, "market_blended_prediction"] = safe["prediction"]
        predictions.loc[index, "market_blended_expected_points"] = safe["expected_points"]
        rank_a = getattr(row, "rank_a", np.nan)
        rank_b = getattr(row, "rank_b", np.nan)
        predictions.loc[index, "market_favorite_2_1_prediction"] = orient_market_fixed_score(
            row.prob_team_a, row.prob_team_b, rank_a, rank_b, 2, 1
        )
        predictions.loc[index, "market_favorite_1_0_prediction"] = orient_market_fixed_score(
            row.prob_team_a, row.prob_team_b, rank_a, rank_b, 1, 0
        )
    return predictions


def stamp_match_odds_rows(
    odds_path: str | Path,
    match_id: object,
    kickoff_utc: pd.Timestamp,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stamp source_timestamp = now() and computed provenance onto one match's odds rows.

    Freshness is recorded from when this function runs, not typed by the user, so
    hours-before-kickoff provenance is honest by construction.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    path = Path(odds_path)
    raw = pd.read_csv(path)
    mask = raw["match_id"].astype(str).eq(str(match_id))
    if not mask.any():
        raise ValueError(f"No odds rows found for match_id {match_id!r} in {path}.")
    if "source_timestamp" not in raw.columns:
        raw["source_timestamp"] = pd.NA
    raw["source_timestamp"] = raw["source_timestamp"].astype("object")
    raw.loc[mask, "source_timestamp"] = now.isoformat()
    hours = float((kickoff_utc - now) / pd.Timedelta(hours=1))
    for column in ("hours_before_kickoff", "implied_prob_team_a", "implied_prob_draw", "implied_prob_team_b"):
        if column not in raw.columns:
            raw[column] = np.nan
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw.loc[mask, "hours_before_kickoff"] = round(hours, 2)
    raw.to_csv(path, index=False)
    loaded = load_match_odds(path)
    stamped = loaded[loaded["match_id"].astype(str).eq(str(match_id))]
    usable = stamped[stamped["odds_status"].eq(ODDS_STATUS_OK)]
    if not usable.empty:
        for column in ("implied_prob_team_a", "implied_prob_draw", "implied_prob_team_b"):
            raw.loc[usable.index, column] = usable[column].round(6).to_numpy()
        raw.to_csv(path, index=False)
    return stamped


def reoptimize_single_match(
    match_id: object,
    odds_path: str | Path = MATCH_ODDS_PATH,
    predictions_path: str | Path = "outputs/final_predictions.csv",
    fixtures: pd.DataFrame | None = None,
    market_weight: float = MARKET_BLEND_WEIGHT,
    max_goals: int = MAX_GOALS_FINAL,
    method: str = SCORE_PMF_METHOD,
    gate_path: str | Path = "outputs/market_blend_gate.json",
    now: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Recompute one fixture's pick from its current odds row without a full pipeline run."""
    predictions_file = Path(predictions_path)
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"{predictions_file} not found. Run scripts/run_final_predictions.py before re-optimizing a match."
        )
    predictions = pd.read_csv(predictions_file)
    match_rows = predictions[predictions["match_id"].astype(str).eq(str(match_id))]
    if match_rows.empty:
        raise ValueError(f"match_id {match_id!r} not present in {predictions_file}.")
    fixture = match_rows.iloc[0]
    kickoff = parse_kickoff_utc(fixture["date"], fixture.get("time"))
    stamped = stamp_match_odds_rows(odds_path, match_id, kickoff, now=now)
    consensus = load_market_consensus(odds_path, method=method)
    blended = apply_market_prior(
        match_rows.copy(),
        consensus,
        market_weight=market_weight,
        max_goals=max_goals,
        method=method,
    )
    gate = load_market_blend_gate(gate_path)
    result = blended.iloc[0]
    has_odds = bool(result["has_market_odds"])
    use_market, selected_market, promoted_strategy = select_promoted_market_prediction(blended, gate)
    promoted = bool(use_market.iloc[0])
    recommended = selected_market.iloc[0] if promoted else fixture["safe_prediction"]
    return {
        "match_id": fixture["match_id"],
        "team_a": fixture["team_a"],
        "team_b": fixture["team_b"],
        "kickoff_utc": kickoff.isoformat(),
        "odds_rows_stamped": int(len(stamped)),
        "has_market_odds": has_odds,
        "market_blend_promoted": promoted,
        "promoted_strategy": promoted_strategy,
        "model_prediction": fixture["safe_prediction"],
        "market_blended_prediction": result["market_blended_prediction"] if has_odds else None,
        "recommended_prediction": recommended,
        "blended_row": result,
    }
