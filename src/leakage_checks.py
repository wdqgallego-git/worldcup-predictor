"""Fail-loud leakage and feature-quality checks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMNS = {"goals_a", "goals_b"}
PLACEHOLDER_FEATURE_PATTERNS = {
    "placeholder",
    "team_name_length",
    "name_length",
    "string_length",
    "final_standing",
    "final_rank",
    "post_match",
    "future_",
}
RANKING_DATE_COLUMN_PAIRS = [
    ("ranking_date_a", "ranking_date_b"),
    ("rank_date_a", "rank_date_b"),
]


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    """Raise a clear error when a leakage-check input is missing columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def check_no_future_matches_used(training_df: pd.DataFrame, cutoff_date: str | pd.Timestamp) -> bool:
    """Fail if any training row is after the permitted cutoff date."""
    require_columns(training_df, ["date"], "training_df")
    dates = pd.to_datetime(training_df["date"], errors="raise")
    cutoff = pd.Timestamp(cutoff_date)
    invalid = training_df.loc[dates > cutoff]
    if not invalid.empty:
        latest_date = dates.max()
        raise ValueError(
            f"Training leakage detected: {len(invalid)} rows are after cutoff {cutoff.date()}. "
            f"Latest training date: {latest_date.date()}."
        )
    return True


def find_ranking_date_columns(feature_df: pd.DataFrame) -> tuple[str, str]:
    """Find supported Team A / Team B ranking provenance columns."""
    for team_a_column, team_b_column in RANKING_DATE_COLUMN_PAIRS:
        if team_a_column in feature_df.columns and team_b_column in feature_df.columns:
            return team_a_column, team_b_column
    raise ValueError(
        "Ranking-date leakage check requires provenance columns. "
        "Expected ranking_date_a/ranking_date_b or rank_date_a/rank_date_b. "
        f"Available columns: {sorted(feature_df.columns.tolist())}"
    )


def check_ranking_dates_before_match(feature_df: pd.DataFrame) -> bool:
    """Fail if a ranking feature uses a ranking published on or after match date."""
    require_columns(feature_df, ["date"], "feature_df")
    ranking_date_a, ranking_date_b = find_ranking_date_columns(feature_df)
    match_dates = pd.to_datetime(feature_df["date"], errors="raise")
    for ranking_column in (ranking_date_a, ranking_date_b):
        ranking_dates = pd.to_datetime(feature_df[ranking_column], errors="coerce")
        invalid = ranking_dates.notna() & (ranking_dates >= match_dates)
        if invalid.any():
            example = feature_df.loc[invalid, ["date", ranking_column]].iloc[0].to_dict()
            raise ValueError(
                f"Ranking leakage detected in {ranking_column}: {int(invalid.sum())} rows use a "
                f"ranking date on or after the match date. Example: {example}"
            )
    return True


def check_no_target_columns_in_features(feature_cols: Iterable[str]) -> bool:
    """Fail if target goals are included as model features."""
    features = set(feature_cols)
    leaked_targets = sorted(features & TARGET_COLUMNS)
    if leaked_targets:
        raise ValueError(f"Target leakage detected in feature_cols: {leaked_targets}")
    return True


def check_no_placeholder_features(feature_cols: Iterable[str]) -> bool:
    """Fail if placeholder or obviously leaky proxy features are present."""
    suspicious = []
    for feature in feature_cols:
        normalized = str(feature).strip().lower()
        if any(pattern in normalized for pattern in PLACEHOLDER_FEATURE_PATTERNS):
            suspicious.append(feature)
    if suspicious:
        raise ValueError(f"Placeholder or leaky proxy features detected: {sorted(suspicious)}")
    return True


def check_no_missing_feature_values(feature_df: pd.DataFrame, feature_cols: Iterable[str]) -> bool:
    """Fail if feature values contain NaN, infinity, or non-numeric values."""
    feature_cols = list(feature_cols)
    require_columns(feature_df, feature_cols, "feature_df")
    try:
        values = feature_df[feature_cols].apply(pd.to_numeric, errors="raise")
    except Exception as error:
        raise ValueError(f"Feature values must be numeric: {error}") from error
    missing_counts = values.isna().sum()
    infinite_counts = pd.Series(np.isinf(values.to_numpy(dtype=float)).sum(axis=0), index=feature_cols)
    invalid_columns = sorted(
        column
        for column in feature_cols
        if missing_counts[column] > 0 or infinite_counts[column] > 0
    )
    if invalid_columns:
        details = {
            column: {"nan": int(missing_counts[column]), "infinite": int(infinite_counts[column])}
            for column in invalid_columns
        }
        raise ValueError(f"Missing or infinite feature values detected: {details}")
    return True


def run_all_leakage_checks(
    training_df: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    feature_df: pd.DataFrame,
    feature_cols: Iterable[str],
    check_ranking_dates: bool = True,
) -> dict[str, bool]:
    """Run all requested leakage checks and fail immediately on any issue."""
    results = {
        "no_future_matches_used": check_no_future_matches_used(training_df, cutoff_date),
        "no_target_columns_in_features": check_no_target_columns_in_features(feature_cols),
        "no_placeholder_features": check_no_placeholder_features(feature_cols),
        "no_missing_feature_values": check_no_missing_feature_values(feature_df, feature_cols),
    }
    if check_ranking_dates:
        results["ranking_dates_before_match"] = check_ranking_dates_before_match(feature_df)
    return results
