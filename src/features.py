"""Leakage-safe feature engineering for match prediction."""

from bisect import bisect_left
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "team_a_recent_points_per_match_5",
    "team_b_recent_points_per_match_5",
    "team_a_recent_points_per_match_10",
    "team_b_recent_points_per_match_10",
    "team_a_recent_goals_scored_5",
    "team_b_recent_goals_scored_5",
    "team_a_recent_goals_conceded_5",
    "team_b_recent_goals_conceded_5",
    "team_a_recent_goal_difference_5",
    "team_b_recent_goal_difference_5",
    "form_diff_5",
    "attack_diff_5",
    "defense_diff_5",
    "neutral_int",
    "is_world_cup",
    "is_friendly",
    "rank_a",
    "rank_b",
    "rank_diff",
    "ranking_points_a",
    "ranking_points_b",
    "ranking_points_diff",
]

BASE_MATCH_COLS = ["team_a", "team_b", "goals_a", "goals_b", "date", "tournament", "neutral"]
FIXTURE_REQUIRED_COLS = ["team_a", "team_b", "date"]
RANKING_REQUIRED_COLS = ["team", "date", "total_points"]


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    """Raise a clear error if a feature input is missing required columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def prepare_match_table(results: pd.DataFrame) -> pd.DataFrame:
    """Convert historical home/away results into the model's team_a/team_b schema."""
    source = results.copy()
    if {"home_team", "away_team", "home_score", "away_score"}.issubset(source.columns):
        source = source.rename(
            columns={
                "home_team": "team_a",
                "away_team": "team_b",
                "home_score": "goals_a",
                "away_score": "goals_b",
            }
        )
    validate_columns(source, BASE_MATCH_COLS, "results")
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    source["goals_a"] = pd.to_numeric(source["goals_a"], errors="raise")
    source["goals_b"] = pd.to_numeric(source["goals_b"], errors="raise")
    return source.sort_values("date", kind="stable").reset_index(drop=True)


def build_team_history(results: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    """Create chronological team histories used for pre-match form features."""
    matches = prepare_match_table(results)
    history: dict[str, list[dict[str, object]]] = defaultdict(list)

    for match in matches.itertuples(index=False):
        goals_a = float(match.goals_a)
        goals_b = float(match.goals_b)
        points_a = 3.0 if goals_a > goals_b else 1.0 if goals_a == goals_b else 0.0
        points_b = 3.0 if goals_b > goals_a else 1.0 if goals_a == goals_b else 0.0
        history[match.team_a].append(
            {"date": match.date, "points": points_a, "goals_scored": goals_a, "goals_conceded": goals_b}
        )
        history[match.team_b].append(
            {"date": match.date, "points": points_b, "goals_scored": goals_b, "goals_conceded": goals_a}
        )

    return dict(history)


def recent_team_form(
    team: object,
    match_date: pd.Timestamp,
    team_history: dict[str, list[dict[str, object]]],
    window: int,
) -> dict[str, float]:
    """Summarize a team's last matches strictly before match_date."""
    matches = team_history.get(team, [])
    dates = [match["date"] for match in matches]
    cutoff = bisect_left(dates, match_date)
    recent_matches = matches[max(0, cutoff - window) : cutoff]

    if not recent_matches:
        return {"points_per_match": np.nan, "goals_scored": np.nan, "goals_conceded": np.nan}

    match_count = len(recent_matches)
    return {
        "points_per_match": sum(float(match["points"]) for match in recent_matches) / match_count,
        "goals_scored": sum(float(match["goals_scored"]) for match in recent_matches) / match_count,
        "goals_conceded": sum(float(match["goals_conceded"]) for match in recent_matches) / match_count,
    }


def add_recent_team_form_features(
    matches: pd.DataFrame,
    historical_results: pd.DataFrame,
) -> pd.DataFrame:
    """Add recent-form features using only matches before each row's date."""
    validate_columns(matches, ["team_a", "team_b", "date"], "matches")
    featured = matches.copy()
    featured["date"] = pd.to_datetime(featured["date"], errors="raise")
    team_history = build_team_history(historical_results)

    feature_rows = []
    for match in featured.itertuples(index=False):
        team_a_5 = recent_team_form(match.team_a, match.date, team_history, 5)
        team_b_5 = recent_team_form(match.team_b, match.date, team_history, 5)
        team_a_10 = recent_team_form(match.team_a, match.date, team_history, 10)
        team_b_10 = recent_team_form(match.team_b, match.date, team_history, 10)
        feature_rows.append(
            {
                "team_a_recent_points_per_match_5": team_a_5["points_per_match"],
                "team_b_recent_points_per_match_5": team_b_5["points_per_match"],
                "team_a_recent_points_per_match_10": team_a_10["points_per_match"],
                "team_b_recent_points_per_match_10": team_b_10["points_per_match"],
                "team_a_recent_goals_scored_5": team_a_5["goals_scored"],
                "team_b_recent_goals_scored_5": team_b_5["goals_scored"],
                "team_a_recent_goals_conceded_5": team_a_5["goals_conceded"],
                "team_b_recent_goals_conceded_5": team_b_5["goals_conceded"],
            }
        )

    form_features = pd.DataFrame(feature_rows, index=featured.index)
    featured = pd.concat([featured, form_features], axis=1)
    featured["team_a_recent_goal_difference_5"] = (
        featured["team_a_recent_goals_scored_5"] - featured["team_a_recent_goals_conceded_5"]
    )
    featured["team_b_recent_goal_difference_5"] = (
        featured["team_b_recent_goals_scored_5"] - featured["team_b_recent_goals_conceded_5"]
    )
    featured["form_diff_5"] = (
        featured["team_a_recent_points_per_match_5"] - featured["team_b_recent_points_per_match_5"]
    )
    featured["attack_diff_5"] = (
        featured["team_a_recent_goals_scored_5"] - featured["team_b_recent_goals_scored_5"]
    )
    featured["defense_diff_5"] = (
        featured["team_b_recent_goals_conceded_5"] - featured["team_a_recent_goals_conceded_5"]
    )
    return featured


def prepare_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Normalize ranking inputs and derive rank from points if needed."""
    prepared = rankings.copy()
    validate_columns(prepared, RANKING_REQUIRED_COLS, "rankings")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    prepared["total_points"] = pd.to_numeric(prepared["total_points"], errors="coerce")
    if "rank" not in prepared.columns:
        prepared["rank"] = prepared.groupby("date")["total_points"].rank(method="min", ascending=False)
    else:
        prepared["rank"] = pd.to_numeric(prepared["rank"], errors="coerce")
    return prepared.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def get_latest_ranking_before_date(
    team: object,
    match_date: object,
    rankings: pd.DataFrame | None,
) -> dict[str, float]:
    """Return the latest FIFA ranking strictly before the match date."""
    if rankings is None or rankings.empty or pd.isna(team):
        return {"rank": np.nan, "total_points": np.nan}
    prepared = rankings if {"rank", "total_points", "team", "date"}.issubset(rankings.columns) else prepare_rankings(rankings)
    date = pd.Timestamp(match_date)
    eligible = prepared[(prepared["team"] == team) & (prepared["date"] < date)]
    if eligible.empty:
        return {"rank": np.nan, "total_points": np.nan}
    latest = eligible.iloc[-1]
    return {"rank": float(latest["rank"]), "total_points": float(latest["total_points"])}


def build_ranking_history(rankings: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    """Create chronological per-team ranking histories."""
    history: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prepare_rankings(rankings).itertuples(index=False):
        history[row.team].append({"date": row.date, "rank": row.rank, "total_points": row.total_points})
    return dict(history)


def latest_indexed_ranking_before_date(
    team: object,
    match_date: pd.Timestamp,
    ranking_history: dict[str, list[dict[str, object]]],
) -> dict[str, float]:
    """Return a team's latest indexed ranking strictly before match_date."""
    rankings = ranking_history.get(team, [])
    dates = [ranking["date"] for ranking in rankings]
    cutoff = bisect_left(dates, match_date)
    if cutoff == 0:
        return {"rank": np.nan, "total_points": np.nan}
    latest = rankings[cutoff - 1]
    return {"rank": float(latest["rank"]), "total_points": float(latest["total_points"])}


def add_ranking_features(matches: pd.DataFrame, rankings: pd.DataFrame | None) -> pd.DataFrame:
    """Add ranking features using only rankings published before each match."""
    featured = matches.copy()
    if rankings is None or rankings.empty:
        for column in ("rank_a", "rank_b", "rank_diff", "ranking_points_a", "ranking_points_b", "ranking_points_diff"):
            featured[column] = np.nan
        return featured

    ranking_history = build_ranking_history(rankings)
    ranking_rows = []
    for match in featured.itertuples(index=False):
        rank_a = latest_indexed_ranking_before_date(match.team_a, match.date, ranking_history)
        rank_b = latest_indexed_ranking_before_date(match.team_b, match.date, ranking_history)
        ranking_rows.append(
            {
                "rank_a": rank_a["rank"],
                "rank_b": rank_b["rank"],
                "ranking_points_a": rank_a["total_points"],
                "ranking_points_b": rank_b["total_points"],
            }
        )

    ranking_features = pd.DataFrame(ranking_rows, index=featured.index)
    featured = pd.concat([featured, ranking_features], axis=1)
    featured["rank_diff"] = featured["rank_a"] - featured["rank_b"]
    featured["ranking_points_diff"] = featured["ranking_points_a"] - featured["ranking_points_b"]
    return featured


def add_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add transparent pre-match tournament context flags."""
    featured = matches.copy()
    if "neutral" not in featured.columns:
        featured["neutral"] = True
    if "tournament" not in featured.columns:
        featured["tournament"] = "FIFA World Cup"
    tournament = featured["tournament"].fillna("").astype(str).str.lower()
    featured["neutral_int"] = featured["neutral"].fillna(False).astype(bool).astype(int)
    featured["is_world_cup"] = tournament.str.contains("world cup", regex=False).astype(int)
    featured["is_friendly"] = tournament.str.contains("friendly", regex=False).astype(int)
    return featured


def build_training_table(
    results: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    start_date: str = "2000-01-01",
) -> tuple[pd.DataFrame, list[str]]:
    """Build a leakage-safe match training table and return its feature columns."""
    matches = prepare_match_table(results)
    matches = matches[matches["date"] >= pd.Timestamp(start_date)].copy()
    matches = add_recent_team_form_features(matches, results)
    matches = add_context_features(matches)
    matches = add_ranking_features(matches, rankings)
    return matches, FEATURE_COLS.copy()


def build_fixture_features(
    fixtures: pd.DataFrame,
    historical_results: pd.DataFrame,
    rankings: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build prediction features for fixtures using historical information only."""
    validate_columns(fixtures, FIXTURE_REQUIRED_COLS, "fixtures")
    featured = fixtures.copy()
    featured["date"] = pd.to_datetime(featured["date"], errors="raise")
    featured = add_recent_team_form_features(featured, historical_results)
    featured = add_context_features(featured)
    featured = add_ranking_features(featured, rankings)
    selected_features = feature_cols.copy() if feature_cols is not None else FEATURE_COLS.copy()
    validate_columns(featured, selected_features, "fixture features")
    return featured, selected_features
