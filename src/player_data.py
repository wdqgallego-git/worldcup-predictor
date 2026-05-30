"""Load and prepare player inputs for award prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_loader import normalize_columns, standardize_team_name


PLAYERS_COLUMNS = [
    "team", "player", "position", "club", "confirmed_squad", "probability_in_squad",
    "projected_starter", "projected_minutes_share", "star_prior",
]
PLAYER_STATS_COLUMNS = [
    "player", "team", "club_minutes", "club_goals", "club_assists", "club_xg", "club_xa",
    "club_penalty_goals",
]
GOALKEEPER_STATS_COLUMNS = [
    "goalkeeper", "team", "minutes", "saves", "goals_conceded", "clean_sheets",
    "save_percentage", "goals_prevented", "reputation_prior",
]
PENALTY_TAKERS_COLUMNS = ["team", "player", "penalty_taker_rank", "penalty_share", "penalty_conversion_rate"]
INJURY_COLUMNS = ["team", "player", "attack_adjustment", "defense_adjustment", "minutes_adjustment", "notes"]
SQUAD_STATUS_COLUMNS = ["team", "player", "confirmed_squad", "probability_in_squad", "status_notes"]


def resolve_data_path(path: str | Path, sample_path: str | Path) -> Path:
    """Use the requested real file or a sample fallback."""
    real_path = Path(path)
    fallback_path = Path(sample_path)
    if real_path.exists():
        return real_path
    if fallback_path.exists():
        print(f"WARNING: {real_path} not found. Using sample fallback: {fallback_path}")
        return fallback_path
    raise FileNotFoundError(f"Neither {real_path} nor sample fallback {fallback_path} exists.")


def read_player_csv(path: str | Path, sample_path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Read a player CSV with normalized columns."""
    selected_path = resolve_data_path(path, sample_path)
    try:
        return normalize_columns(pd.read_csv(selected_path))
    except Exception as error:
        raise ValueError(f"Failed to load {dataset_name} from {selected_path}: {error}") from error


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    """Raise a clear error when canonical columns are missing."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def normalize_player_name(name: object) -> object:
    """Normalize a player display name while preserving missing values."""
    if pd.isna(name):
        return name
    return " ".join(str(name).strip().split())


def normalize_team_names(df: pd.DataFrame, columns: Iterable[str] = ("team",), mapping=None) -> pd.DataFrame:
    """Normalize team whitespace and optionally apply team mappings."""
    normalized = df.copy()
    mapping = mapping or {}
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(lambda name: standardize_team_name(name, mapping))
    return normalized


def coerce_boolean(series: pd.Series, default: bool = False) -> pd.Series:
    """Convert common CSV boolean values into bool."""
    if series.empty:
        return series.astype(bool)
    truthy = {"true", "1", "yes", "y", "confirmed", "likely"}
    falsy = {"false", "0", "no", "n", "out", "unlikely"}

    def convert(value):
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return default

    return series.map(convert).astype(bool)


def numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column with a stable default."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def load_players(
    path: str | Path = "data/raw/players.csv",
    sample_path: str | Path = "data/raw/players_sample.csv",
    team_mapping=None,
) -> pd.DataFrame:
    """Load canonical player metadata with sample-schema compatibility."""
    players = read_player_csv(path, sample_path, "players")
    players = players.rename(columns={"player_name": "player", "expected_starter": "projected_starter"})
    if "player" not in players.columns:
        raise ValueError("players is missing required player/player_name column.")
    players["player"] = players["player"].map(normalize_player_name)
    players = normalize_team_names(players, mapping=team_mapping)
    players["club"] = players.get("club", "")
    players["confirmed_squad"] = coerce_boolean(players.get("confirmed_squad", pd.Series(False, index=players.index)))
    players["probability_in_squad"] = numeric_column(players, "probability_in_squad", 0.75).clip(0, 1)
    players["projected_starter"] = coerce_boolean(players.get("projected_starter", pd.Series(False, index=players.index)))
    players["projected_minutes_share"] = numeric_column(players, "projected_minutes_share", 0.65).clip(0, 1)
    players["star_prior"] = numeric_column(players, "star_prior", 0.0)
    validate_columns(players, PLAYERS_COLUMNS, "players")
    return players


def load_player_stats(
    path: str | Path = "data/raw/player_stats.csv",
    sample_path: str | Path = "data/raw/player_stats_sample.csv",
    players: pd.DataFrame | None = None,
    team_mapping=None,
) -> pd.DataFrame:
    """Load outfield player statistics with sample-schema compatibility."""
    stats = read_player_csv(path, sample_path, "player stats")
    stats = stats.rename(
        columns={
            "minutes": "club_minutes", "goals": "club_goals", "assists": "club_assists",
            "non_penalty_xg": "club_xg", "penalty_goals": "club_penalty_goals",
        }
    )
    if "player" not in stats.columns and "player_id" in stats.columns:
        if players is None or "player_id" not in players.columns:
            raise ValueError("player stats uses player_id; players with player_id are required.")
        stats = stats.merge(players[["player_id", "player", "team"]], on="player_id", how="left", validate="many_to_one")
    stats["player"] = stats["player"].map(normalize_player_name)
    stats = normalize_team_names(stats, mapping=team_mapping)
    for column in ("club_minutes", "club_goals", "club_assists", "club_xg", "club_xa", "club_penalty_goals"):
        stats[column] = numeric_column(stats, column, 0.0)
    validate_columns(stats, PLAYER_STATS_COLUMNS, "player stats")
    return stats


def load_goalkeeper_stats(
    path: str | Path = "data/raw/goalkeeper_stats.csv",
    sample_path: str | Path = "data/raw/goalkeeper_stats_sample.csv",
    players: pd.DataFrame | None = None,
    team_mapping=None,
) -> pd.DataFrame:
    """Load goalkeeper statistics with sample-schema compatibility."""
    stats = read_player_csv(path, sample_path, "goalkeeper stats")
    stats = stats.rename(columns={"post_shot_xg_minus_goals": "goals_prevented"})
    if "goalkeeper" not in stats.columns and "player_id" in stats.columns:
        if players is None or "player_id" not in players.columns:
            raise ValueError("goalkeeper stats uses player_id; players with player_id are required.")
        stats = stats.merge(
            players[["player_id", "player", "team"]].rename(columns={"player": "goalkeeper"}),
            on="player_id",
            how="left",
            validate="many_to_one",
        )
    stats["goalkeeper"] = stats["goalkeeper"].map(normalize_player_name)
    stats = normalize_team_names(stats, mapping=team_mapping)
    for column in ("minutes", "saves", "goals_conceded", "clean_sheets", "goals_prevented", "reputation_prior"):
        stats[column] = numeric_column(stats, column, 0.0)
    if "save_percentage" not in stats.columns:
        shots_on_target = stats["saves"] + stats["goals_conceded"]
        stats["save_percentage"] = np.where(shots_on_target > 0, stats["saves"] / shots_on_target, 0.0)
    else:
        stats["save_percentage"] = numeric_column(stats, "save_percentage", 0.0)
    validate_columns(stats, GOALKEEPER_STATS_COLUMNS, "goalkeeper stats")
    return stats


def load_penalty_takers(
    path: str | Path = "data/raw/penalty_takers.csv",
    sample_path: str | Path = "data/raw/penalty_takers_sample.csv",
    players: pd.DataFrame | None = None,
    team_mapping=None,
) -> pd.DataFrame:
    """Load penalty-taker priors with sample-schema compatibility."""
    takers = read_player_csv(path, sample_path, "penalty takers")
    if "player" not in takers.columns and "primary_taker" in takers.columns:
        takers = takers.rename(columns={"primary_taker": "player"})
        takers["penalty_taker_rank"] = 1
        takers["penalty_share"] = 1.0
        takers["penalty_conversion_rate"] = np.nan
    if players is not None and "player_id" in players.columns:
        player_by_id = players.set_index("player_id")["player"].to_dict()
        takers["player"] = takers["player"].map(lambda player: player_by_id.get(player, player))
    takers["player"] = takers["player"].map(normalize_player_name)
    takers = normalize_team_names(takers, mapping=team_mapping)
    takers["penalty_taker_rank"] = numeric_column(takers, "penalty_taker_rank", 1.0)
    takers["penalty_share"] = numeric_column(takers, "penalty_share", 0.0).clip(0, 1)
    takers["penalty_conversion_rate"] = numeric_column(takers, "penalty_conversion_rate", np.nan)
    validate_columns(takers, PENALTY_TAKERS_COLUMNS, "penalty takers")
    return takers


def load_injury_adjustments(
    path: str | Path = "data/raw/injury_adjustments.csv",
    sample_path: str | Path = "data/raw/injury_adjustments_sample.csv",
    team_mapping=None,
) -> pd.DataFrame:
    """Load late player injury and minutes adjustments."""
    adjustments = read_player_csv(path, sample_path, "injury adjustments")
    adjustments = normalize_team_names(adjustments, mapping=team_mapping)
    if "player" in adjustments.columns:
        adjustments["player"] = adjustments["player"].map(normalize_player_name)
    for column in ("attack_adjustment", "defense_adjustment", "minutes_adjustment"):
        adjustments[column] = numeric_column(adjustments, column, 0.0)
    adjustments["notes"] = adjustments.get("notes", "")
    validate_columns(adjustments, INJURY_COLUMNS, "injury adjustments")
    return adjustments


def load_squad_status(
    path: str | Path = "data/raw/squad_status.csv",
    sample_path: str | Path = "data/raw/squad_status_sample.csv",
    team_mapping=None,
) -> pd.DataFrame:
    """Load final squad status overrides."""
    status = read_player_csv(path, sample_path, "squad status")
    status = normalize_team_names(status, mapping=team_mapping)
    if "player" in status.columns:
        status["player"] = status["player"].map(normalize_player_name)
    status["confirmed_squad"] = coerce_boolean(status.get("confirmed_squad", pd.Series(False, index=status.index)))
    status["probability_in_squad"] = numeric_column(status, "probability_in_squad", 0.0).clip(0, 1)
    status["status_notes"] = status.get("status_notes", "")
    validate_columns(status, SQUAD_STATUS_COLUMNS, "squad status")
    return status


def apply_squad_status(player_features: pd.DataFrame, squad_status: pd.DataFrame | None) -> pd.DataFrame:
    """Apply final squad-status values when available."""
    if squad_status is None or squad_status.empty:
        return player_features.copy()
    validate_columns(squad_status, SQUAD_STATUS_COLUMNS, "squad status")
    overrides = squad_status.drop_duplicates(["team", "player"], keep="last")
    merged = player_features.merge(
        overrides[["team", "player", "confirmed_squad", "probability_in_squad", "status_notes"]],
        on=["team", "player"],
        how="left",
        suffixes=("", "_status"),
    )
    for column in ("confirmed_squad", "probability_in_squad"):
        override = f"{column}_status"
        merged[column] = merged[override].combine_first(merged[column])
        merged = merged.drop(columns=override)
    return merged


def apply_injury_adjustments(player_features: pd.DataFrame, injury_adjustments: pd.DataFrame | None) -> pd.DataFrame:
    """Apply late injury and minutes adjustments after base features are built."""
    adjusted = player_features.copy()
    for column in ("attack_adjustment", "defense_adjustment", "minutes_adjustment"):
        adjusted[column] = 0.0
    if injury_adjustments is not None and not injury_adjustments.empty:
        validate_columns(injury_adjustments, INJURY_COLUMNS, "injury adjustments")
        latest = injury_adjustments.drop_duplicates(["team", "player"], keep="last")
        adjusted = adjusted.drop(columns=["attack_adjustment", "defense_adjustment", "minutes_adjustment"]).merge(
            latest[["team", "player", "attack_adjustment", "defense_adjustment", "minutes_adjustment", "notes"]],
            on=["team", "player"],
            how="left",
        )
        for column in ("attack_adjustment", "defense_adjustment", "minutes_adjustment"):
            adjusted[column] = numeric_column(adjusted, column, 0.0)
    adjusted["adjusted_projected_minutes_share"] = (
        adjusted["projected_minutes_share"] + adjusted["minutes_adjustment"]
    ).clip(0, 1)
    return adjusted


def build_player_feature_table(
    players: pd.DataFrame | None = None,
    player_stats: pd.DataFrame | None = None,
    penalty_takers: pd.DataFrame | None = None,
    squad_status: pd.DataFrame | None = None,
    injury_adjustments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build player award features, applying late status and injury overrides last."""
    players = load_players() if players is None else players.copy()
    player_stats = load_player_stats(players=players) if player_stats is None else player_stats.copy()
    penalty_takers = load_penalty_takers(players=players) if penalty_takers is None else penalty_takers.copy()
    features = players.merge(player_stats, on=["team", "player"], how="left")
    features = features.merge(
        penalty_takers[PENALTY_TAKERS_COLUMNS],
        on=["team", "player"],
        how="left",
    )
    for column in ("club_minutes", "club_goals", "club_assists", "club_xg", "club_xa", "club_penalty_goals"):
        features[column] = numeric_column(features, column, 0.0)
    safe_minutes = features["club_minutes"].replace(0, np.nan)
    features["goals_per_90"] = (90 * features["club_goals"] / safe_minutes).fillna(0.0)
    features["xg_per_90"] = (90 * features["club_xg"] / safe_minutes).fillna(0.0)
    features["assists_per_90"] = (90 * features["club_assists"] / safe_minutes).fillna(0.0)
    features["xa_per_90"] = (90 * features["club_xa"] / safe_minutes).fillna(0.0)
    features = apply_squad_status(features, squad_status)
    return apply_injury_adjustments(features, injury_adjustments)
