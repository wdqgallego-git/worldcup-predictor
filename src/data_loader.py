"""Load, validate, and standardize raw project data."""

import csv
import re
from pathlib import Path

import pandas as pd

from config import DATA_RAW_DIR


PLACEHOLDER_TEAM_NAMES = {
    "",
    "placeholder",
    "tbd",
    "to be determined",
    "unknown",
}


def normalize_column_name(column: object) -> str:
    """Convert a source column name into a predictable snake_case name."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())
    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    normalized = df.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    return normalized


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...] | set[str],
    dataset_name: str,
) -> None:
    """Raise a clear error when required columns are missing."""
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        available_columns = sorted(df.columns.tolist())
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )


def read_csv(path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Read a CSV file and normalize its columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {csv_path}")
    try:
        return normalize_columns(pd.read_csv(csv_path, comment="#"))
    except Exception as error:
        raise ValueError(f"Failed to load {dataset_name} from {csv_path}: {error}") from error


def parse_date_column(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Parse a required date column without silently coercing invalid values."""
    validate_required_columns(df, ["date"], dataset_name)
    parsed = df.copy()
    try:
        parsed["date"] = pd.to_datetime(parsed["date"], errors="raise")
    except Exception as error:
        raise ValueError(f"{dataset_name} contains invalid dates: {error}") from error
    return parsed


def load_team_name_mapping(path: str | Path = "data/raw/team_name_mapping.csv") -> dict[str, str]:
    """Load source-to-standard team names."""
    mapping_df = read_csv(path, "team name mapping")
    validate_required_columns(mapping_df, ["source_name", "standard_name"], "team name mapping")
    return {
        str(source_name).strip(): str(standard_name).strip()
        for source_name, standard_name in zip(mapping_df["source_name"], mapping_df["standard_name"])
    }


def standardize_team_name(name: object, mapping: dict[str, str]) -> object:
    """Apply a team-name mapping while preserving missing values."""
    if pd.isna(name):
        return name
    original_name = str(name).strip()
    return mapping.get(original_name, original_name)


def apply_team_mapping(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Apply team-name mapping to the requested DataFrame columns."""
    validate_required_columns(df, columns, "team mapping input")
    mapped = df.copy()
    for column in columns:
        mapped[column] = mapped[column].map(lambda name: standardize_team_name(name, mapping))
    return mapped


def load_results(path: str | Path = "data/raw/results.csv") -> pd.DataFrame:
    """Load historical international match results."""
    results = read_csv(path, "results")
    validate_required_columns(
        results,
        ["date", "home_team", "away_team", "home_score", "away_score"],
        "results",
    )
    results = parse_date_column(results, "results")
    return apply_team_mapping(results, ["home_team", "away_team"], load_team_name_mapping())


def load_shootouts(path: str | Path = "data/raw/shootouts.csv") -> pd.DataFrame:
    """Load historical penalty shootout results."""
    shootouts = read_csv(path, "shootouts")
    validate_required_columns(shootouts, ["date", "home_team", "away_team", "winner"], "shootouts")
    shootouts = parse_date_column(shootouts, "shootouts")
    return apply_team_mapping(shootouts, ["home_team", "away_team", "winner"], load_team_name_mapping())


def load_rankings(path: str | Path = "data/raw/rankings.csv") -> pd.DataFrame:
    """Load historical FIFA rankings."""
    rankings = read_csv(path, "rankings")
    validate_required_columns(rankings, ["date", "team"], "rankings")
    rankings = parse_date_column(rankings, "rankings")
    return apply_team_mapping(rankings, ["team"], load_team_name_mapping())


def is_placeholder_team(name: object) -> bool:
    """Return whether a fixture team value is unresolved."""
    if pd.isna(name):
        return True
    normalized_name = str(name).strip().lower()
    return (
        normalized_name in PLACEHOLDER_TEAM_NAMES
        or normalized_name.startswith("winner ")
        or normalized_name.startswith("runner-up ")
        or normalized_name.startswith("loser ")
        or re.fullmatch(r"[12][a-l]", normalized_name) is not None
        or re.fullmatch(r"3[a-l](?:/[a-l])+", normalized_name) is not None
        or re.fullmatch(r"[wl]\d+", normalized_name) is not None
    )


def load_fixtures(path: str | Path = "data/raw/fixtures.csv") -> pd.DataFrame:
    """Load World Cup fixtures and clearly flag unresolved teams."""
    fixtures = read_csv(path, "fixtures")
    validate_required_columns(
        fixtures,
        ["match_id", "stage", "round", "group", "date", "time", "team_a", "team_b"],
        "fixtures",
    )
    fixtures = parse_date_column(fixtures, "fixtures")
    fixtures = apply_team_mapping(fixtures, ["team_a", "team_b"], load_team_name_mapping())
    fixtures["team_a_status"] = fixtures["team_a"].map(
        lambda name: "placeholder" if is_placeholder_team(name) else "available"
    )
    fixtures["team_b_status"] = fixtures["team_b"].map(
        lambda name: "placeholder" if is_placeholder_team(name) else "available"
    )
    fixtures["has_placeholder_team"] = (
        fixtures["team_a_status"].eq("placeholder") | fixtures["team_b_status"].eq("placeholder")
    )
    return fixtures


def load_raw_csv(filename: str, raw_dir: str | Path = DATA_RAW_DIR) -> list[dict[str, str]]:
    """Load a raw CSV as dictionaries for lightweight scripts."""
    with (Path(raw_dir) / filename).open(newline="", encoding="utf-8") as csv_file:
        rows = (line for line in csv_file if not line.lstrip().startswith("#"))
        return list(csv.DictReader(rows))


def load_company_scoring_rules() -> list[dict[str, str]]:
    """Load company-game award scoring rules."""
    return load_raw_csv("company_scoring_rules.csv")
