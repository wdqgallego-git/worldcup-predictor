"""Shared normalization and matching helpers for the 2022 odds shadow test."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import apply_team_mapping, load_team_name_mapping


HISTORICAL_ODDS_2022_COLUMNS = [
    "match_id", "backtest_year", "penka_phase", "date", "kickoff_time",
    "team_a", "team_b", "bookmaker", "odds_team_a", "odds_draw", "odds_team_b",
    "over_under_line", "odds_over", "odds_under", "spread", "market_type",
    "source_name", "source_url", "source_timestamp", "hours_before_kickoff",
    "value_type", "confidence", "notes",
]

DIRECT_ALIASES = {
    "date": ["date", "Date", "match_date", "starts", "date_start"],
    "kickoff_time": ["kickoff_time", "time", "Heure", "match_time"],
    "team_a": ["team_a", "home_team", "HomeTeam", "home_team_name", "Équipe Domicile", "team1"],
    "team_b": ["team_b", "away_team", "AwayTeam", "away_team_name", "Équipe Extérieur", "team2"],
    "bookmaker": ["bookmaker", "vendor", "book"],
    "odds_team_a": ["odds_team_a", "home_odds", "home_team_odd", "Cote 1"],
    "odds_draw": ["odds_draw", "draw_odds", "tie_odd", "Cote X"],
    "odds_team_b": ["odds_team_b", "away_odds", "away_team_odd", "Cote 2"],
    "over_under_line": ["over_under_line", "total_value"],
    "odds_over": ["odds_over", "total_over_odds"],
    "odds_under": ["odds_under", "total_under_odds"],
    "spread": ["spread", "spread_home_value"],
    "source_timestamp": ["source_timestamp", "logged_time", "date_created", "updated_at"],
}

WIDE_BOOKMAKERS = {
    "bet365": ("B365H", "B365D", "B365A"),
    "betway": ("BWH", "BWD", "BWA"),
    "interwetten": ("IWH", "IWD", "IWA"),
    "william_hill": ("WHH", "WHD", "WHA"),
    "vbet": ("VCH", "VCD", "VCA"),
}


def read_source_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix == ".json":
        return pd.read_json(source)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV, XLSX, XLS, or JSON.")


def parse_column_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    possible_path = Path(value)
    payload = possible_path.read_text(encoding="utf-8") if possible_path.exists() else value
    mapping = json.loads(payload)
    if not isinstance(mapping, dict):
        raise ValueError("--column-map-json must decode to an object mapping target columns to source columns.")
    return {str(key): str(source) for key, source in mapping.items()}


def resolve_column(frame: pd.DataFrame, target: str, explicit: dict[str, str]) -> str | None:
    if target in explicit:
        source = explicit[target]
        if source not in frame.columns:
            raise ValueError(f"Mapped source column {source!r} for {target!r} is missing.")
        return source
    return next((candidate for candidate in DIRECT_ALIASES.get(target, []) if candidate in frame.columns), None)


def odds_to_decimal(value: object, odds_format: str = "auto") -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).strip()
    selected = odds_format
    if selected == "auto":
        selected = "fractional" if "/" in text else "american" if text.startswith(("+", "-")) else "decimal"
    if selected == "fractional":
        decimal = 1.0 + float(Fraction(text))
    elif selected == "american":
        american = float(text)
        if american == 0:
            raise ValueError("American odds cannot be zero.")
        decimal = 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))
    elif selected == "decimal":
        decimal = float(text)
    else:
        raise ValueError(f"Unsupported odds format: {odds_format}")
    if not np.isfinite(decimal) or decimal <= 1.0:
        raise ValueError(f"Decimal odds must be finite and > 1.0; received {value!r}.")
    return float(decimal)


def load_2022_backtest(path: str | Path = "outputs/backtest_predictions.csv") -> pd.DataFrame:
    backtests = pd.read_csv(path)
    required = {"date", "team_a", "team_b", "backtest_year", "penka_phase"}
    missing = sorted(required - set(backtests.columns))
    if missing:
        raise ValueError(f"backtest_predictions.csv is missing: {missing}")
    rows = backtests[backtests["backtest_year"].eq(2022)].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="raise").dt.normalize()
    rows = rows.sort_values(["date", "team_a", "team_b"], kind="stable").reset_index(drop=True)
    rows["shadow_match_id"] = np.arange(1, len(rows) + 1)
    return rows


def assign_backtest_identity(rows: pd.DataFrame, backtests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach shadow match IDs and phases by date/team orientation."""
    odds = rows.copy()
    odds["date"] = pd.to_datetime(odds["date"], errors="raise").dt.normalize()
    direct = backtests[["shadow_match_id", "date", "team_a", "team_b", "penka_phase"]].rename(
        columns={"shadow_match_id": "backtest_match_id", "penka_phase": "backtest_phase"}
    )
    swapped = direct.rename(columns={"team_a": "team_b", "team_b": "team_a"})
    lookup = pd.concat([direct, swapped], ignore_index=True).drop_duplicates(["date", "team_a", "team_b"])
    matched = odds.merge(lookup, on=["date", "team_a", "team_b"], how="left")
    existing_ids = matched["match_id"] if "match_id" in matched else pd.Series(pd.NA, index=matched.index)
    matched["match_id"] = existing_ids.where(existing_ids.notna(), matched["backtest_match_id"])
    existing_phase = matched["penka_phase"] if "penka_phase" in matched else pd.Series(pd.NA, index=matched.index)
    has_existing_phase = existing_phase.notna() & existing_phase.astype(str).str.strip().ne("")
    matched["penka_phase"] = existing_phase.where(has_existing_phase, matched["backtest_phase"])
    unmatched = matched[matched["backtest_match_id"].isna()].copy()
    return matched.drop(columns=["backtest_match_id", "backtest_phase"], errors="ignore"), unmatched


def expand_source_rows(
    frame: pd.DataFrame,
    explicit_map: dict[str, str],
    odds_format: str,
    default_bookmaker: str,
) -> pd.DataFrame:
    """Return one row per match/bookmaker from direct or football-data-style wide odds."""
    resolved = {target: resolve_column(frame, target, explicit_map) for target in DIRECT_ALIASES}
    base_targets = ("date", "team_a", "team_b")
    missing = [target for target in base_targets if resolved[target] is None]
    if missing:
        raise ValueError(f"Input is missing match identity columns: {missing}. Available: {list(frame.columns)}")
    wide_available = {
        bookmaker: columns for bookmaker, columns in WIDE_BOOKMAKERS.items() if set(columns).issubset(frame.columns)
    }
    direct_available = all(resolved[target] is not None for target in ("odds_team_a", "odds_draw", "odds_team_b"))
    if not direct_available and not wide_available:
        raise ValueError("No complete match-level 1X2 odds columns were found. Model probabilities are not accepted as odds.")
    rows = []
    for _, source_row in frame.iterrows():
        base = {
            target: source_row[resolved[target]] if resolved[target] is not None else pd.NA
            for target in DIRECT_ALIASES
        }
        books = []
        if direct_available:
            books.append(
                (
                    str(base["bookmaker"]).strip() if pd.notna(base["bookmaker"]) and str(base["bookmaker"]).strip() else default_bookmaker,
                    base["odds_team_a"], base["odds_draw"], base["odds_team_b"],
                )
            )
        books.extend((bookmaker, source_row[home], source_row[draw], source_row[away]) for bookmaker, (home, draw, away) in wide_available.items())
        for bookmaker, home, draw, away in books:
            if any(pd.isna(value) or str(value).strip() == "" for value in (home, draw, away)):
                continue
            row = base.copy()
            row.update(
                {
                    "bookmaker": bookmaker,
                    "odds_team_a": odds_to_decimal(home, odds_format),
                    "odds_draw": odds_to_decimal(draw, odds_format),
                    "odds_team_b": odds_to_decimal(away, odds_format),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def normalize_import(
    frame: pd.DataFrame,
    year: int,
    source_name: str,
    source_url: str,
    explicit_map: dict[str, str],
    odds_format: str,
    value_type: str,
    confidence: str,
    backtests: pd.DataFrame,
) -> pd.DataFrame:
    default_bookmaker = f"kaggle_{re.sub(r'[^a-z0-9]+', '_', source_name.lower()).strip('_')}"
    expanded = expand_source_rows(frame, explicit_map, odds_format, default_bookmaker)
    expanded["date"] = pd.to_datetime(expanded["date"], errors="raise", dayfirst=False)
    expanded = expanded[expanded["date"].dt.year.eq(year)].copy()
    mapping = load_team_name_mapping()
    expanded = apply_team_mapping(expanded, ["team_a", "team_b"], mapping)
    expanded, _ = assign_backtest_identity(expanded, backtests)
    expanded = expanded[expanded["match_id"].notna()].copy()
    output = pd.DataFrame(index=expanded.index, columns=HISTORICAL_ODDS_2022_COLUMNS)
    for column in ("match_id", "penka_phase", "date", "kickoff_time", "team_a", "team_b", "bookmaker", "odds_team_a", "odds_draw", "odds_team_b", "over_under_line", "odds_over", "odds_under", "spread", "source_timestamp"):
        if column in expanded:
            output[column] = expanded[column]
    output["backtest_year"] = year
    output["market_type"] = "1x2"
    output["source_name"] = source_name
    output["source_url"] = source_url
    output["hours_before_kickoff"] = pd.NA
    output["value_type"] = value_type
    output["confidence"] = confidence
    output["notes"] = "Source timestamp/kickoff-relative timestamp unavailable unless supplied by the dataset; no timestamp was fabricated."
    output["date"] = pd.to_datetime(output["date"]).dt.date.astype(str)
    return output[HISTORICAL_ODDS_2022_COLUMNS].reset_index(drop=True)
