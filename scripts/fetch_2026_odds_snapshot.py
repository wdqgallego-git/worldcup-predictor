"""Fetch a current BALLDONTLIE 2026 odds snapshot and refresh advisory artifacts.

The market blend remains gated. This script updates current market inputs and
candidate-only reports without retraining models or changing baseline picks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BALLDONTLIE_API_KEY_ENV, MARKET_BLEND_WEIGHT, SCORE_PMF_METHOD  # noqa: E402
from data_loader import apply_team_mapping, load_team_name_mapping  # noqa: E402
from scripts.fetch_odds_historical import (  # noqa: E402
    DEFAULT_BASE_URL,
    REQUEST_PAUSE_SECONDS,
    ConservativeRateLimiter,
    american_to_decimal,
    first_present,
    get_paginated,
)
from market_blend import apply_market_prior, load_market_consensus, parse_kickoff_utc  # noqa: E402
from odds_priors import (  # noqa: E402
    FUTURES_MARKETS,
    FUTURES_ODDS_COLUMNS,
    MATCH_ODDS_COLUMNS,
    load_market_blend_gate,
    run_optional_odds_diagnostics,
)
from pick_sheet import build_pick_sheet, write_pick_sheet  # noqa: E402


FIXTURES_PATH = PROJECT_ROOT / "data" / "raw" / "fixtures.csv"
MATCH_ODDS_PATH = PROJECT_ROOT / "data" / "raw" / "match_odds.csv"
FUTURES_ODDS_PATH = PROJECT_ROOT / "data" / "raw" / "futures_odds.csv"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "raw" / "odds_snapshots"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEASON = 2026

RAW_ARCHIVE_COLUMNS = [
    "snapshot_run_utc",
    "api_odds_id",
    "api_match_id",
    "vendor",
    "moneyline_home_odds",
    "moneyline_draw_odds",
    "moneyline_away_odds",
    "spread_home_value",
    "spread_home_odds",
    "spread_away_value",
    "spread_away_odds",
    "total_value",
    "total_over_odds",
    "total_under_odds",
    "source_timestamp",
    "local_match_id",
    "fixture_orientation",
    "expanded_market_types",
    "markets_json",
    "processing_status",
    "rejection_reason",
]

FUTURES_MARKET_MAPPING = {
    "outright": "champion",
    "champion": "champion",
    "winner": "champion",
    "runner_up": "runner_up",
    "top_scorer": "top_scorer",
    "golden_boot": "golden_boot",
}

PLACEHOLDER_MARKERS = ("tbd", "placeholder", "winner of", "runner-up", "third-place", "3rd place")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current 2026 match odds and tournament futures.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--request-pause-seconds", type=float, default=REQUEST_PAUSE_SECONDS)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def _api_key() -> str:
    key = os.environ.get(BALLDONTLIE_API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"BLOCKED: {BALLDONTLIE_API_KEY_ENV} not set (environment or git-ignored root .env).")
    return key


def _is_placeholder(value: object) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return not text or any(marker in text for marker in PLACEHOLDER_MARKERS)


def _load_fixtures(mapping: dict[str, str]) -> pd.DataFrame:
    fixtures = pd.read_csv(FIXTURES_PATH)
    required = {"match_id", "stage", "date", "time", "team_a", "team_b"}
    missing = sorted(required - set(fixtures.columns))
    if missing:
        raise ValueError(f"fixtures.csv is missing required columns: {missing}")
    fixtures = apply_team_mapping(fixtures, ["team_a", "team_b"], mapping)
    fixtures["match_id_key"] = fixtures["match_id"].astype(str)
    if fixtures["match_id_key"].duplicated().any():
        raise ValueError("fixtures.csv contains duplicate match_id values.")
    return fixtures.set_index("match_id_key", drop=False)


def _expanded_market_types(entry: dict) -> set[str]:
    return {
        str(market.get("type", "")).strip().lower()
        for market in entry.get("markets", [])
        if isinstance(market, dict) and market.get("type")
    }


def _team_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("name", "")).strip()
    return str(value or "").strip()


def _build_api_fixture_map(
    match_entries: list[dict],
    fixtures: pd.DataFrame,
    mapping: dict[str, str],
    rejection_counts: Counter,
) -> dict[str, dict[str, object]]:
    """Map BALLDONTLIE match IDs to local fixtures by canonical teams and UTC kickoff."""
    local_by_key: dict[tuple[str, str, pd.Timestamp], pd.Series] = {}
    for _, fixture in fixtures.iterrows():
        if _is_placeholder(fixture["team_a"]) or _is_placeholder(fixture["team_b"]):
            continue
        kickoff = parse_kickoff_utc(fixture["date"], fixture["time"])
        local_by_key[(str(fixture["team_a"]), str(fixture["team_b"]), kickoff)] = fixture

    api_map: dict[str, dict[str, object]] = {}
    for entry in match_entries:
        api_match_id = entry.get("id")
        team_a = mapping.get(_team_name(entry.get("home_team")), _team_name(entry.get("home_team")))
        team_b = mapping.get(_team_name(entry.get("away_team")), _team_name(entry.get("away_team")))
        kickoff_value = entry.get("datetime")
        if api_match_id is None or not team_a or not team_b or not kickoff_value:
            rejection_counts["api_match_missing_resolved_team_or_kickoff"] += 1
            continue
        try:
            kickoff = pd.Timestamp(kickoff_value)
            kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
        except (TypeError, ValueError):
            rejection_counts["api_match_invalid_kickoff"] += 1
            continue
        fixture = local_by_key.get((team_a, team_b, kickoff))
        orientation = "same"
        if fixture is None:
            fixture = local_by_key.get((team_b, team_a, kickoff))
            orientation = "swapped"
        if fixture is None:
            rejection_counts["api_match_not_matched_by_teams_and_kickoff"] += 1
            continue
        api_map[str(api_match_id)] = {"fixture": fixture, "orientation": orientation}
    return api_map


def _raw_archive_row(entry: dict, run_utc: pd.Timestamp) -> dict[str, object]:
    market_types = sorted(_expanded_market_types(entry))
    return {
        "snapshot_run_utc": run_utc.isoformat(),
        "api_odds_id": entry.get("id"),
        "api_match_id": first_present(entry, "match_id", "game_id"),
        "vendor": first_present(entry, "vendor", "bookmaker", "book"),
        "moneyline_home_odds": entry.get("moneyline_home_odds"),
        "moneyline_draw_odds": entry.get("moneyline_draw_odds"),
        "moneyline_away_odds": entry.get("moneyline_away_odds"),
        "spread_home_value": entry.get("spread_home_value"),
        "spread_home_odds": entry.get("spread_home_odds"),
        "spread_away_value": entry.get("spread_away_value"),
        "spread_away_odds": entry.get("spread_away_odds"),
        "total_value": entry.get("total_value"),
        "total_over_odds": entry.get("total_over_odds"),
        "total_under_odds": entry.get("total_under_odds"),
        "source_timestamp": entry.get("updated_at"),
        "local_match_id": "",
        "fixture_orientation": "",
        "expanded_market_types": ",".join(market_types),
        "markets_json": json.dumps(entry.get("markets", []), ensure_ascii=True, separators=(",", ":")),
        "processing_status": "pending",
        "rejection_reason": "",
    }


def _optional_decimal(value: object) -> float | None:
    if value in (None, ""):
        return None
    return american_to_decimal(value)


def _build_match_odds_row(
    entry: dict,
    api_fixture_map: dict[str, dict[str, object]],
) -> tuple[dict[str, object] | None, str | None]:
    api_match_id = first_present(entry, "match_id", "game_id")
    if api_match_id is None:
        return None, "missing_api_match_id"
    key = str(api_match_id)
    if key not in api_fixture_map:
        return None, "api_match_not_in_resolved_fixture_map"
    fixture = api_fixture_map[key]["fixture"]
    orientation = str(api_fixture_map[key]["orientation"])
    source_timestamp = entry.get("updated_at")
    if not source_timestamp:
        return None, "missing_source_timestamp"
    try:
        snapshot_utc = pd.Timestamp(source_timestamp)
        snapshot_utc = snapshot_utc.tz_localize("UTC") if snapshot_utc.tzinfo is None else snapshot_utc.tz_convert("UTC")
        kickoff_utc = parse_kickoff_utc(fixture["date"], fixture["time"])
    except (TypeError, ValueError) as exc:
        return None, f"invalid_timestamp_or_kickoff:{type(exc).__name__}"
    if snapshot_utc >= kickoff_utc:
        return None, "post_kickoff_snapshot"
    api_moneyline = [
        entry.get("moneyline_home_odds"),
        entry.get("moneyline_draw_odds"),
        entry.get("moneyline_away_odds"),
    ]
    if any(value in (None, "") for value in api_moneyline):
        return None, "missing_complete_1x2"
    try:
        home_odds, odds_draw, away_odds = [american_to_decimal(value) for value in api_moneyline]
    except (TypeError, ValueError):
        return None, "invalid_1x2_odds"
    odds_a, odds_b = (home_odds, away_odds) if orientation == "same" else (away_odds, home_odds)
    total_value = entry.get("total_value")
    total_over = entry.get("total_over_odds")
    total_under = entry.get("total_under_odds")
    if not all(value not in (None, "") for value in (total_value, total_over, total_under)):
        total_value = total_over = total_under = None
    try:
        odds_over = _optional_decimal(total_over)
        odds_under = _optional_decimal(total_under)
        spread_value = (
            entry.get("spread_home_value")
            if orientation == "same"
            else entry.get("spread_away_value")
        )
        spread = float(spread_value) if spread_value not in (None, "") else None
    except (TypeError, ValueError):
        odds_over = odds_under = spread = None
        total_value = None
    hours_before = float((kickoff_utc - snapshot_utc) / pd.Timedelta(hours=1))
    row = {column: pd.NA for column in MATCH_ODDS_COLUMNS}
    row.update(
        {
            "match_id": fixture["match_id"],
            "date": pd.Timestamp(fixture["date"]).date().isoformat(),
            "team_a": fixture["team_a"],
            "team_b": fixture["team_b"],
            "bookmaker": str(first_present(entry, "vendor", "bookmaker", "book") or "balldontlie"),
            "odds_team_a": odds_a,
            "odds_draw": odds_draw,
            "odds_team_b": odds_b,
            "market_type": "1x2",
            "source_timestamp": snapshot_utc.isoformat(),
            "over_under_line": float(total_value) if total_value is not None else pd.NA,
            "odds_over": odds_over if odds_over is not None else pd.NA,
            "odds_under": odds_under if odds_under is not None else pd.NA,
            "spread": spread if spread is not None else pd.NA,
            "hours_before_kickoff": round(hours_before, 2),
        }
    )
    return row, None


def _merge_latest(
    path: Path,
    new_rows: pd.DataFrame,
    columns: list[str],
    keys: list[str],
    replace_bookmakers: set[str] | None = None,
) -> pd.DataFrame:
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in existing.columns:
            existing[column] = pd.NA
        if column not in new_rows.columns:
            new_rows[column] = pd.NA
    if replace_bookmakers and "bookmaker" in existing.columns:
        normalized = {bookmaker.strip().lower() for bookmaker in replace_bookmakers}
        existing = existing[~existing["bookmaker"].astype(str).str.strip().str.lower().isin(normalized)]
    merged = pd.concat([existing[columns], new_rows[columns]], ignore_index=True)
    if merged.empty:
        merged.to_csv(path, index=False)
        return merged
    merged["_timestamp_sort"] = pd.to_datetime(merged["source_timestamp"], errors="coerce", utc=True)
    merged = merged.sort_values("_timestamp_sort", na_position="first", kind="stable")
    merged = merged.drop_duplicates(subset=keys, keep="last").drop(columns="_timestamp_sort")
    merged.to_csv(path, index=False)
    return merged


def _build_futures_rows(
    entries: list[dict], mapping: dict[str, str], rejection_counts: Counter
) -> pd.DataFrame:
    rows = []
    for entry in entries:
        api_market = str(entry.get("market_type", "")).strip().lower()
        market = FUTURES_MARKET_MAPPING.get(api_market)
        if market not in FUTURES_MARKETS:
            rejection_counts[f"unsupported_futures_market:{api_market or 'missing'}"] += 1
            continue
        subject = entry.get("subject") if isinstance(entry.get("subject"), dict) else {}
        candidate = str(subject.get("name", "")).strip()
        if not candidate:
            rejection_counts["missing_futures_subject"] += 1
            continue
        if market in {"champion", "runner_up"}:
            candidate = mapping.get(candidate, candidate)
        try:
            decimal_odds = float(entry.get("decimal_odds"))
            if decimal_odds <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            try:
                decimal_odds = american_to_decimal(entry.get("american_odds"))
            except (TypeError, ValueError):
                rejection_counts["invalid_futures_odds"] += 1
                continue
        source_timestamp = entry.get("updated_at")
        if not source_timestamp:
            rejection_counts["missing_futures_timestamp"] += 1
            continue
        rows.append(
            {
                "team": candidate,
                "bookmaker": str(entry.get("vendor") or "balldontlie"),
                "market": market,
                "decimal_odds": decimal_odds,
                "source_timestamp": source_timestamp,
            }
        )
    return pd.DataFrame(rows, columns=FUTURES_ODDS_COLUMNS)


def _prediction_values_checksum(path: Path) -> str:
    table = pd.read_csv(path)
    values = table[["match_id", "recommended_prediction"]].astype(str).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(values).hexdigest()


def _market_types_available(entries: list[dict]) -> list[str]:
    types = set()
    for entry in entries:
        if all(entry.get(column) not in (None, "") for column in ("moneyline_home_odds", "moneyline_draw_odds", "moneyline_away_odds")):
            types.add("1X2")
        expanded = _expanded_market_types(entry)
        if "total" in expanded or all(entry.get(column) not in (None, "") for column in ("total_value", "total_over_odds", "total_under_odds")):
            types.add("over/under")
        if "spread" in expanded or entry.get("spread_home_value") not in (None, ""):
            types.add("spread")
    return [label for label in ("1X2", "over/under", "spread") if label in types]


def _top_disagreement_lines(features: pd.DataFrame, limit: int = 5) -> list[str]:
    if features.empty:
        return ["  none (no fixture odds consensus matched saved predictions)"]
    table = features.copy()
    table["team_a_win_gap"] = table["prob_team_a"] - table["model_prob_team_a"]
    table["team_b_win_gap"] = table["prob_team_b"] - table["model_prob_team_b"]
    table["largest_win_gap"] = table[["team_a_win_gap", "team_b_win_gap"]].abs().max(axis=1)
    lines = []
    for row in table.nlargest(limit, "largest_win_gap").itertuples(index=False):
        if abs(row.team_a_win_gap) >= abs(row.team_b_win_gap):
            team, model, market, gap = row.team_a, row.model_prob_team_a, row.prob_team_a, row.team_a_win_gap
        else:
            team, model, market, gap = row.team_b, row.model_prob_team_b, row.prob_team_b, row.team_b_win_gap
        lines.append(
            f"  {row.team_a} vs {row.team_b}: {team} win model={model:.3f}, market={market:.3f}, market-model={gap:+.3f}"
        )
    return lines


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_utc = pd.Timestamp.now(tz="UTC")
    recommendations_path = output_dir / "final_match_recommendations.csv"
    if not recommendations_path.exists():
        raise FileNotFoundError(f"Saved recommendations are required: {recommendations_path}")
    recommended_before = _prediction_values_checksum(recommendations_path)

    mapping = load_team_name_mapping(PROJECT_ROOT / "data" / "raw" / "team_name_mapping.csv")
    fixtures = _load_fixtures(mapping)
    request_stats: Counter = Counter()
    limiter = ConservativeRateLimiter(args.request_pause_seconds)
    session = requests.Session()
    session.headers["Authorization"] = _api_key()

    print("Fetching current 2026 match odds...")
    odds_entries = get_paginated(
        session, f"{args.base_url}/odds", {"seasons[]": SEASON}, limiter, request_stats
    )
    print("Fetching 2026 match metadata for fixture-safe joining...")
    match_entries = get_paginated(
        session, f"{args.base_url}/matches", {"seasons[]": SEASON}, limiter, request_stats
    )
    print("Fetching current 2026 tournament futures...")
    futures_entries = get_paginated(
        session, f"{args.base_url}/odds/futures", {"seasons[]": SEASON}, limiter, request_stats
    )

    fixture_map_rejections: Counter = Counter()
    api_fixture_map = _build_api_fixture_map(match_entries, fixtures, mapping, fixture_map_rejections)
    raw_rows = [_raw_archive_row(entry, run_utc) for entry in odds_entries]
    accepted_rows = []
    rejection_counts: Counter = Counter()
    rejection_rows = []
    for raw_row, entry in zip(raw_rows, odds_entries):
        row, reason = _build_match_odds_row(entry, api_fixture_map)
        if reason:
            raw_row["processing_status"] = "rejected"
            raw_row["rejection_reason"] = reason
            rejection_counts[reason] += 1
            rejection_rows.append(raw_row.copy())
        else:
            raw_row["processing_status"] = "matched"
            api_match_id = str(first_present(entry, "match_id", "game_id"))
            raw_row["local_match_id"] = row["match_id"]
            raw_row["fixture_orientation"] = api_fixture_map[api_match_id]["orientation"]
            accepted_rows.append(row)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = run_utc.strftime("%Y%m%dT%H%M%S%fZ")
    archive_path = SNAPSHOT_DIR / f"odds_2026_{stamp}.csv"
    pd.DataFrame(raw_rows, columns=RAW_ARCHIVE_COLUMNS).to_csv(archive_path, index=False)
    rejection_path = output_dir / "odds_snapshot_2026_rejections.csv"
    pd.DataFrame(rejection_rows, columns=RAW_ARCHIVE_COLUMNS).to_csv(rejection_path, index=False)

    current_rows = pd.DataFrame(accepted_rows, columns=MATCH_ODDS_COLUMNS)
    if not current_rows.empty:
        current_rows["_timestamp_sort"] = pd.to_datetime(current_rows["source_timestamp"], utc=True)
        current_rows = (
            current_rows.sort_values("_timestamp_sort", kind="stable")
            .drop_duplicates(["match_id", "bookmaker"], keep="last")
            .drop(columns="_timestamp_sort")
        )
    MATCH_ODDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    api_match_vendors = {
        str(first_present(entry, "vendor", "bookmaker", "book") or "balldontlie")
        for entry in odds_entries
    }
    merged_match_odds = _merge_latest(
        MATCH_ODDS_PATH,
        current_rows,
        MATCH_ODDS_COLUMNS,
        ["match_id", "bookmaker"],
        replace_bookmakers=api_match_vendors,
    )

    futures_rejections: Counter = Counter()
    futures_rows = _build_futures_rows(futures_entries, mapping, futures_rejections)
    api_futures_vendors = {str(entry.get("vendor") or "balldontlie") for entry in futures_entries}
    merged_futures = _merge_latest(
        FUTURES_ODDS_PATH,
        futures_rows,
        FUTURES_ODDS_COLUMNS,
        ["team", "bookmaker", "market"],
        replace_bookmakers=api_futures_vendors,
    )

    diagnostics = run_optional_odds_diagnostics(
        output_dir=output_dir,
        match_odds_path=MATCH_ODDS_PATH,
        futures_odds_path=FUTURES_ODDS_PATH,
    )
    predictions_path = output_dir / "final_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    consensus = load_market_consensus(MATCH_ODDS_PATH, method=SCORE_PMF_METHOD)
    blended = apply_market_prior(
        predictions,
        consensus,
        market_weight=MARKET_BLEND_WEIGHT,
        method=SCORE_PMF_METHOD,
    )
    gate = load_market_blend_gate(output_dir / "market_blend_gate.json")
    pick_sheet = build_pick_sheet(blended, gate)
    write_pick_sheet(pick_sheet, output_dir)

    recommended_after = _prediction_values_checksum(recommendations_path)
    recommended_unchanged = recommended_before == recommended_after
    covered = int(blended["has_market_odds"].astype(bool).sum())
    total_group_fixtures = int(len(predictions))
    match_vendors = sorted(set(current_rows["bookmaker"].dropna().astype(str))) if not current_rows.empty else []
    futures_vendors = sorted(set(futures_rows["bookmaker"].dropna().astype(str))) if not futures_rows.empty else []
    vendors = sorted(set(match_vendors) | set(futures_vendors))
    market_types = _market_types_available(odds_entries)
    futures_markets = sorted(set(futures_rows["market"])) if not futures_rows.empty else []
    if not futures_rows.empty:
        market_types.append("futures")
    disagreement_lines = _top_disagreement_lines(diagnostics["market_prior_features"])
    candidate_rows = pick_sheet[pick_sheet["market_blended_prediction"].notna()]
    badge_rows = pick_sheet[
        pick_sheet["odds_provenance_badge"].notna()
        & ~pick_sheet["odds_provenance_badge"].astype(str).str.startswith("no odds")
    ]

    summary_lines = [
        "2026 BALLDONTLIE Odds Snapshot Summary",
        "======================================",
        f"snapshot_utc: {run_utc.isoformat()}",
        f"requests_made: {request_stats['requests_attempted']}",
        f"match_odds_rows_fetched: {len(odds_entries)}",
        f"match_odds_fixtures_covered: {covered} of {total_group_fixtures}",
        f"match_odds_rows_matched_pre_dedup: {len(accepted_rows)}",
        f"match_odds_rows_latest_snapshot: {len(current_rows)}",
        f"match_odds_rows_rejected: {sum(rejection_counts.values())}",
        "match_odds_rejections:",
        *([f"  {reason}: {count}" for reason, count in sorted(rejection_counts.items())] or ["  none"]),
        f"api_match_metadata_rows_fetched: {len(match_entries)}",
        f"api_matches_mapped_to_local_fixtures: {len(api_fixture_map)}",
        "api_match_mapping_rejections:",
        *([f"  {reason}: {count}" for reason, count in sorted(fixture_map_rejections.items())] or ["  none"]),
        f"bookmakers_vendors_represented: {', '.join(vendors) if vendors else 'none'}",
        f"match_odds_vendors: {', '.join(match_vendors) if match_vendors else 'none'}",
        f"futures_rows_fetched: {len(futures_entries)}",
        f"futures_rows_written_this_snapshot: {len(futures_rows)}",
        f"futures_rows_current_file: {len(merged_futures)}",
        f"futures_markets_covered: {', '.join(futures_markets) if futures_markets else 'none'}",
        f"futures_vendors: {', '.join(futures_vendors) if futures_vendors else 'none'}",
        f"market_types_available: {', '.join(market_types) if market_types else 'none'}",
        "futures_rows_skipped:",
        *([f"  {reason}: {count}" for reason, count in sorted(futures_rejections.items())] or ["  none"]),
        f"raw_archive: {archive_path}",
        f"merged_match_odds_rows_current_file: {len(merged_match_odds)}",
        f"market_blend_gate_promote: {bool(gate.get('promote', False))}",
        f"recommended_prediction_values_unchanged: {recommended_unchanged}",
        f"recommended_prediction_checksum_before: {recommended_before}",
        f"recommended_prediction_checksum_after: {recommended_after}",
        f"pick_sheet_market_blended_prediction_rows: {len(candidate_rows)}",
        f"pick_sheet_odds_provenance_badge_rows: {len(badge_rows)}",
        "top_5_model_vs_market_win_probability_disagreements:",
        *disagreement_lines,
        "",
        "2026 odds are advisory/live information only. They cannot validate promotion because outcomes are not known yet.",
        "Gate still BLOCKED: 2018/2022 historical odds must be sourced elsewhere before market blending can be promoted.",
    ]
    summary_path = output_dir / "odds_snapshot_2026_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))
    if not recommended_unchanged:
        raise AssertionError("Baseline recommended_prediction values changed while the market gate was not promoted.")


if __name__ == "__main__":
    main()
