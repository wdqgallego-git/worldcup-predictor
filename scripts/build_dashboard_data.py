"""Build the mobile Penka dashboard data from existing prediction outputs."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
DASHBOARD = PROJECT_ROOT / "dashboard"


def _required_csv(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dashboard source is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def _optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _value(value, default=None):
    if pd.isna(value):
        return default
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        return default
    return value.item() if hasattr(value, "item") else value


def _probability(value) -> float | None:
    value = _value(value)
    return round(float(value) * 100, 1) if value is not None else None


def build_dashboard_payload() -> dict:
    recommendations = _required_csv(
        OUTPUTS / "final_match_recommendations.csv",
        {"match_id", "team_a", "team_b", "recommended_prediction", "recommended_strategy"},
    )
    predictions = _required_csv(
        OUTPUTS / "final_predictions.csv",
        {
            "match_id", "date", "time", "group", "p_team_a_win", "p_draw", "p_team_b_win",
            "has_market_odds", "market_bookmaker_count", "market_bookmakers",
            "market_odds_captured_at", "market_prob_team_a", "market_prob_draw", "market_prob_team_b",
        },
    )
    contender = _required_csv(
        OUTPUTS / "live_context_contender_2026.csv",
        {"match_id", "live_context_prediction", "updated_expected_penka_points"},
    )
    disagreements = _optional_csv(OUTPUTS / "model_vs_market_disagreements.csv")

    large_disagreement_ids: set[int] = set()
    if not disagreements.empty and {"match_id", "large_disagreement"}.issubset(disagreements.columns):
        flagged = disagreements[disagreements["large_disagreement"].astype(str).str.lower().eq("true")]
        large_disagreement_ids = set(pd.to_numeric(flagged["match_id"], errors="coerce").dropna().astype(int))

    merged = (
        recommendations.merge(predictions, on="match_id", how="left", suffixes=("", "_prediction"), validate="one_to_one")
        .merge(contender[["match_id", "live_context_prediction", "updated_expected_penka_points"]], on="match_id", how="left", validate="one_to_one")
        .sort_values(["date", "time", "match_id"], kind="stable")
    )
    if merged[["date", "time", "group"]].isna().any().any():
        raise ValueError("Dashboard merge left fixtures without date, time, or group.")

    matches = []
    for row in merged.itertuples(index=False):
        market_probs = [
            _probability(row.market_prob_team_a),
            _probability(row.market_prob_draw),
            _probability(row.market_prob_team_b),
        ]
        has_odds = bool(_value(row.has_market_odds, False)) and all(value is not None for value in market_probs)
        recommended_result = "draw"
        rec_a, rec_b = (int(value) for value in str(row.recommended_prediction).split("-"))
        if rec_a > rec_b:
            recommended_result = "team_a"
        elif rec_b > rec_a:
            recommended_result = "team_b"
        market_result = None
        if has_odds:
            market_result = ("team_a", "draw", "team_b")[market_probs.index(max(market_probs))]
        match_id = int(row.match_id)
        matches.append(
            {
                "matchId": match_id,
                "date": str(row.date),
                "time": str(row.time),
                "group": str(row.group).replace("Group ", ""),
                "teamA": str(row.team_a),
                "teamB": str(row.team_b),
                "recommended": str(row.recommended_prediction),
                "recommendedStrategy": str(row.recommended_strategy),
                "contender": str(row.live_context_prediction),
                "contenderExpectedPoints": round(float(row.updated_expected_penka_points), 3),
                "marketProbabilities": market_probs if has_odds else None,
                "bookmakerCount": int(_value(row.market_bookmaker_count, 0) or 0),
                "bookmakers": str(_value(row.market_bookmakers, "") or "").split(", ") if has_odds else [],
                "oddsCapturedAt": _value(row.market_odds_captured_at),
                "hasOdds": has_odds,
                "marketDisagrees": bool(
                    has_odds
                    and (market_result != recommended_result or match_id in large_disagreement_ids)
                ),
            }
        )

    comparison = _optional_csv(OUTPUTS / "live_context_contender_comparison.csv")
    favorite_wc = 1.798611
    contender_wc = None
    if not comparison.empty:
        favorite_rows = comparison[
            comparison["scope"].eq("wc_only_group") & comparison["strategy"].eq("favorite_2_1")
        ]
        contender_rows = comparison[
            comparison["scope"].eq("wc_only_group") & comparison["strategy"].eq("live_context_walk_forward")
        ]
        if not favorite_rows.empty:
            favorite_wc = float(favorite_rows.iloc[0]["average_penka_points"])
        if not contender_rows.empty:
            contender_wc = float(contender_rows.iloc[0]["average_penka_points"])

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "strategy": {
            "name": "favorite 2-1",
            "wcBacktestPoints": round(favorite_wc, 3),
            "contenderName": "live context",
            "contenderWcBacktestPoints": round(contender_wc, 3) if contender_wc is not None else None,
            "gatePromoted": False,
        },
        "oddsFreshHours": 6,
        "matches": matches,
    }


def main() -> None:
    DASHBOARD.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload()
    path = DASHBOARD / "data.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    covered = sum(match["hasOdds"] for match in payload["matches"])
    print(f"Dashboard data written: {path}")
    print(f"Matches: {len(payload['matches'])}; odds coverage: {covered}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Dashboard build failed: {exc}", file=sys.stderr)
        raise
