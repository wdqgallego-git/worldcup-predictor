"""Three-request diagnostic: does BALLDONTLIE have ANY odds data, and for which seasons?

Distinguishes the two readings of the empty 2018/2022 fetch:
  (a) seasons[] is honored and historical odds simply do not exist, vs
  (b) the odds dataset is empty/inaccessible in general.
Never prints the API key. Stays far below 5 requests/minute.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import BALLDONTLIE_API_KEY_ENV  # noqa: E402  (also loads the root .env)

BASE_URL = os.environ.get("BALLDONTLIE_BASE_URL", "https://api.balldontlie.io/fifa/worldcup/v1")
PAUSE_SECONDS = 15


def redact(obj: object, depth: int = 0) -> object:
    """Return structure + sample values only; truncate long strings, cap nesting."""
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        return {key: redact(value, depth + 1) for key, value in list(obj.items())[:25]}
    if isinstance(obj, list):
        return [redact(value, depth + 1) for value in obj[:2]] + (["...more"] if len(obj) > 2 else [])
    if isinstance(obj, str) and len(obj) > 60:
        return obj[:60] + "..."
    return obj


def probe(session: requests.Session, label: str, path: str, params: dict) -> dict | None:
    response = session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
    print(f"\n=== {label} ===")
    print(f"GET /{path} params={params} -> HTTP {response.status_code}")
    if not response.ok:
        print(f"body (first 300 chars): {response.text[:300]}")
        return None
    payload = response.json()
    data = payload.get("data", [])
    print(f"top_level_keys: {sorted(payload.keys())}")
    print(f"data_rows: {len(data)}; meta: {payload.get('meta')}")
    if data:
        print("sample row (redacted):")
        print(json.dumps(redact(data[0]), indent=2, default=str))
    return payload


def main() -> None:
    key = os.environ.get(BALLDONTLIE_API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"BLOCKED: {BALLDONTLIE_API_KEY_ENV} not set (env or root .env).")
    session = requests.Session()
    session.headers["Authorization"] = key

    # 1. Do 2026 odds exist? (controls for "odds dataset empty in general")
    payload_2026 = probe(session, "odds for 2026", "odds", {"seasons[]": 2026, "per_page": 2})
    time.sleep(PAUSE_SECONDS)

    # 2. Grab one real 2018 match id.
    payload_matches = probe(session, "one 2018 match", "matches", {"seasons[]": 2018, "per_page": 1})
    match_id = None
    if payload_matches and payload_matches.get("data"):
        match_id = payload_matches["data"][0].get("id")
    time.sleep(PAUSE_SECONDS)

    # 3. Ask for odds on that specific 2018 match via match_ids[] (bypasses seasons[] entirely).
    payload_2018 = None
    if match_id is not None:
        payload_2018 = probe(session, f"odds for 2018 match {match_id}", "odds", {"match_ids[]": match_id})
    else:
        print("\nSkipped per-match 2018 odds probe: no 2018 match id available.")

    print("\n=== VERDICT ===")
    rows_2026 = len(payload_2026.get("data", [])) if payload_2026 else 0
    rows_2018 = len(payload_2018.get("data", [])) if payload_2018 else 0
    if rows_2026 > 0 and rows_2018 == 0:
        print("2026 odds EXIST but per-match 2018 odds are EMPTY ->")
        print("BALLDONTLIE did not backfill historical 2018/2022 odds; the fetch script is fine.")
        print("Use a different historical-odds source for the backtest gate.")
    elif rows_2026 == 0 and rows_2018 == 0:
        print("No odds rows for 2026 OR 2018 -> odds endpoint returns nothing on this plan/trial.")
        print("Likely a tier/entitlement issue (odds are GOAT-tier) even though HTTP 200 is returned.")
    elif rows_2018 > 0:
        print("Per-match 2018 odds EXIST -> seasons[] filtering was the problem;")
        print("update fetch_odds_historical.py to query odds via match_ids[] per match.")
    else:
        print("Inconclusive; inspect the responses above.")


if __name__ == "__main__":
    main()
