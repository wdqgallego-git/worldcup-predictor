"""Download raw data and write a reproducibility manifest.

Run from the repository root:

    python src/download_data.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DATA_RAW_DIR = Path("data/raw")
MANIFEST_PATH = DATA_RAW_DIR / "data_manifest.json"
FIXTURES_PLACEHOLDER_COLUMNS = [
    "match_id",
    "stage",
    "round",
    "group",
    "date",
    "time",
    "team_a",
    "team_b",
    "ground",
    "neutral",
    "source_status",
]

DOWNLOAD_SOURCES = [
    {
        "file_name": "results.csv",
        "source_url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "kind": "csv",
        "notes": "Historical international football results from martj42/international_results.",
    },
    {
        "file_name": "shootouts.csv",
        "source_url": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
        "kind": "csv",
        "notes": "Historical international penalty shootout results from martj42/international_results.",
    },
    {
        "file_name": "rankings.csv",
        "source_url": "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/ranking_fifa_historical.csv",
        "kind": "csv",
        "notes": "Historical FIFA rankings from Dato-Futbol/fifa-ranking.",
    },
    {
        "file_name": "fixtures.csv",
        "source_url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json",
        "kind": "fixtures_2026",
        "notes": "World Cup 2026 fixtures from openfootball/worldcup.json.",
    },
]


def ensure_raw_dir() -> None:
    """Create data/raw if it does not exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def sha256_hash(path: Path) -> str:
    """Return the SHA-256 hash for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_downloaded_file(path: Path) -> pd.DataFrame:
    """Read a downloaded CSV without altering the schema."""
    return pd.read_csv(path, comment="#")


def parse_2026_fixtures(response: requests.Response) -> pd.DataFrame:
    """Parse openfootball World Cup 2026 fixtures into the project schema."""
    data = response.json()
    matches = data.get("matches") if isinstance(data, dict) else None
    if not isinstance(matches, list):
        raise ValueError('World Cup 2026 fixtures JSON must contain a top-level "matches" list.')

    rows = []
    for match_id, match in enumerate(matches, start=1):
        group = match.get("group")
        rows.append(
            {
                "match_id": match_id,
                "stage": "group_stage" if group else "knockout",
                "round": match.get("round"),
                "group": group,
                "date": match.get("date"),
                "time": match.get("time"),
                "team_a": match.get("team1"),
                "team_b": match.get("team2"),
                "ground": match.get("ground"),
                "neutral": True,
                "source_status": "downloaded",
            }
        )

    return pd.DataFrame(rows, columns=FIXTURES_PLACEHOLDER_COLUMNS)


def create_fixtures_placeholder() -> pd.DataFrame:
    """Create a fixtures placeholder with the expected schema."""
    placeholder = {column: "" for column in FIXTURES_PLACEHOLDER_COLUMNS}
    placeholder["source_status"] = "placeholder"
    placeholder["neutral"] = True
    return pd.DataFrame([placeholder], columns=FIXTURES_PLACEHOLDER_COLUMNS)


def warn_and_continue(file_name: str, error: Exception) -> None:
    """Print a clear warning and continue only when a local file exists."""
    local_path = DATA_RAW_DIR / file_name
    if local_path.exists():
        print(f"WARNING: failed to download {file_name}: {error}. Continuing with existing local file.")
    else:
        print(f"WARNING: failed to download {file_name}: {error}. No local sample file is available.")


def response_to_dataframe(response: requests.Response, kind: str) -> pd.DataFrame:
    """Convert a response into a DataFrame without silently changing schemas."""
    if kind == "fixtures_2026":
        return parse_2026_fixtures(response)

    content_type = response.headers.get("content-type", "").lower()
    text = response.text.strip()

    if kind == "csv" or (kind == "auto" and "json" not in content_type and not text.startswith(("{", "["))):
        temp_path = DATA_RAW_DIR / "_download_temp.csv"
        temp_path.write_text(response.text, encoding="utf-8")
        try:
            return pd.read_csv(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    data = response.json()
    if isinstance(data, dict):
        for key in ("matches", "fixtures", "data", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("JSON fixture response is not a list and has no known list field.")

    return pd.json_normalize(data)


def download_one(source: dict[str, str]) -> dict[str, object] | None:
    """Download one source and return its manifest entry."""
    file_name = source["file_name"]
    target_path = DATA_RAW_DIR / file_name
    status_code = "request_failed"

    try:
        if file_name == "fixtures.csv":
            print(f"Fixtures URL: {source['source_url']}")
        response = requests.get(source["source_url"], timeout=30)
        status_code = response.status_code
        if file_name == "fixtures.csv":
            print(f"Fixtures HTTP status code: {status_code}")
            if status_code != 200:
                print(f"Fixtures response preview: {response.text[:300]}")
        response.raise_for_status()
        data = response_to_dataframe(response, source["kind"])

        target_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(target_path, index=False)

        return {
            "file_name": file_name,
            "source_url": source["source_url"],
            "download_timestamp": utc_timestamp(),
            "row_count": int(len(data)),
            "columns": list(data.columns),
            "sha256_hash": sha256_hash(target_path),
            "notes": source["notes"],
        }
    except Exception as error:
        warn_and_continue(file_name, error)
        if file_name == "fixtures.csv":
            data = create_fixtures_placeholder()
            data.to_csv(target_path, index=False)
            print("WARNING: created fixtures.csv placeholder. Replace it with downloaded 2026 fixtures before final predictions.")
            return {
                "file_name": file_name,
                "source_url": source["source_url"],
                "download_timestamp": utc_timestamp(),
                "row_count": int(len(data)),
                "columns": list(data.columns),
                "sha256_hash": sha256_hash(target_path),
                "notes": (
                    f"{source['notes']} Download failed for URL {source['source_url']} "
                    f"with HTTP status code {status_code}: {error}. "
                    "Placeholder schema created with source_status=placeholder."
                ),
            }
        if target_path.exists():
            data = read_downloaded_file(target_path)
            return {
                "file_name": file_name,
                "source_url": source["source_url"],
                "download_timestamp": utc_timestamp(),
                "row_count": int(len(data)),
                "columns": list(data.columns),
                "sha256_hash": sha256_hash(target_path),
                "notes": f"{source['notes']} Existing local file used after download failure.",
            }
        return None


def write_manifest(entries: list[dict[str, object]]) -> None:
    """Write the raw data manifest."""
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def download_all() -> list[dict[str, object]]:
    """Download all configured raw data files."""
    ensure_raw_dir()
    manifest_entries = []

    for source in DOWNLOAD_SOURCES:
        entry = download_one(source)
        if entry is not None:
            manifest_entries.append(entry)

    write_manifest(manifest_entries)
    print(f"Wrote manifest with {len(manifest_entries)} entries to {MANIFEST_PATH}")
    return manifest_entries


if __name__ == "__main__":
    download_all()
