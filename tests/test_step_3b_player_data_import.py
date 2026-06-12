from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_step_3b_player_data import (  # noqa: E402
    EXPECTED_HEADERS,
    REQUIRED_CSV_FILES,
    copy_release_files,
    sha256_file,
    validate_release,
)
from qa_imported_player_data import qa_imported_data  # noqa: E402


def write_csv(path: Path, header: list[str], row: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerow(row)


def test_step_3b_import_and_qa_accepts_objective_release(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    imported_dir = tmp_path / "imported"
    release_dir.mkdir()

    rows = {
        "players.csv": [
            "Argentina",
            "Sample Player",
            "FW",
            "Sample Club",
            "",
            "",
            "",
            "",
            "",
            "soccerdata FBref",
            "https://fbref.com/en/",
            "2026-06-02T00:00:00+00:00",
            "observed",
            "medium",
            "soccerdata club-season evidence; not official World Cup squad confirmation",
        ],
        "player_stats.csv": [
            "Sample Player",
            "Argentina",
            "900",
            "5",
            "2",
            "4.2",
            "1.1",
            "1",
            "",
            "",
            "",
            "soccerdata FBref",
            "https://fbref.com/en/",
            "2026-06-02T00:00:00+00:00",
            "observed",
            "medium",
            "soccerdata club-season evidence; national-team fields are intentionally blank",
        ],
        "goalkeeper_stats.csv": [
            "Sample Keeper",
            "Argentina",
            "900",
            "30",
            "10",
            "4",
            "0.75",
            "",
            "",
            "",
            "soccerdata FBref",
            "https://fbref.com/en/",
            "2026-06-02T00:00:00+00:00",
            "observed",
            "medium",
            "soccerdata club-season goalkeeper evidence; advanced priors intentionally blank",
        ],
        "penalty_takers.csv": [
            "Argentina",
            "Sample Player",
            "",
            "",
            "0.8",
            "soccerdata FBref",
            "https://fbref.com/en/",
            "2026-06-02T00:00:00+00:00",
            "observed",
            "medium",
            "club penalty evidence only; not national-team penalty hierarchy",
        ],
    }

    for file_name in REQUIRED_CSV_FILES:
        write_csv(release_dir / file_name, EXPECTED_HEADERS[file_name], rows[file_name])

    manifest = {
        "csv_row_counts": {file_name: 1 for file_name in REQUIRED_CSV_FILES},
        "files_exported": [
            {
                "path": file_name,
                "row_count": 1,
                "sha256": sha256_file(release_dir / file_name),
                "type": "csv",
            }
            for file_name in REQUIRED_CSV_FILES
        ],
        "intentionally_blank_fields": [
            "confirmed_squad",
            "probability_in_squad",
            "projected_starter",
            "projected_minutes_share",
            "star_prior",
            "reputation_prior",
            "penalty_taker_rank",
            "penalty_share",
        ],
        "no_deep_research_used": True,
        "no_fabricated_data": True,
        "no_live_api_calls_in_qa": True,
        "no_scraping_in_qa": True,
        "penalty_file_semantics": "penalty_takers.csv contains club penalty evidence only",
    }
    (release_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _, import_errors = validate_release(release_dir, manifest)
    assert import_errors == []

    copy_release_files(release_dir, imported_dir)
    _, summary, qa_errors = qa_imported_data(imported_dir)
    assert qa_errors == []
    assert summary["manifest_qa_result"] == "PASS"
    assert summary["objective_evidence_usable_for_award_modeling"] is True
    assert summary["final_awards_submission_ready"] is False
