"""Load raw and processed project data."""

import csv
from pathlib import Path

from config import RAW_DATA_DIR


def load_raw_csv(filename: str, raw_dir: Path = RAW_DATA_DIR) -> list[dict[str, str]]:
    """Load a CSV file from data/raw."""
    with (raw_dir / filename).open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_company_scoring_rules() -> list[dict[str, str]]:
    """Load company-game award scoring rules."""
    return load_raw_csv("company_scoring_rules.csv")
