"""Load raw and processed project data."""

import csv
from pathlib import Path

from config import DATA_RAW_DIR


def load_raw_csv(filename: str, raw_dir: str | Path = DATA_RAW_DIR) -> list[dict[str, str]]:
    """Load a CSV file from data/raw."""
    with (Path(raw_dir) / filename).open(newline="", encoding="utf-8") as csv_file:
        rows = (line for line in csv_file if not line.lstrip().startswith("#"))
        return list(csv.DictReader(rows))


def load_company_scoring_rules() -> list[dict[str, str]]:
    """Load company-game award scoring rules."""
    return load_raw_csv("company_scoring_rules.csv")
