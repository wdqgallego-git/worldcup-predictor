"""Parse locally saved public odds archive HTML; this script performs no scraping."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_odds_2022 import HISTORICAL_ODDS_2022_COLUMNS, load_2022_backtest  # noqa: E402


class TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self.canonical_url = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            self.canonical_url = attributes.get("href", "")
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def decimal_values(cells: list[str]) -> list[float]:
    values = []
    for cell in cells:
        match = re.fullmatch(r"\s*(\d{1,3}(?:\.\d{1,3})?)\s*", cell)
        if match:
            value = float(match.group(1))
            if 1.0 < value <= 200.0:
                values.append(value)
    return values


def main() -> None:
    cache = PROJECT_ROOT / "data" / "raw" / "odds_pages_cache"
    cache.mkdir(parents=True, exist_ok=True)
    output_path = PROJECT_ROOT / "data" / "raw" / "historical_match_odds_2022.csv"
    fixtures = load_2022_backtest(PROJECT_ROOT / "outputs" / "backtest_predictions.csv").set_index("shadow_match_id")
    discovery_path = PROJECT_ROOT / "outputs" / "odds_2022_source_discovery.csv"
    discovery = pd.read_csv(discovery_path) if discovery_path.exists() else pd.DataFrame()
    discovery_urls = {
        (int(row.match_id), str(row.source).lower()): row.candidate_search_url
        for row in discovery.itertuples(index=False)
    } if not discovery.empty else {}
    rows = []
    for path in sorted(cache.glob("2022_*_*.html")) if cache.exists() else []:
        filename = re.fullmatch(r"2022_(\d{3})_([a-z0-9_-]+)\.html", path.name, re.I)
        if not filename:
            continue
        match_id, source = int(filename.group(1)), filename.group(2).lower()
        if match_id not in fixtures.index:
            continue
        parser = TableTextParser()
        parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
        fixture = fixtures.loc[match_id]
        for cells in reversed(parser.rows):
            values = decimal_values(cells)
            if len(values) < 3:
                continue
            text = " ".join(cells).lower()
            bookmaker = cells[0] if cells and not re.fullmatch(r"\d+(?:\.\d+)?", cells[0]) else f"{source}_archive"
            value_type = "closing" if "closing" in text or "final" in text else "opening" if "opening" in text else "archive_consensus"
            confidence = "low" if value_type == "opening" else "medium"
            row = {column: pd.NA for column in HISTORICAL_ODDS_2022_COLUMNS}
            row.update(
                {
                    "match_id": match_id, "backtest_year": 2022, "penka_phase": fixture.penka_phase,
                    "date": fixture.date.date().isoformat(), "team_a": fixture.team_a, "team_b": fixture.team_b,
                    "bookmaker": bookmaker, "odds_team_a": values[0], "odds_draw": values[1], "odds_team_b": values[2],
                    "market_type": "1x2", "source_name": source,
                    "source_url": parser.canonical_url or discovery_urls.get((match_id, source), ""),
                    "value_type": value_type, "confidence": confidence,
                    "notes": f"Parsed from user-saved local HTML {path.name}; source timestamp unavailable and not fabricated.",
                }
            )
            rows.append(row)
            break
    output = pd.DataFrame(rows, columns=HISTORICAL_ODDS_2022_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Parsed {len(output)} local HTML odds row(s) into {output_path}. No network requests were made.")


if __name__ == "__main__":
    main()
