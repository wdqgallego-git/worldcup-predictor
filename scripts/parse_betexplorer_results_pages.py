"""Parse cached BetExplorer World Cup result pages without network access."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import apply_team_mapping, load_team_name_mapping  # noqa: E402
from odds_priors import MATCH_ODDS_COLUMNS, load_match_odds  # noqa: E402


YEARS = (2014, 2018, 2022)
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "odds_pages_cache"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "historical_match_odds.csv"
BACKTEST_PATH = PROJECT_ROOT / "outputs" / "backtest_predictions.csv"
TYPED_2022_PATH = PROJECT_ROOT / "data" / "raw" / "historical_match_odds_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATE_PATTERN = re.compile(r"^\d{2}\.\d{2}\.20\d{2}$")


class ResultsTableParser(HTMLParser):
    """Collect table cells, visible text, and nested data-odd attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, object]]] = []
        self._row: list[dict[str, object]] | None = None
        self._cell: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"text": [], "odds": []}
            if attributes.get("data-odd"):
                self._cell["odds"].append(attributes["data-odd"])
        elif self._cell is not None and attributes.get("data-odd"):
            self._cell["odds"].append(attributes["data-odd"])

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            text = " ".join("".join(self._cell["text"]).split())
            self._cell["text"] = text
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
            self._cell = None


def _cell_odd(cell: dict[str, object]) -> float:
    odds = cell.get("odds") or []
    value = odds[0] if odds else cell.get("text", "")
    decimal = float(str(value).strip())
    if not np.isfinite(decimal) or decimal <= 1.0:
        raise ValueError(f"Invalid archive decimal odd: {value!r}")
    return decimal


def parse_page(path: Path, year: int, stage: str) -> pd.DataFrame:
    parser = ResultsTableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    records = []
    for cells in parser.rows:
        if len(cells) < 6 or not DATE_PATTERN.fullmatch(str(cells[-1]["text"])):
            continue
        identity = str(cells[0]["text"])
        if " - " not in identity:
            raise ValueError(f"Cannot split archive teams in {path.name}: {identity!r}")
        team_a, team_b = identity.split(" - ", 1)
        records.append(
            {
                "date": pd.to_datetime(cells[-1]["text"], format="%d.%m.%Y"),
                "team_a": team_a.strip(),
                "team_b": team_b.strip(),
                "odds_team_a": _cell_odd(cells[2]),
                "odds_draw": _cell_odd(cells[3]),
                "odds_team_b": _cell_odd(cells[4]),
                "world_cup_year": year,
                "archive_stage": stage,
                "source_file": path.name,
            }
        )
    return pd.DataFrame(records)


def load_cached_pages() -> pd.DataFrame:
    parts = []
    for year in YEARS:
        for suffix, stage in (("main", "group_stage"), ("playoffs", "knockout")):
            path = CACHE_DIR / f"results_{year}_betexplorer_{suffix}.html"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required cached archive page: {path}. Save it as Webpage, HTML Only; no odds will be inferred."
                )
            parsed = parse_page(path, year, stage)
            expected = 48 if suffix == "main" else 16
            if len(parsed) != expected:
                raise ValueError(f"{path.name} contains {len(parsed)} parsed matches; expected exactly {expected}.")
            parts.append(parsed)
    return pd.concat(parts, ignore_index=True)


def match_to_backtests(parsed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    backtests = pd.read_csv(BACKTEST_PATH)
    backtests["date"] = pd.to_datetime(backtests["date"], errors="raise").dt.normalize()
    backtests = backtests[backtests["backtest_year"].isin(YEARS)].copy()
    backtests["world_cup_year"] = backtests["backtest_year"].astype(int)
    backtests["year_match_number"] = backtests.groupby("world_cup_year").cumcount() + 1
    backtests["formal_match_id"] = backtests["world_cup_year"] * 1000 + backtests["year_match_number"]
    lookup_columns = [
        "formal_match_id", "world_cup_year", "date", "team_a", "team_b", "penka_phase"
    ]
    direct = backtests[lookup_columns].copy()
    direct["orientation"] = "direct"
    swapped = direct.rename(columns={"team_a": "team_b", "team_b": "team_a"}).copy()
    swapped["orientation"] = "swapped"
    lookup = pd.concat([direct, swapped], ignore_index=True)
    matched = parsed.merge(
        lookup,
        on=["world_cup_year", "date", "team_a", "team_b"],
        how="left",
        validate="many_to_one",
    )
    unmatched = matched[matched["formal_match_id"].isna()].copy()
    if not unmatched.empty:
        return matched, unmatched
    reversed_rows = matched["orientation"].eq("swapped")
    matched.loc[reversed_rows, ["team_a", "team_b"]] = matched.loc[
        reversed_rows, ["team_b", "team_a"]
    ].to_numpy()
    matched.loc[reversed_rows, ["odds_team_a", "odds_team_b"]] = matched.loc[
        reversed_rows, ["odds_team_b", "odds_team_a"]
    ].to_numpy()
    return matched, unmatched


def build_formal_odds(matched: pd.DataFrame) -> pd.DataFrame:
    formal = pd.DataFrame(index=matched.index)
    formal["match_id"] = matched["formal_match_id"].astype(int)
    formal["date"] = matched["date"].dt.date.astype(str)
    formal["team_a"] = matched["team_a"]
    formal["team_b"] = matched["team_b"]
    formal["bookmaker"] = "betexplorer_archive_average"
    formal["odds_team_a"] = matched["odds_team_a"]
    formal["odds_draw"] = matched["odds_draw"]
    formal["odds_team_b"] = matched["odds_team_b"]
    formal["market_type"] = "1x2"
    formal["source_timestamp"] = pd.NA
    for column in MATCH_ODDS_COLUMNS:
        if column not in formal:
            formal[column] = np.nan
    formal["world_cup_year"] = matched["world_cup_year"].astype(int)
    formal["penka_phase"] = matched["penka_phase"]
    formal["archive_stage"] = matched["archive_stage"]
    formal["source_name"] = "BetExplorer / OddsPortal archive average"
    formal["source_url"] = [
        f"https://www.betexplorer.com/football/world/world-cup-{year}/results/"
        for year in matched["world_cup_year"]
    ]
    formal["source_file"] = matched["source_file"]
    formal["value_type"] = "archive_average"
    formal["confidence"] = "medium"
    formal["notes"] = (
        "Cached public archive result page; source timestamp and kickoff-relative timestamp unavailable and left blank."
    )
    leading = MATCH_ODDS_COLUMNS + [
        "world_cup_year", "penka_phase", "archive_stage", "source_name", "source_url",
        "source_file", "value_type", "confidence", "notes",
    ]
    return formal[leading].sort_values(["world_cup_year", "date", "match_id"], kind="stable").reset_index(drop=True)


def compare_typed_2022(formal: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date", "team_a", "team_b", "odds_team_a", "odds_draw", "odds_team_b"
    ]
    parsed = formal[formal["world_cup_year"].eq(2022)][columns].copy()
    if not TYPED_2022_PATH.exists():
        return parsed.assign(diff_status="typed_file_missing")
    typed = pd.read_csv(TYPED_2022_PATH, usecols=columns)
    typed["date"] = pd.to_datetime(typed["date"]).dt.date.astype(str)
    parsed["date"] = pd.to_datetime(parsed["date"]).dt.date.astype(str)
    mapping = load_team_name_mapping(PROJECT_ROOT / "data" / "raw" / "team_name_mapping.csv")
    typed = apply_team_mapping(typed, ["team_a", "team_b"], mapping)
    direct = typed.copy()
    direct["typed_orientation"] = "direct"
    reversed_typed = typed.rename(columns={"team_a": "team_b", "team_b": "team_a"}).copy()
    reversed_typed[["odds_team_a", "odds_team_b"]] = reversed_typed[
        ["odds_team_b", "odds_team_a"]
    ].to_numpy()
    reversed_typed["typed_orientation"] = "swapped"
    oriented_typed = pd.concat([direct, reversed_typed], ignore_index=True).drop_duplicates(
        ["date", "team_a", "team_b"], keep="first"
    )
    merged = parsed.merge(
        oriented_typed,
        on=["date", "team_a", "team_b"],
        how="left",
        suffixes=("_parsed", "_typed"),
        indicator=True,
    )
    differences = []
    for column in ("odds_team_a", "odds_draw", "odds_team_b"):
        differences.append((merged[f"{column}_parsed"] - merged[f"{column}_typed"]).abs())
    merged["max_absolute_odds_difference"] = pd.concat(differences, axis=1).max(axis=1)
    merged["diff_status"] = np.where(
        merged["_merge"].ne("both"),
        merged["_merge"].astype(str),
        np.where(merged["max_absolute_odds_difference"].le(1e-9), "exact_match", "odds_mismatch"),
    )
    return merged


def write_qa(formal: pd.DataFrame, typed_diff: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_match_odds(OUTPUT_PATH)
    qa_rows = []
    for year, rows in loaded.groupby("world_cup_year", sort=True):
        qa_rows.append(
            {
                "scope": "matched_odds_all_matches",
                "world_cup_year": int(year),
                "rows": len(rows),
                "matches": rows["match_id"].nunique(),
                "group_stage_matches": int(rows["penka_phase"].eq("group").sum()),
                "odds_status_flags": "; ".join(
                    f"{key}={value}" for key, value in rows["odds_status"].value_counts().sort_index().items()
                ),
                "overround_min": rows["overround"].min(),
                "overround_median": rows["overround"].median(),
                "overround_max": rows["overround"].max(),
            }
        )
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(OUTPUT_DIR / "historical_match_odds_qa.csv", index=False)
    typed_diff.to_csv(OUTPUT_DIR / "historical_match_odds_2022_typed_diff.csv", index=False)
    references = [
        (2014, "Brazil", "Germany"),
        (2014, "Germany", "Argentina"),
        (2014, "Brazil", "Croatia"),
        (2014, "Spain", "Netherlands"),
        (2014, "Netherlands", "Argentina"),
        (2018, "Germany", "Mexico"),
        (2018, "France", "Croatia"),
        (2018, "Russia", "Saudi Arabia"),
        (2018, "Portugal", "Spain"),
        (2018, "Belgium", "Japan"),
        (2022, "Argentina", "Saudi Arabia"),
        (2022, "Argentina", "France"),
        (2022, "Qatar", "Ecuador"),
        (2022, "Spain", "Costa Rica"),
        (2022, "Morocco", "Portugal"),
    ]
    provenance_rows = []
    for year, team_a, team_b in references:
        row = formal[
            formal["world_cup_year"].eq(year)
            & formal["team_a"].eq(team_a)
            & formal["team_b"].eq(team_b)
        ]
        if row.empty:
            row = formal[
                formal["world_cup_year"].eq(year)
                & formal["team_a"].eq(team_b)
                & formal["team_b"].eq(team_a)
            ]
        if row.empty:
            raise ValueError(f"Provenance reference missing: {year} {team_a} vs {team_b}")
        provenance_rows.append(row.iloc[0][[
            "world_cup_year", "date", "team_a", "team_b", "odds_team_a", "odds_draw",
            "odds_team_b", "source_file", "source_url",
        ]].to_dict())
    pd.DataFrame(provenance_rows).to_csv(OUTPUT_DIR / "historical_match_odds_provenance.csv", index=False)
    mismatch_count = int(typed_diff["diff_status"].ne("exact_match").sum())
    lines = [
        "Historical Match Odds QA",
        "========================",
        qa.to_string(index=False),
        "",
        f"total_rows: {len(formal)}",
        f"total_matches: {formal['match_id'].nunique()}",
        f"typed_2022_diff_mismatches: {mismatch_count}",
        "formal_gate_input_ready: True",
    ]
    (OUTPUT_DIR / "historical_match_odds_qa.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parsed = load_cached_pages()
    mapping = load_team_name_mapping(PROJECT_ROOT / "data" / "raw" / "team_name_mapping.csv")
    parsed = apply_team_mapping(parsed, ["team_a", "team_b"], mapping)
    matched, unmatched = match_to_backtests(parsed)
    if not unmatched.empty:
        examples = unmatched[["world_cup_year", "date", "team_a", "team_b", "source_file"]]
        raise ValueError("Archive matches failed to map to the backtest:\n" + examples.to_string(index=False))
    counts = matched.groupby("world_cup_year")["formal_match_id"].nunique()
    if counts.to_dict() != {2014: 64, 2018: 64, 2022: 64}:
        raise ValueError(f"Expected exactly 64 matched fixtures per year; received {counts.to_dict()}")
    formal = build_formal_odds(matched)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    formal.to_csv(OUTPUT_PATH, index=False)
    typed_diff = compare_typed_2022(formal)
    write_qa(formal, typed_diff)
    print(f"Wrote {len(formal)} archive odds rows to {OUTPUT_PATH}")
    print(formal.groupby('world_cup_year').size().to_string())
    print(f"2022 typed-vs-parsed mismatches: {int(typed_diff['diff_status'].ne('exact_match').sum())}")


if __name__ == "__main__":
    main()
