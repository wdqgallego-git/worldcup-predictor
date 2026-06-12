"""Normalize a sourced historical odds dataset for the 2022 shadow backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_odds_2022 import (  # noqa: E402
    load_2022_backtest,
    normalize_import,
    parse_column_map,
    read_source_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import free sourced 2022 World Cup odds.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--column-map-json")
    parser.add_argument("--odds-format", choices=["auto", "decimal", "american", "fractional"], default="auto")
    parser.add_argument("--value-type", default="archive_consensus")
    parser.add_argument("--confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--output", default="data/raw/historical_match_odds_2022.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = read_source_table(args.input_file)
    normalized = normalize_import(
        source,
        args.year,
        args.source_name,
        args.source_url or f"https://www.kaggle.com/datasets/{args.source_name}",
        parse_column_map(args.column_map_json),
        args.odds_format,
        args.value_type,
        args.confidence,
        load_2022_backtest(PROJECT_ROOT / "outputs" / "backtest_predictions.csv"),
    )
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    print(f"Wrote {len(normalized)} sourced odds row(s) covering {normalized['match_id'].nunique()} matched match(es) to {output}")


if __name__ == "__main__":
    main()
