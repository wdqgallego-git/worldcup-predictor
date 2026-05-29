"""Run award prediction workflow."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_raw_csv


def main() -> None:
    players = load_raw_csv("players_sample.csv")
    print("Awards prediction skeleton")
    print(f"Loaded sample players: {len(players)}")


if __name__ == "__main__":
    main()

