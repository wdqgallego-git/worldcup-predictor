"""Run final match, tournament, and company-game predictions."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import N_SIMULATIONS_FINAL
from tournament_simulator import validate_2026_format


def main() -> None:
    validate_2026_format()
    print("Final predictions skeleton")
    print(f"Final simulation budget: up to {N_SIMULATIONS_FINAL:,}")


if __name__ == "__main__":
    main()
