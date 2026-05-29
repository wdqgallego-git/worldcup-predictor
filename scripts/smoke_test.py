"""Smoke test the project skeleton."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_company_scoring_rules
from tournament_simulator import validate_2026_format


def main() -> None:
    validate_2026_format()
    scoring = load_company_scoring_rules()
    required_awards = {"champion", "runner_up", "top_scorer", "mvp", "golden_glove"}
    found_awards = {row["category"] for row in scoring}
    missing_awards = required_awards - found_awards
    if missing_awards:
        raise AssertionError(f"Missing scoring rules: {sorted(missing_awards)}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
