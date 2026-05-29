"""Run World Cup 2026 simulations from the command line."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from worldcup_predictor.config import DEV_SIMULATIONS, FINAL_SIMULATIONS, SimulationConfig
from worldcup_predictor.format_2026 import default_format
from worldcup_predictor.io import ensure_project_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run World Cup 2026 prediction simulations.")
    parser.add_argument("--n-simulations", type=int, default=DEV_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(n_simulations=args.n_simulations, random_seed=args.seed)
    config.validate()
    tournament_format = default_format()
    ensure_project_dirs()

    print("World Cup 2026 simulation skeleton")
    print(f"Simulations: {config.n_simulations:,} (max {FINAL_SIMULATIONS:,})")
    print(f"Teams: {tournament_format.n_teams}")
    print(f"Groups: {tournament_format.n_groups} x {tournament_format.teams_per_group}")
    print(f"Knockout qualifiers: {tournament_format.total_knockout_qualifiers}")
    print("Model implementation will be added module by module.")


if __name__ == "__main__":
    main()

