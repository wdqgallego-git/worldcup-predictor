"""Project configuration and constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

N_TEAMS = 48
N_GROUPS = 12
TEAMS_PER_GROUP = 4
DIRECT_QUALIFIERS_PER_GROUP = 2
BEST_THIRD_PLACE_QUALIFIERS = 8
KNOCKOUT_TEAMS = 32
FINALIST_MATCHES = 8

DEV_SIMULATIONS = 1_000
FINAL_SIMULATIONS = 20_000

COMPANY_SCORING = {
    "champion": 25,
    "runner_up": 15,
    "top_scorer": 35,
    "mvp": 10,
    "golden_glove": 10,
}

STRATEGIC_EV_MIN_RATIO = 0.85
STRATEGIC_EV_TARGET_RATIO = 0.90
RANDOM_SEED = 2026

