"""Project configuration and constants."""

import os
from pathlib import Path

from contest_config import AWARD_POINTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from an optional .env file without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean flag. Received: {value!r}")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


_load_dotenv(PROJECT_ROOT / ".env")

TRAIN_START_DATE = "2000-01-01"
PREDICTION_REFERENCE_DATE = "2026-06-01"

MAX_GOALS_DEV = 6
MAX_GOALS_FINAL = 7

N_SIMULATIONS_DEV = 1000
N_SIMULATIONS_FINAL = 20000

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"

MODEL_PARAMS_POISSON_HGB = {
    "loss": "poisson",
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.1,
    "random_state": 42,
}

MODEL_PARAMS_SQUARED_HGB = {
    "loss": "squared_error",
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 0.1,
    "random_state": 42,
}

DEVELOPMENT_MODE = _env_flag("DEVELOPMENT_MODE", True)

# Market-odds prior. The blend is a gated candidate strategy: blended picks only
# replace the validated baseline when the backtest promotion gate passes.
MARKET_BLEND_ENABLED = _env_flag("MARKET_BLEND_ENABLED", True)
MARKET_BLEND_WEIGHT = _env_float("MARKET_BLEND_WEIGHT", 0.5)
if not 0.0 <= MARKET_BLEND_WEIGHT <= 1.0:
    raise ValueError(f"MARKET_BLEND_WEIGHT must be between 0 and 1. Received: {MARKET_BLEND_WEIGHT}")

# Score-probability backend used by the pick optimizer ("independent" or "dixon_coles").
# Default switched to dixon_coles after the 11-tournament expanded backtest
# (outputs/expanded_strategy_decision.txt): it beat independent on the World Cup
# (+0.083 pts/match), Euro (+0.084), and pooled (+0.045) group scopes.
SCORE_PMF_METHOD = os.environ.get("SCORE_PMF_METHOD", "dixon_coles").strip() or "dixon_coles"

MATCH_ODDS_PATH = "data/raw/match_odds.csv"
FUTURES_ODDS_PATH = "data/raw/futures_odds.csv"
HISTORICAL_MATCH_ODDS_PATH = "data/raw/historical_match_odds.csv"
MARKET_BLEND_GATE_PATH = "outputs/market_blend_gate.json"

# Secrets are read from the environment (or the git-ignored root .env). Never commit keys.
BALLDONTLIE_API_KEY_ENV = "BALLDONTLIE_API_KEY"
THE_ODDS_API_KEY_ENV = "THE_ODDS_API_KEY"
