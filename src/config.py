"""Project configuration and constants."""

from contest_config import AWARD_POINTS

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

DEVELOPMENT_MODE = True
