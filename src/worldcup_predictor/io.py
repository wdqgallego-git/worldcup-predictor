"""Input/output helpers for local and Colab execution."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_project_dirs() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

