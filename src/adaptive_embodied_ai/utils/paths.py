from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

MOVEMENT_DATA_DIR = DATA_DIR / "movement"

MODELS_DIR = PROJECT_ROOT / "models"