"""General utilities (model persistence, etc.)."""

import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def save_model(model, filename: str = "best_model.joblib"):
    """Save a model (or pipeline) to the models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(filename: str = "best_model.joblib"):
    """Load a model from the models/ directory."""
    path = MODELS_DIR / filename
    return joblib.load(path)
