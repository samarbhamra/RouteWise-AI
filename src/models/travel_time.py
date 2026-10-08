"""Load travel-time predictors by variant name."""
from pathlib import Path

from src.config import LIGHTGBM_MODEL_PATH, LSTM_MODEL_PATH, MODELS_DIR


def load_predictor(variant: str = "lightgbm"):
    variant = (variant or "lightgbm").lower()
    if variant in ("lightgbm", "lgbm", "gbm"):
        from src.models.lightgbm_travel_time import LightGBMTravelTime

        if not LIGHTGBM_MODEL_PATH.exists() and not (MODELS_DIR / "lightgbm_travel_time.joblib").exists():
            raise FileNotFoundError("Train LightGBM first: python scripts/train_lightgbm.py")
        return LightGBMTravelTime.load()
    if variant in ("lstm", "pytorch", "torch"):
        from src.models.lstm_travel_time import LSTMTravelTime

        if not LSTM_MODEL_PATH.exists():
            raise FileNotFoundError("Train LSTM first: python scripts/train_lstm.py")
        return LSTMTravelTime.load()
    raise ValueError(f"Unknown model variant: {variant}")
