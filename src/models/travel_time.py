"""Travel-time prediction model (scikit-learn) for congestion-aware routing."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS_PATH, TRAVEL_TIME_MODEL_PATH

# Features used for prediction (numeric only; exclude identifiers and target)
FEATURE_NAMES = [
    "length_m",
    "out_degree_u",
    "in_degree_v",
    "trip_count",
    "std_travel_time_min",
    "free_flow_time_min",
    "congestion_ratio",
]
TARGET = "mean_travel_time_min"


class TravelTimePredictor:
    """Predict segment travel time from congestion features."""

    def __init__(self, model=None, feature_names=None):
        self.feature_names = feature_names or FEATURE_NAMES
        self.model = model or Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X[self.feature_names].fillna(0)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        X = X[self.feature_names].fillna(0)
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        pred = self.predict(X)
        return {
            "mae": mean_absolute_error(y, pred),
            "rmse": mean_squared_error(y, pred, squared=False),
        }

    def save(self, path: Path | None = None, feature_columns_path: Path | None = None):
        path = path or TRAVEL_TIME_MODEL_PATH
        feature_columns_path = feature_columns_path or FEATURE_COLUMNS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        joblib.dump(self.feature_names, feature_columns_path)

    @classmethod
    def load(cls, path: Path | None = None, feature_columns_path: Path | None = None) -> "TravelTimePredictor":
        path = path or TRAVEL_TIME_MODEL_PATH
        feature_columns_path = feature_columns_path or FEATURE_COLUMNS_PATH
        model = joblib.load(path)
        feature_names = joblib.load(feature_columns_path)
        return cls(model=model, feature_names=feature_names)


def load_predictor() -> TravelTimePredictor:
    """Load trained predictor from config paths."""
    return TravelTimePredictor.load()
