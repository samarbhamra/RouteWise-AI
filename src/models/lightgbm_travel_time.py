"""Gradient boosting travel-time regressor (LightGBM, or sklearn if libomp is missing)."""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import FEATURE_COLUMNS_PATH, LIGHTGBM_META_PATH, LIGHTGBM_MODEL_PATH, MODELS_DIR
from src.features.build import TABULAR_FEATURES, TARGET

SKLEARN_MODEL_PATH = MODELS_DIR / "lightgbm_travel_time.joblib"


def _try_import_lightgbm():
    try:
        import lightgbm as lgb

        _ = lgb.__version__
        return lgb
    except OSError:
        warnings.warn(
            "LightGBM native library failed to load (often missing libomp on macOS). "
            "Using sklearn HistGradientBoostingRegressor instead. "
            "Fix LightGBM with: brew install libomp"
        )
        return None
    except Exception:
        return None


lgb = _try_import_lightgbm()


class LightGBMTravelTime:
    def __init__(self, model=None, feature_names=None, backend: str | None = None):
        self.feature_names = feature_names or list(TABULAR_FEATURES)
        self.model = model
        self.backend = backend or ("lightgbm" if lgb is not None else "sklearn_hist")

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None):
        X = train[self.feature_names].fillna(0)
        y = train[TARGET]
        if self.backend == "lightgbm" and lgb is not None:
            params = {
                "objective": "regression",
                "metric": "mae",
                "learning_rate": 0.08,
                "num_leaves": 48,
                "max_depth": 8,
                "verbosity": -1,
                "seed": 42,
            }
            dtrain = lgb.Dataset(X, label=y)
            valid_sets = [dtrain]
            if valid is not None and not valid.empty:
                dvalid = lgb.Dataset(valid[self.feature_names].fillna(0), label=valid[TARGET])
                valid_sets.append(dvalid)
            self.model = lgb.train(params, dtrain, num_boost_round=200, valid_sets=valid_sets)
        else:
            self.backend = "sklearn_hist"
            self.model = HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.08,
                max_iter=200,
                random_state=42,
            )
            self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.feature_names].fillna(0)
        return np.asarray(self.model.predict(arr))

    def evaluate(self, df: pd.DataFrame) -> dict:
        pred = self.predict(df)
        y = df[TARGET]
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
        }

    def save(self, path: Path | None = None):
        path = path or LIGHTGBM_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "lightgbm" and hasattr(self.model, "save_model"):
            self.model.save_model(str(path))
        else:
            joblib.dump(self.model, SKLEARN_MODEL_PATH)
        joblib.dump({"feature_names": self.feature_names, "backend": self.backend}, LIGHTGBM_META_PATH)
        joblib.dump(self.feature_names, FEATURE_COLUMNS_PATH)

    @classmethod
    def load(cls, path: Path | None = None) -> "LightGBMTravelTime":
        path = path or LIGHTGBM_MODEL_PATH
        meta = joblib.load(LIGHTGBM_META_PATH)
        backend = meta.get("backend", "lightgbm")
        names = meta["feature_names"]
        if backend == "sklearn_hist" or lgb is None:
            model = joblib.load(SKLEARN_MODEL_PATH)
            return cls(model=model, feature_names=names, backend="sklearn_hist")
        booster = lgb.Booster(model_file=str(path))
        return cls(model=booster, feature_names=names, backend="lightgbm")
