"""Build features and train LightGBM travel-time model with MLflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd

from src.config import SEGMENT_FEATURES_PATH
from src.data import load_network, load_trips
from src.features import TARGET, build_segment_congestion_features, time_aware_split
from src.models.lightgbm_travel_time import LightGBMTravelTime
from src.models.mlflow_utils import start_run


def main():
    print("Loading network and trips...")
    G = load_network()
    trips = load_trips()
    if SEGMENT_FEATURES_PATH.exists():
        features = pd.read_parquet(SEGMENT_FEATURES_PATH)
        print(f"Loaded cached features: {len(features):,}")
    else:
        print("Building segment-hour features...")
        features = build_segment_congestion_features(G, trips)
        print(f"Wrote {len(features):,} rows to {SEGMENT_FEATURES_PATH}")

    train, test = time_aware_split(features)
    print(f"Time-aware split: train={len(train):,} test={len(test):,}")

    model = LightGBMTravelTime()
    model.fit(train, valid=test)
    metrics = model.evaluate(test)
    baseline_mae = (test[TARGET] - test["free_flow_time_min"]).abs().mean()
    lift = (1 - metrics["mae"] / baseline_mae) * 100 if baseline_mae else 0
    print(f"LightGBM MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
    print(f"Free-flow MAE={baseline_mae:.4f} → {lift:.1f}% lower error")

    model.save()
    try:
        with start_run("routewise", "lightgbm") as run:
            if run is not None:
                mlflow.log_metrics(
                    {**metrics, "baseline_mae": float(baseline_mae), "pct_lower_error": float(lift)}
                )
                mlflow.log_params(
                    {"n_train": len(train), "n_test": len(test), "backend": model.backend}
                )
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
    print("Saved LightGBM model. Next: python scripts/train_lstm.py")


if __name__ == "__main__":
    main()
