"""Train PyTorch LSTM travel-time model with MLflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd

from src.config import SEGMENT_FEATURES_PATH
from src.data import load_network, load_trips
from src.features import TARGET, build_segment_congestion_features, time_aware_split
from src.models.lstm_travel_time import LSTMTravelTime
from src.models.mlflow_utils import start_run


def main():
    G = load_network()
    trips = load_trips()
    if SEGMENT_FEATURES_PATH.exists():
        features = pd.read_parquet(SEGMENT_FEATURES_PATH)
    else:
        features = build_segment_congestion_features(G, trips)

    train, test = time_aware_split(features)
    print(f"Training LSTM on {len(train):,} rows...")
    model = LSTMTravelTime()
    model.fit(train, epochs=6)
    metrics = model.evaluate(test)
    baseline_mae = (test[TARGET] - test["free_flow_time_min"]).abs().mean()
    lift = (1 - metrics["mae"] / baseline_mae) * 100 if baseline_mae else 0
    print(f"LSTM MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}")
    print(f"Free-flow MAE={baseline_mae:.4f} → {lift:.1f}% lower error")
    model.save()
    try:
        with start_run("routewise", "lstm") as run:
            if run is not None:
                mlflow.log_metrics(
                    {**metrics, "baseline_mae": float(baseline_mae), "pct_lower_error": float(lift)}
                )
                mlflow.log_params({"epochs": 6, "n_train": len(train)})
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
    print("Saved LSTM model.")


if __name__ == "__main__":
    main()
