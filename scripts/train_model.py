"""Build congestion features and train travel-time prediction model."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEGMENT_FEATURES_PATH, TRIPS_PATH
from src.data import load_network, load_trips
from src.features import build_segment_congestion_features
from src.models.travel_time import TravelTimePredictor, FEATURE_NAMES, TARGET
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    print("Loading network and trips...")
    G = load_network()
    trips = load_trips()
    print(f"Trips: {len(trips):,}")

    print("Building segment congestion features (Pandas + NetworkX)...")
    features = build_segment_congestion_features(G, trips)
    print(f"Segments with features: {len(features):,}")
    print(f"Saved to {SEGMENT_FEATURES_PATH}")

    # Train only on segments that have trip data (mean_travel_time_min > 0)
    train_df = features[features[TARGET] > 0].copy()
    if len(train_df) < 10:
        train_df = features.copy()
        train_df[TARGET] = train_df["free_flow_time_min"].fillna(1.0)

    X = train_df[FEATURE_NAMES].fillna(0)
    y = train_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training travel-time model (GradientBoostingRegressor)...")
    predictor = TravelTimePredictor()
    predictor.fit(X_train, y_train)

    metrics = predictor.evaluate(X_test, y_test)
    print(f"Test MAE: {metrics['mae']:.4f} min, RMSE: {metrics['rmse']:.4f} min")

    # Compare to baseline (free-flow time) on test set
    test_idx = X_test.index
    baseline_pred = train_df.loc[test_idx, "free_flow_time_min"].values
    baseline_mae = np.abs(y_test.values - baseline_pred).mean()
    print(f"Baseline (free-flow) MAE: {baseline_mae:.4f} min")
    if baseline_mae > 0:
        pct = (1 - metrics["mae"] / baseline_mae) * 100
        print(f"Model achieves ~{pct:.0f}% lower error than static free-flow baseline")

    predictor.save()
    print("Model saved. You can run the Flask app: python -m src.app.main")


if __name__ == "__main__":
    main()
