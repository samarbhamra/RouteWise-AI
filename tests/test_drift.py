from src.monitoring.drift import compute_drift
import pandas as pd


def test_drift_missing_files():
    report = compute_drift(current=None, baseline=None)
    # When snapshots are absent, report error or computed dict
    assert "error" in report or "drift_detected" in report


def test_drift_on_shifted_frame():
    base = pd.DataFrame({"congestion_ratio": [1.0] * 50, "hist_mean_speed_kmh": [30] * 50, "hour": [8] * 50, "mean_travel_time_min": [2.0] * 50})
    cur = pd.DataFrame({"congestion_ratio": [3.0] * 50, "hist_mean_speed_kmh": [10] * 50, "hour": [18] * 50, "mean_travel_time_min": [8.0] * 50})
    report = compute_drift(current=cur, baseline=base)
    assert report["drift_detected"] is True
