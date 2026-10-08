"""Distribution drift checks vs the training feature snapshot."""
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import DATA_DIR, SEGMENT_FEATURES_PATH, TRAIN_SNAPSHOT_PATH

DRIFT_REPORT_PATH = DATA_DIR / "drift_report.json"
WATCH_COLS = ["congestion_ratio", "hist_mean_speed_kmh", "hour", "mean_travel_time_min"]


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    cuts = np.linspace(np.min(expected), np.max(expected) + 1e-9, bins + 1)
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e = e_counts / max(e_counts.sum(), 1)
    a = a_counts / max(a_counts.sum(), 1)
    e = np.clip(e, 1e-6, 1)
    a = np.clip(a, 1e-6, 1)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_drift(
    current: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
) -> dict:
    if baseline is None and TRAIN_SNAPSHOT_PATH.exists():
        baseline = pd.read_parquet(TRAIN_SNAPSHOT_PATH)
    if current is None and SEGMENT_FEATURES_PATH.exists():
        current = pd.read_parquet(SEGMENT_FEATURES_PATH)
    if baseline is None or current is None:
        return {"error": "Missing baseline or current features. Train models first."}

    columns = {}
    flagged = []
    for col in WATCH_COLS:
        if col not in baseline.columns or col not in current.columns:
            continue
        psi = _psi(baseline[col].to_numpy(dtype=float), current[col].to_numpy(dtype=float))
        ks = ks_2samp(baseline[col].dropna(), current[col].dropna())
        rec = {"psi": psi, "ks_pvalue": float(ks.pvalue), "ks_stat": float(ks.statistic)}
        columns[col] = rec
        if psi > 0.2 or ks.pvalue < 0.01:
            flagged.append(col)

    report = {
        "flagged_columns": flagged,
        "drift_detected": bool(flagged),
        "columns": columns,
        "n_baseline": int(len(baseline)),
        "n_current": int(len(current)),
    }
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report
