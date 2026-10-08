"""A/B/C evaluation: model MAE/RMSE and route-level variants."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import EVAL_REPORT_PATH, LIGHTGBM_MODEL_PATH, LSTM_MODEL_PATH, SEGMENT_FEATURES_PATH
from src.data import load_network, load_trips
from src.features import TARGET, build_segment_congestion_features, time_aware_split
from src.models.lightgbm_travel_time import LightGBMTravelTime
from src.models.lstm_travel_time import LSTMTravelTime
from src.routing.optimizer import RouteConstraints, RouteOptimizer


def _route_stats(opt: RouteOptimizer, pairs, constraints: RouteConstraints):
    times = []
    for o, d in pairs:
        path, t, _ = opt.route(o, d, constraints)
        if path and t > 0:
            times.append(t)
    return np.array(times) if times else np.array([])


def main():
    G = load_network()
    trips = load_trips()
    features = (
        pd.read_parquet(SEGMENT_FEATURES_PATH)
        if SEGMENT_FEATURES_PATH.exists()
        else build_segment_congestion_features(G, trips)
    )
    train, test = time_aware_split(features)
    y = test[TARGET]
    ff = test["free_flow_time_min"]
    baseline_mae = float((y - ff).abs().mean())
    baseline_rmse = float(np.sqrt(((y - ff) ** 2).mean()))

    model_metrics = {
        "free_flow": {"mae": baseline_mae, "rmse": baseline_rmse},
    }
    if LIGHTGBM_MODEL_PATH.exists():
        lgbm = LightGBMTravelTime.load()
        model_metrics["lightgbm"] = lgbm.evaluate(test)
        model_metrics["lightgbm"]["pct_lower_error_vs_freeflow"] = (
            1 - model_metrics["lightgbm"]["mae"] / baseline_mae
        ) * 100
    if LSTM_MODEL_PATH.exists():
        lstm = LSTMTravelTime.load()
        model_metrics["lstm"] = lstm.evaluate(test)
        model_metrics["lstm"]["pct_lower_error_vs_freeflow"] = (
            1 - model_metrics["lstm"]["mae"] / baseline_mae
        ) * 100

    nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
    rng = np.random.default_rng(42)
    pairs = []
    for _ in range(min(80, max(10, len(nodes)))):
        o, d = rng.choice(nodes, size=2, replace=False)
        pairs.append((int(o), int(d)))

    constraints = RouteConstraints()
    static_opt = RouteOptimizer(G, features, variant="static")
    static_t = _route_stats(static_opt, pairs, constraints)
    route_metrics = {"static_mean_min": float(static_t.mean()) if len(static_t) else None}

    for variant in ("lightgbm", "lstm"):
        path = LIGHTGBM_MODEL_PATH if variant == "lightgbm" else LSTM_MODEL_PATH
        if not path.exists():
            continue
        try:
            opt = RouteOptimizer(G, features, variant=variant, hour=8, dow=2)
            times = _route_stats(opt, pairs, constraints)
        except FileNotFoundError:
            continue
        if len(times) and len(static_t):
            n = min(len(times), len(static_t))
            improved = float((times[:n] < static_t[:n]).mean() * 100)
            pct_shorter = float((1 - times.mean() / static_t.mean()) * 100) if static_t.mean() else 0
            delay = float((times - times.min()).mean()) if len(times) else 0
            route_metrics[variant] = {
                "mean_min": float(times.mean()),
                "pct_shorter_vs_static": pct_shorter,
                "pct_routes_improved": improved,
                "mean_delay_vs_best": delay,
            }

    # Constraint satisfaction sample
    constrained = RouteConstraints(avoid_highways=True, minimize_delay_risk=True)
    if LIGHTGBM_MODEL_PATH.exists():
        opt = RouteOptimizer(G, features, variant="lightgbm")
        ok = 0
        n = 0
        for o, d in pairs[:20]:
            path, t, feasible = opt.route(o, d, constrained)
            if path:
                n += 1
                ok += int(feasible)
        route_metrics["constraint_satisfaction_rate"] = (ok / n * 100) if n else None

    report = {"model_level": model_metrics, "route_level": route_metrics}
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
