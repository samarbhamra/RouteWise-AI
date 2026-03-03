"""Congestion-aware route optimization using ML-predicted travel times."""
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.config import SEGMENT_FEATURES_PATH
from src.models.travel_time import load_predictor


class RouteOptimizer:
    """Compute shortest path by predicted travel time (congestion-aware)."""

    def __init__(self, G, segment_features: pd.DataFrame | None = None, predictor=None):
        self.G = G
        self.predictor = predictor or load_predictor()
        if segment_features is None and SEGMENT_FEATURES_PATH.exists():
            segment_features = pd.read_parquet(SEGMENT_FEATURES_PATH)
        self.segment_features = segment_features
        self._edge_weight_cache: dict[tuple, float] = {}

    def _get_predicted_time(self, u: int, v: int, k: int = 0) -> float:
        """Predicted travel time for edge (u, v, k); fallback to length-based if no features."""
        key = (u, v, k)
        if key in self._edge_weight_cache:
            return self._edge_weight_cache[key]
        if self.segment_features is not None:
            row = self.segment_features[
                (self.segment_features["u"] == u)
                & (self.segment_features["v"] == v)
                & (self.segment_features["k"] == k)
            ]
            if not row.empty:
                pred = self.predictor.predict(row).item()
                self._edge_weight_cache[key] = pred
                return pred
        # Fallback: use graph length and assume 50 km/h
        data = self.G.get_edge_data(u, v, k) or {}
        length = data.get("length") or 0
        speed_kmh = 50.0
        t = (length / 1000.0) / speed_kmh * 60.0 if length else 1.0
        self._edge_weight_cache[key] = t
        return t

    def _weight_fn(self, u, v, data):
        """Edge weight = predicted travel time (minutes)."""
        keys = list(self.G[u][v].keys())
        k = keys[0] if keys else 0
        return self._get_predicted_time(u, v, k)

    def shortest_path_by_travel_time(self, origin, destination) -> tuple[list, float]:
        """Return (node list, total predicted time in minutes)."""
        try:
            path = nx.shortest_path(
                self.G, origin, destination, weight=self._weight_fn
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            keys = list(self.G[u][v].keys())
            k = keys[0]
            total += self._get_predicted_time(u, v, k)
        return path, total

    def static_shortest_path_by_length(self, origin, destination) -> tuple[list, float]:
        """Static shortest path by geometric length (for comparison)."""
        try:
            path = nx.shortest_path(self.G, origin, destination, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 0.0
        total_length = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            keys = list(self.G[u][v].keys())
            total_length += self.G[u][v][keys[0]].get("length", 0) or 0
        # Convert to approximate time at 50 km/h for comparison
        total_time = (total_length / 1000.0) / 50.0 * 60.0
        return path, total_time
