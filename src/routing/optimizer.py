"""Constraint-aware congestion routing (NetworkX)."""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from src.config import DEFAULT_ROUTE_MODEL, SEGMENT_FEATURES_PATH
from src.features.highway import is_highway, parse_highway


@dataclass
class RouteConstraints:
    avoid_highways: bool = False
    prefer_fewer_turns: bool = False
    minimize_delay_risk: bool = False
    max_travel_time_min: float | None = None


class RouteOptimizer:
    def __init__(
        self,
        G,
        segment_features: pd.DataFrame | None = None,
        variant: str = DEFAULT_ROUTE_MODEL,
        hour: int = 8,
        dow: int = 2,
    ):
        self.G = G
        self.variant = variant
        self.hour = hour
        self.dow = dow
        if segment_features is None and SEGMENT_FEATURES_PATH.exists():
            segment_features = pd.read_parquet(SEGMENT_FEATURES_PATH)
        self.segment_features = segment_features
        self.predictor = None
        if variant != "static":
            from src.models.travel_time import load_predictor

            self.predictor = load_predictor(variant)
        self._row_index = None
        self._lstm_profiles = {}
        if self.segment_features is not None and not self.segment_features.empty:
            self._row_index = self.segment_features.set_index(["u", "v", "k", "hour", "dow"], drop=False)
            if variant == "lstm":
                for (u, v, k, d), g in self.segment_features.groupby(["u", "v", "k", "dow"]):
                    by_h = g.groupby("hour")["mean_travel_time_min"].mean()
                    self._lstm_profiles[(int(u), int(v), int(k), int(d))] = np.array(
                        [float(by_h.get(h, by_h.mean())) for h in range(24)], dtype=np.float32
                    )

    def _edge_row(self, u, v, k) -> pd.Series | None:
        if self._row_index is None:
            return None
        for key in ((u, v, k, self.hour, self.dow),):
            try:
                row = self._row_index.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row
            except KeyError:
                pass
        sub = self.segment_features[
            (self.segment_features["u"] == u)
            & (self.segment_features["v"] == v)
            & (self.segment_features["k"] == k)
        ]
        if sub.empty:
            return None
        return sub.iloc[(sub["hour"] - self.hour).abs().argmin()]

    def _predicted_time(self, u, v, k) -> float:
        data = self.G.get_edge_data(u, v, k) or {}
        length = data.get("length") or 0
        fallback = (length / 1000.0) / 50.0 * 60.0 if length else 1.0
        row = self._edge_row(u, v, k)
        if row is None or self.predictor is None:
            return fallback
        try:
            from src.models.lstm_travel_time import LSTMTravelTime

            if isinstance(self.predictor, LSTMTravelTime):
                profile = self._lstm_profiles.get(
                    (int(u), int(v), int(k), self.dow),
                    np.full(24, float(row["mean_travel_time_min"]), dtype=np.float32),
                )
                return max(0.01, self.predictor.predict_row(row, profile))
            pred = self.predictor.predict(pd.DataFrame([row]))
            t = float(np.asarray(pred).reshape(-1)[0])
            if not np.isfinite(t) or t <= 0:
                return fallback
            return t
        except Exception:
            return fallback

    def _cost(self, u, v, k, constraints: RouteConstraints, prev_u=None) -> float:
        t = self._predicted_time(u, v, k)
        data = self.G.get_edge_data(u, v, k) or {}
        hwy = parse_highway(data.get("highway"))
        row = self._edge_row(u, v, k)
        if constraints.avoid_highways and is_highway(hwy):
            t *= 50.0
        if constraints.minimize_delay_risk:
            std = float(row["std_travel_time_min"]) if row is not None else 0.0
            cong = float(row["congestion_ratio"]) if row is not None else 1.0
            t *= 1.0 + 0.15 * cong + 0.05 * std
        if constraints.prefer_fewer_turns and prev_u is not None:
            t += 0.35
        if not np.isfinite(t) or t <= 0:
            return 0.01
        return float(t)

    def route(
        self,
        origin,
        destination,
        constraints: RouteConstraints | None = None,
    ) -> tuple[list, float, bool]:
        constraints = constraints or RouteConstraints()
        if origin not in self.G:
            try:
                origin = int(origin)
            except (TypeError, ValueError):
                pass
        if destination not in self.G:
            try:
                destination = int(destination)
            except (TypeError, ValueError):
                pass
        if origin not in self.G or destination not in self.G:
            return [], 0.0, False
        if origin == destination:
            return [origin], 0.0, True

        def weight(u, v, data):
            keys = list(self.G[u][v].keys())
            k = keys[0]
            return self._cost(u, v, k, constraints)

        try:
            path = nx.shortest_path(self.G, origin, destination, weight=weight)
        except Exception:
            try:
                path = nx.shortest_path(self.G, origin, destination, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return [], 0.0, False
        total = 0.0
        prev = None
        for a, b in zip(path[:-1], path[1:]):
            k = list(self.G[a][b].keys())[0]
            total += self._cost(a, b, k, constraints, prev_u=prev)
            prev = a
        ok = constraints.max_travel_time_min is None or total <= constraints.max_travel_time_min
        return path, total, ok

    def shortest_path_by_travel_time(self, origin, destination) -> tuple[list, float]:
        path, t, _ = self.route(origin, destination)
        return path, t

    def static_shortest_path_by_length(self, origin, destination) -> tuple[list, float]:
        try:
            path = nx.shortest_path(self.G, origin, destination, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], 0.0
        total_length = 0.0
        for u, v in zip(path[:-1], path[1:]):
            k = list(self.G[u][v].keys())[0]
            total_length += self.G[u][v][k].get("length", 0) or 0
        return path, (total_length / 1000.0) / 50.0 * 60.0
