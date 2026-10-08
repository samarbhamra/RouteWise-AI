"""Trip record loading and synthetic trip generation."""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import N_SYNTHETIC_TRIPS, TRIPS_PATH
from src.features.highway import parse_highway


def load_trips(path: Path | None = None) -> pd.DataFrame:
    path = path or TRIPS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trips not found at {path}. Run scripts/generate_data.py first."
        )
    return pd.read_parquet(path)


def _peak_factor(hour: int) -> float:
    if hour in (7, 8, 9, 16, 17, 18):
        return 1.45
    if hour in (10, 11, 15, 19):
        return 1.2
    if 0 <= hour <= 5:
        return 0.85
    return 1.0


def generate_synthetic_trips(
    G,
    n_trips: int | None = None,
    output_path: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic trips with timestamps, weather/event proxies, and paths.
    Default n_trips is 100K+ for resume-aligned scale.
    """
    n_trips = n_trips or N_SYNTHETIC_TRIPS
    output_path = output_path or TRIPS_PATH
    rng = np.random.default_rng(seed)

    nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
    if len(nodes) < 2:
        raise ValueError("Graph has too few nodes with outgoing edges.")

    deg = np.array([G.out_degree(n) for n in nodes], dtype=float)
    deg = deg / deg.sum()

    records = []
    attempts = 0
    max_attempts = n_trips * 8
    while len(records) < n_trips and attempts < max_attempts:
        attempts += 1
        o, d = rng.choice(nodes, size=2, replace=False, p=deg)
        try:
            path = nx.shortest_path(G, o, d, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2:
            continue

        segments = []
        total_length = 0.0
        highway_codes = []
        for u, v in zip(path[:-1], path[1:]):
            k = list(G[u][v].keys())[0]
            data = G[u][v][k]
            seg_len = data.get("length", 0) or 0
            total_length += seg_len
            segments.append((int(u), int(v), int(k)))
            highway_codes.append(parse_highway(data.get("highway")))

        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        day_index = int(rng.integers(0, 14))
        rain = int(rng.random() < 0.18)
        event = int(rng.random() < 0.08)

        avg_speed_kmh = 32.0
        base_time_min = (total_length / 1000.0) / avg_speed_kmh * 60.0
        cong = _peak_factor(hour) * (1.0 + 0.25 * rain + 0.15 * event)
        if any(h >= 4 for h in highway_codes):
            cong *= 0.92
        travel_time_min = float(base_time_min * cong * rng.uniform(0.9, 1.15))

        records.append(
            {
                "origin_node": int(o),
                "dest_node": int(d),
                "path_nodes": [int(n) for n in path],
                "segments": segments,
                "travel_time_min": travel_time_min,
                "length_m": float(total_length),
                "hour": hour,
                "dow": dow,
                "day_index": day_index,
                "rain": rain,
                "event": event,
            }
        )

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df
