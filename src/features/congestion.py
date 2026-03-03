"""Road-segment congestion feature generation using Pandas and NetworkX."""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import SEGMENT_FEATURES_PATH


def _segment_travel_time_per_trip(trip_row, G):
    """For one trip, estimate per-segment travel time (proportional to length)."""
    path_nodes = trip_row["path_nodes"]
    segments = trip_row["segments"]
    total_time = trip_row["travel_time_min"]
    if not segments or total_time <= 0:
        return []
    lengths = []
    for u, v, k in segments:
        length = G[u][v][k].get("length", 0) or 0
        lengths.append(length)
    total_len = sum(lengths)
    if total_len <= 0:
        return []
    out = []
    for (u, v, k), length in zip(segments, lengths):
        seg_time = total_time * (length / total_len)
        out.append({"u": u, "v": v, "k": k, "travel_time_min": seg_time, "length_m": length})
    return out


def build_segment_congestion_features(
    G: nx.MultiDiGraph,
    trips_df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build road-segment congestion features using Pandas and NetworkX.
    - Graph-based: degree, length, maxspeed from G.
    - Trip-based: trip count, mean travel time, std, congestion ratio (actual vs free-flow).
    """
    output_path = output_path or SEGMENT_FEATURES_PATH

    # 1) Segment-level trip stats (Pandas)
    rows = []
    for _, row in trips_df.iterrows():
        rows.extend(_segment_travel_time_per_trip(row, G))
    seg_trips = pd.DataFrame(rows)

    if seg_trips.empty:
        raise ValueError("No segment-level trip data; check trips and graph.")

    agg = seg_trips.groupby(["u", "v", "k"]).agg(
        trip_count=("travel_time_min", "count"),
        mean_travel_time_min=("travel_time_min", "mean"),
        std_travel_time_min=("travel_time_min", "std"),
        mean_length_m=("length_m", "mean"),
    ).reset_index()
    agg["std_travel_time_min"] = agg["std_travel_time_min"].fillna(0)

    # 2) Graph-based features (NetworkX)
    edge_list = []
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length") or 0
        maxspeed = data.get("maxspeed")
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0] if maxspeed else None
        if maxspeed is not None and isinstance(maxspeed, str):
            try:
                maxspeed = float(maxspeed.replace(" mph", "").replace(" km/h", "").strip())
            except ValueError:
                maxspeed = None
        out_deg = G.out_degree(u)
        in_deg = G.in_degree(v)
        edge_list.append({
            "u": u, "v": v, "k": k,
            "length_m": length,
            "maxspeed": maxspeed,
            "out_degree_u": out_deg,
            "in_degree_v": in_deg,
        })
    edge_df = pd.DataFrame(edge_list)

    # 3) Merge and add derived features
    features = edge_df.merge(
        agg, on=["u", "v", "k"], how="left"
    )
    features["trip_count"] = features["trip_count"].fillna(0).astype(int)
    features["mean_travel_time_min"] = features["mean_travel_time_min"].fillna(0)
    features["std_travel_time_min"] = features["std_travel_time_min"].fillna(0)

    # Free-flow time (length / speed); assume 50 km/h if maxspeed missing
    speed_kmh = features["maxspeed"].fillna(50.0)
    features["free_flow_time_min"] = (features["length_m"] / 1000.0) / speed_kmh * 60.0
    features["congestion_ratio"] = (
        features["mean_travel_time_min"] / features["free_flow_time_min"].replace(0, np.nan)
    ).fillna(1.0)

    features.to_parquet(output_path, index=False)
    return features
