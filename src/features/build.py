"""Segment-hour congestion feature pipeline (Pandas + NetworkX)."""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import SEGMENT_FEATURES_PATH, TRAIN_SNAPSHOT_PATH
from src.features.highway import parse_highway, parse_maxspeed


TABULAR_FEATURES = [
    "hour",
    "dow",
    "is_weekend",
    "peak_hour",
    "length_m",
    "highway_code",
    "maxspeed",
    "out_degree_u",
    "in_degree_v",
    "trip_count",
    "std_travel_time_min",
    "hist_mean_speed_kmh",
    "free_flow_time_min",
    "congestion_ratio",
    "rain_rate",
    "event_rate",
    "neighbor_congestion",
    "neighbor_travel_time",
]
TARGET = "mean_travel_time_min"


def _edge_table(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get("length") or 0)
        maxspeed = parse_maxspeed(data.get("maxspeed")) or 50.0
        rows.append(
            {
                "u": int(u),
                "v": int(v),
                "k": int(k),
                "length_m": length,
                "maxspeed": float(maxspeed),
                "highway_code": parse_highway(data.get("highway")),
                "out_degree_u": int(G.out_degree(u)),
                "in_degree_v": int(G.in_degree(v)),
            }
        )
    return pd.DataFrame(rows)


def _as_segment_list(segments) -> list[tuple[int, int, int]]:
    if segments is None:
        return []
    if isinstance(segments, np.ndarray):
        if segments.size == 0:
            return []
        segments = segments.tolist()
    if not isinstance(segments, (list, tuple)):
        return []
    out = []
    for item in segments:
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        out.append((int(item[0]), int(item[1]), int(item[2])))
    return out


def _edge_length(G: nx.MultiDiGraph, u: int, v: int, k: int) -> float:
    for uu, vv, kk in ((u, v, k), (str(u), str(v), k), (u, v, str(k))):
        try:
            return float(G[uu][vv][kk].get("length", 0) or 0)
        except (KeyError, TypeError):
            continue
        except Exception:
            continue
    try:
        keys = list(G[u][v].keys())
        return float(G[u][v][keys[0]].get("length", 0) or 0)
    except Exception:
        return 0.0


def _explode_segments(trips_df: pd.DataFrame, G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for row in trips_df.itertuples(index=False):
        segments = _as_segment_list(row.segments)
        total_time = float(row.travel_time_min)
        if not segments or total_time <= 0:
            continue
        lengths = [_edge_length(G, u, v, k) for u, v, k in segments]
        total_len = sum(lengths)
        if total_len <= 0:
            continue
        for (u, v, k), length in zip(segments, lengths):
            rows.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "k": int(k),
                    "travel_time_min": total_time * (length / total_len),
                    "length_m": length,
                    "hour": int(row.hour),
                    "dow": int(row.dow),
                    "day_index": int(row.day_index),
                    "rain": int(row.rain),
                    "event": int(row.event),
                }
            )
    return pd.DataFrame(rows)


def _neighbor_maps(G: nx.MultiDiGraph) -> dict[tuple, list[tuple]]:
    mapping = {}
    for u, v, k in G.edges(keys=True):
        nbrs = []
        for pred in G.predecessors(u):
            for pk in G[pred][u]:
                nbrs.append((int(pred), int(u), int(pk)))
        for succ in G.successors(v):
            for sk in G[v][succ]:
                nbrs.append((int(v), int(succ), int(sk)))
        mapping[(int(u), int(v), int(k))] = nbrs
    return mapping


def build_segment_congestion_features(
    G: nx.MultiDiGraph,
    trips_df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build segment-hour features with neighbor and weather/event aggregates."""
    output_path = output_path or SEGMENT_FEATURES_PATH
    edge_df = _edge_table(G)
    exploded = _explode_segments(trips_df, G)
    if exploded.empty:
        raise ValueError("No segment-level trip data; check trips and graph.")

    agg = exploded.groupby(["u", "v", "k", "hour", "dow"], as_index=False).agg(
        trip_count=("travel_time_min", "count"),
        mean_travel_time_min=("travel_time_min", "mean"),
        std_travel_time_min=("travel_time_min", "std"),
        rain_rate=("rain", "mean"),
        event_rate=("event", "mean"),
        mean_day_index=("day_index", "mean"),
    )
    agg["std_travel_time_min"] = agg["std_travel_time_min"].fillna(0)

    features = agg.merge(edge_df, on=["u", "v", "k"], how="left")
    features["is_weekend"] = (features["dow"] >= 5).astype(int)
    features["peak_hour"] = features["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    features["free_flow_time_min"] = (
        (features["length_m"] / 1000.0) / features["maxspeed"].replace(0, np.nan) * 60.0
    ).fillna(0.1)
    features["congestion_ratio"] = (
        features["mean_travel_time_min"] / features["free_flow_time_min"].replace(0, np.nan)
    ).fillna(1.0)
    hours = features["mean_travel_time_min"].replace(0, np.nan)
    features["hist_mean_speed_kmh"] = (
        (features["length_m"] / 1000.0) / (hours / 60.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(features["maxspeed"])

    lookup = features.set_index(["u", "v", "k", "hour", "dow"])
    nbrs = _neighbor_maps(G)
    n_cong, n_tt = [], []
    for rec in features.itertuples(index=False):
        keys = nbrs.get((int(rec.u), int(rec.v), int(rec.k)), [])
        cong_vals, tt_vals = [], []
        for nu, nv, nk in keys:
            try:
                cell = lookup.loc[(nu, nv, nk, rec.hour, rec.dow)]
                if isinstance(cell, pd.DataFrame):
                    cell = cell.iloc[0]
                cong_vals.append(float(cell["congestion_ratio"]))
                tt_vals.append(float(cell["mean_travel_time_min"]))
            except KeyError:
                continue
        n_cong.append(float(np.mean(cong_vals)) if cong_vals else float(rec.congestion_ratio))
        n_tt.append(float(np.mean(tt_vals)) if tt_vals else float(rec.mean_travel_time_min))
    features["neighbor_congestion"] = n_cong
    features["neighbor_travel_time"] = n_tt

    features.to_parquet(output_path, index=False)
    try:
        from src.data.db import save_features_to_db

        save_features_to_db(features)
    except Exception:
        pass
    return features


def time_aware_split(features: pd.DataFrame, train_frac: float = 0.8):
    """Split by mean_day_index so later days are held out."""
    df = features.sort_values("mean_day_index").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    train.to_parquet(TRAIN_SNAPSHOT_PATH, index=False)
    return train, test
