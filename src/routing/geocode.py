"""Resolve place names or coordinates to graph nodes (OSMnx)."""
from typing import Tuple

import osmnx as ox

from src.config import OSM_PLACE


def geocode_place(place: str) -> Tuple[float, float] | None:
    """Return (lat, lon) for a place string, or None."""
    place = (place or "").strip()
    if not place:
        return None
    queries = [place]
    if OSM_PLACE.lower() not in place.lower():
        queries.insert(0, f"{place}, {OSM_PLACE}")
    for q in queries:
        try:
            result = ox.geocode(q)
            if result:
                return (float(result[0]), float(result[1]))
        except Exception:
            continue
    return None


def nearest_node(G, lat: float, lon: float):
    """Return nearest graph node to (lat, lon), as the graph's node id type."""
    node = ox.nearest_nodes(G, lon, lat)
    try:
        node = int(node)
    except (TypeError, ValueError):
        pass
    if node in G:
        return node
    if str(node) in G:
        return str(node)
    return node
