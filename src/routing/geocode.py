"""Resolve place names or coordinates to graph nodes (OSMnx)."""
from typing import Tuple

import osmnx as ox


def geocode_place(place: str) -> Tuple[float, float] | None:
    """Return (lat, lon) for a place string, or None."""
    try:
        result = ox.geocode(place)
        if result:
            return (float(result[0]), float(result[1]))
    except Exception:
        pass
    return None


def nearest_node(G, lat: float, lon: float):
    """Return nearest graph node to (lat, lon)."""
    return ox.nearest_nodes(G, lon, lat)
