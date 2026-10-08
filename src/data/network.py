"""OSMnx road network loading and caching."""
from pathlib import Path

import networkx as nx

from src.config import NETWORK_PATH, OSM_PLACE


def _normalize_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    mapping = {}
    for n in G.nodes():
        try:
            mapping[n] = int(n)
        except (TypeError, ValueError):
            mapping[n] = n
    if any(k != v for k, v in mapping.items()):
        G = nx.relabel_nodes(G, mapping)
    if G.number_of_nodes() and not nx.is_weakly_connected(G):
        keep = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(keep).copy()
    return G


def get_network(place: str | None = None, use_drive: bool = True) -> nx.MultiDiGraph:
    import osmnx as ox

    place = place or OSM_PLACE
    if NETWORK_PATH.exists():
        G = ox.load_graphml(NETWORK_PATH)
        return _normalize_graph(G)
    G = ox.graph_from_place(place, network_type="drive" if use_drive else "all")
    ox.save_graphml(G, NETWORK_PATH)
    return _normalize_graph(G)


def load_network(path: Path | None = None) -> nx.MultiDiGraph:
    import osmnx as ox

    path = path or NETWORK_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Network not found at {path}. Run scripts/generate_data.py first."
        )
    return _normalize_graph(ox.load_graphml(path))
