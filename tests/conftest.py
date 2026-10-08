import networkx as nx
import pandas as pd
import pytest


def tiny_graph():
    G = nx.MultiDiGraph()
    G.add_node(1, x=-122.2, y=37.8)
    G.add_node(2, x=-122.19, y=37.81)
    G.add_node(3, x=-122.18, y=37.82)
    G.add_edge(1, 2, 0, length=400.0, highway="residential", maxspeed="40")
    G.add_edge(2, 3, 0, length=500.0, highway="primary", maxspeed="50")
    G.add_edge(1, 3, 0, length=1200.0, highway="motorway", maxspeed="90")
    return G


def tiny_trips():
    return pd.DataFrame(
        [
            {
                "origin_node": 1,
                "dest_node": 3,
                "path_nodes": [1, 2, 3],
                "segments": [(1, 2, 0), (2, 3, 0)],
                "travel_time_min": 4.0,
                "length_m": 900.0,
                "hour": 8,
                "dow": 2,
                "day_index": 1,
                "rain": 0,
                "event": 0,
            },
            {
                "origin_node": 1,
                "dest_node": 3,
                "path_nodes": [1, 3],
                "segments": [(1, 3, 0)],
                "travel_time_min": 3.5,
                "length_m": 1200.0,
                "hour": 8,
                "dow": 2,
                "day_index": 2,
                "rain": 1,
                "event": 0,
            },
            {
                "origin_node": 1,
                "dest_node": 3,
                "path_nodes": [1, 2, 3],
                "segments": [(1, 2, 0), (2, 3, 0)],
                "travel_time_min": 5.5,
                "length_m": 900.0,
                "hour": 17,
                "dow": 4,
                "day_index": 10,
                "rain": 0,
                "event": 1,
            },
        ]
    )
