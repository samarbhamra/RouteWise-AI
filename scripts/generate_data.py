"""Download OSM network and generate synthetic trip records (100K+)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import N_SYNTHETIC_TRIPS, OSM_PLACE
from src.data import generate_synthetic_trips, get_network
from src.data.db import init_schema


def main():
    init_schema()
    print(f"Fetching OSM network for: {OSM_PLACE}")
    G = get_network()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Generating {N_SYNTHETIC_TRIPS:,} synthetic trip records...")
    df = generate_synthetic_trips(G, n_trips=N_SYNTHETIC_TRIPS)
    print(f"Wrote {len(df):,} trips to data/trips.parquet")
    print("Next: python scripts/train_lightgbm.py && python scripts/train_lstm.py")


if __name__ == "__main__":
    main()
