"""Download OSM network and generate synthetic trip records (100K+)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import N_SYNTHETIC_TRIPS, OSM_PLACE
from src.data import get_network, generate_synthetic_trips


def main():
    print(f"Fetching OSM network for: {OSM_PLACE}")
    G = get_network()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print(f"Generating {N_SYNTHETIC_TRIPS:,} synthetic trip records...")
    df = generate_synthetic_trips(G, n_trips=N_SYNTHETIC_TRIPS)
    print(f"Wrote {len(df):,} trips to data/trips.parquet")
    print("Done. Run: python scripts/train_model.py")


if __name__ == "__main__":
    main()


