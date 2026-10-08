from src.features.build import TABULAR_FEATURES, TARGET, build_segment_congestion_features
from tests.conftest import tiny_graph, tiny_trips


def test_feature_rows_have_time_and_neighbors(tmp_path):
    G = tiny_graph()
    trips = tiny_trips()
    out = tmp_path / "feat.parquet"
    feats = build_segment_congestion_features(G, trips, output_path=out)
    assert TARGET in feats.columns
    for col in TABULAR_FEATURES:
        assert col in feats.columns
    assert (feats["hour"] >= 0).all()
    assert feats["neighbor_congestion"].notna().all()
    assert out.exists()
