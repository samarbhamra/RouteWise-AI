from src.routing.optimizer import RouteConstraints, RouteOptimizer
from tests.conftest import tiny_graph


def test_static_route_exists():
    G = tiny_graph()
    opt = RouteOptimizer(G, segment_features=None, variant="static")
    path, t = opt.static_shortest_path_by_length(1, 3)
    assert path[0] == 1 and path[-1] == 3
    assert t > 0


def test_avoid_highways_prefers_surface_streets():
    G = tiny_graph()
    opt = RouteOptimizer(G, segment_features=None, variant="static")
    path, _, ok = opt.route(1, 3, RouteConstraints(avoid_highways=True))
    assert path == [1, 2, 3]
    assert ok is True
