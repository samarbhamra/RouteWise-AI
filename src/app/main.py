"""Flask API and web UI for RouteWise AI."""
import json
from pathlib import Path

import flask

# Add project root for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import OPENAI_API_KEY
from src.data import load_network
from src.nlp import parse_route_query
from src.routing.geocode import geocode_place, nearest_node
from src.routing.optimizer import RouteOptimizer

app = flask.Flask(__name__, static_folder=None)
app.config["JSONIFY_PRETTY_PRINT_REGULAR"] = True

# Lazy-loaded globals
_network = None
_optimizer = None


def get_network():
    global _network
    if _network is None:
        _network = load_network()
    return _network


def get_optimizer():
    global _optimizer
    if _optimizer is None:
        G = get_network()
        _optimizer = RouteOptimizer(G)
    return _optimizer


def resolve_origin_dest(origin=None, destination=None, origin_lat=None, origin_lon=None, dest_lat=None, dest_lon=None):
    """Resolve origin and destination to graph node IDs."""
    G = get_network()
    o_node, d_node = None, None

    if origin_lat is not None and origin_lon is not None:
        o_node = nearest_node(G, float(origin_lat), float(origin_lon))
    elif origin:
        coords = geocode_place(origin.strip())
        if coords:
            o_node = nearest_node(G, coords[0], coords[1])

    if dest_lat is not None and dest_lon is not None:
        d_node = nearest_node(G, float(dest_lat), float(dest_lon))
    elif destination:
        coords = geocode_place(destination.strip())
        if coords:
            d_node = nearest_node(G, coords[0], coords[1])

    return o_node, d_node


@app.route("/")
def index():
    """Serve simple web UI."""
    return flask.render_template("index.html", has_openai=bool(OPENAI_API_KEY))


@app.route("/api/route", methods=["GET"])
def api_route():
    """
    Get congestion-aware optimized route.
    Query params: origin, destination (place names) OR origin_lat, origin_lon, dest_lat, dest_lon.
    """
    origin = flask.request.args.get("origin", "").strip()
    destination = flask.request.args.get("destination", "").strip()
    origin_lat = flask.request.args.get("origin_lat")
    origin_lon = flask.request.args.get("origin_lon")
    dest_lat = flask.request.args.get("dest_lat")
    dest_lon = flask.request.args.get("dest_lon")

    o_node, d_node = resolve_origin_dest(
        origin=origin or None,
        destination=destination or None,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
    )

    if o_node is None or d_node is None:
        return flask.jsonify({"error": "Could not resolve origin or destination"}), 400

    opt = get_optimizer()
    path, time_min = opt.shortest_path_by_travel_time(o_node, d_node)
    _, static_time_min = opt.static_shortest_path_by_length(o_node, d_node)

    if not path:
        return flask.jsonify({"error": "No route found"}), 404

    G = get_network()
    # Build list of (lat, lon) for the path
    coords = []
    for node in path:
        n = G.nodes[node]
        lat = n.get("y")
        lon = n.get("x")
        if lat is not None and lon is not None:
            coords.append({"lat": lat, "lon": lon})

    return flask.jsonify({
        "path": coords,
        "path_node_ids": path,
        "travel_time_min": round(time_min, 2),
        "static_baseline_time_min": round(static_time_min, 2),
        "origin_node": o_node,
        "destination_node": d_node,
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Natural-language route query (LangChain NLP).
    Body: {"query": "from downtown to the airport avoiding traffic"}
    """
    data = flask.request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return flask.jsonify({"error": "Missing 'query' in body"}), 400

    parsed = parse_route_query(query)
    origin = parsed.get("origin", "")
    destination = parsed.get("destination", "")

    if not origin or not destination:
        return flask.jsonify({
            "error": "Could not parse origin/destination from query",
            "parsed": parsed,
        }), 400

    o_node, d_node = resolve_origin_dest(origin=origin, destination=destination)
    if o_node is None or d_node is None:
        return flask.jsonify({
            "error": "Could not geocode origin or destination",
            "origin": origin,
            "destination": destination,
        }), 400

    opt = get_optimizer()
    path, time_min = opt.shortest_path_by_travel_time(o_node, d_node)
    _, static_time_min = opt.static_shortest_path_by_length(o_node, d_node)

    if not path:
        return flask.jsonify({"error": "No route found"}), 404

    G = get_network()
    coords = []
    for node in path:
        n = G.nodes[node]
        lat, lon = n.get("y"), n.get("x")
        if lat is not None and lon is not None:
            coords.append({"lat": lat, "lon": lon})

    return flask.jsonify({
        "parsed": parsed,
        "path": coords,
        "path_node_ids": path,
        "travel_time_min": round(time_min, 2),
        "static_baseline_time_min": round(static_time_min, 2),
    })


def main():
    from flask_cors import CORS
    CORS(app)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
