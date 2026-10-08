"""FastAPI inference service and web UI."""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT, DEFAULT_ROUTE_MODEL, OPENAI_API_KEY, OSM_PLACE
from src.data import load_network
from src.monitoring.drift import compute_drift
from src.nlp import constraints_from_parsed, parse_route_query
from src.routing.geocode import geocode_place, nearest_node
from src.routing.optimizer import RouteConstraints, RouteOptimizer

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="RouteWise AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_network = None
_optimizers: dict = {}


def get_network():
    global _network
    if _network is None:
        _network = load_network()
    return _network


def get_optimizer(variant: str, hour: int = 8, dow: int = 2) -> RouteOptimizer:
    key = (variant, hour, dow)
    if key not in _optimizers:
        _optimizers[key] = RouteOptimizer(get_network(), variant=variant, hour=hour, dow=dow)
    return _optimizers[key]


def resolve_origin_dest(
    origin=None,
    destination=None,
    origin_lat=None,
    origin_lon=None,
    dest_lat=None,
    dest_lon=None,
):
    G = get_network()
    o_node = d_node = None
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


def path_coords(path):
    G = get_network()
    coords = []
    for node in path:
        n = G.nodes[node]
        lat, lon = n.get("y"), n.get("x")
        if lat is not None and lon is not None:
            coords.append({"lat": lat, "lon": lon})
    return coords


class RouteBody(BaseModel):
    origin: str | None = None
    destination: str | None = None
    origin_lat: float | None = None
    origin_lon: float | None = None
    dest_lat: float | None = None
    dest_lon: float | None = None
    model: str = Field(default=DEFAULT_ROUTE_MODEL)
    hour: int = 8
    dow: int = 2
    avoid_highways: bool = False
    prefer_fewer_turns: bool = False
    minimize_delay_risk: bool = False
    max_travel_time_min: float | None = None


class QueryBody(BaseModel):
    query: str
    model: str = Field(default=DEFAULT_ROUTE_MODEL)
    hour: int = 8
    dow: int = 2


def _run_route(o_node, d_node, variant: str, constraints: RouteConstraints, hour: int, dow: int):
    if o_node is None or d_node is None:
        raise HTTPException(
            400,
            "Could not geocode origin/destination. Use a place inside the map area "
            f"(default: {OSM_PLACE}), e.g. a street name plus the city.",
        )
    if o_node == d_node:
        raise HTTPException(
            400,
            "Origin and destination snapped to the same intersection. Use more specific addresses.",
        )
    opt = get_optimizer(variant, hour, dow)
    path, time_min, feasible = opt.route(o_node, d_node, constraints)
    _, static_time = opt.static_shortest_path_by_length(o_node, d_node)
    if not path:
        raise HTTPException(
            404,
            "No path between those points on this road network. Try two streets in "
            f"{OSM_PLACE}.",
        )
    return {
        "path": path_coords(path),
        "path_node_ids": path,
        "model": variant,
        "travel_time_min": round(time_min, 2),
        "static_baseline_time_min": round(static_time, 2),
        "constraints_satisfied": feasible,
        "origin_node": o_node,
        "destination_node": d_node,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    html = (TEMPLATE_DIR / "index.html").read_text()
    html = html.replace("{% if not has_openai %}", "").replace("{% endif %}", "")
    if OPENAI_API_KEY:
        html = html.replace(
            "Regex parser is default. Set OPENAI_API_KEY or USE_HF_PARSER=1 for LLM/Hugging Face parsing.",
            "OpenAI parsing is enabled.",
        )
    return HTMLResponse(html)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/traffic")
def traffic_overlay(hour: int = 8, dow: int = 2):
    """Predicted congestion by road segment (not live Google traffic)."""
    import pandas as pd

    from src.config import SEGMENT_FEATURES_PATH

    G = get_network()
    if not SEGMENT_FEATURES_PATH.exists():
        return {"segments": []}
    df = pd.read_parquet(SEGMENT_FEATURES_PATH)
    sub = df[(df["hour"] == hour) & (df["dow"] == dow)]
    if sub.empty:
        sub = df
    agg = sub.groupby(["u", "v", "k"], as_index=False)["congestion_ratio"].mean()
    segments = []
    for rec in agg.itertuples(index=False):
        u, v = int(rec.u), int(rec.v)
        if u not in G.nodes or v not in G.nodes:
            continue
        a, b = G.nodes[u], G.nodes[v]
        if a.get("y") is None or b.get("y") is None:
            continue
        segments.append(
            {
                "coords": [[float(a["y"]), float(a["x"])], [float(b["y"]), float(b["x"])]],
                "congestion": float(rec.congestion_ratio),
            }
        )
        if len(segments) >= 2500:
            break
    return {"segments": segments}


@app.get("/drift")
def drift():
    return compute_drift()


@app.post("/route")
@app.post("/api/route")
def post_route(body: RouteBody):
    o, d = resolve_origin_dest(
        body.origin, body.destination, body.origin_lat, body.origin_lon, body.dest_lat, body.dest_lon
    )
    constraints = RouteConstraints(
        avoid_highways=body.avoid_highways,
        prefer_fewer_turns=body.prefer_fewer_turns,
        minimize_delay_risk=body.minimize_delay_risk,
        max_travel_time_min=body.max_travel_time_min,
    )
    return _run_route(o, d, body.model, constraints, body.hour, body.dow)


@app.get("/api/route")
def get_route(
    origin: str = "",
    destination: str = "",
    model: str = Query(default=DEFAULT_ROUTE_MODEL),
    hour: int = 8,
    dow: int = 2,
    avoid_highways: bool = False,
    prefer_fewer_turns: bool = False,
    minimize_delay_risk: bool = False,
):
    o, d = resolve_origin_dest(origin or None, destination or None)
    constraints = RouteConstraints(
        avoid_highways=avoid_highways,
        prefer_fewer_turns=prefer_fewer_turns,
        minimize_delay_risk=minimize_delay_risk,
    )
    return _run_route(o, d, model, constraints, hour, dow)


@app.post("/query")
@app.post("/api/query")
def query_route(body: QueryBody):
    parsed = parse_route_query(body.query)
    if not parsed.get("origin") or not parsed.get("destination"):
        raise HTTPException(400, detail={"error": "Could not parse origin/destination", "parsed": parsed})
    o, d = resolve_origin_dest(parsed["origin"], parsed["destination"])
    constraints = constraints_from_parsed(parsed)
    result = _run_route(o, d, body.model, constraints, body.hour, body.dow)
    result["parsed"] = parsed
    return result


def main():
    import os
    import uvicorn

    reload_on = os.getenv("UVICORN_RELOAD", "0") == "1"
    uvicorn.run(
        "src.app.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=reload_on,
        reload_excludes=[".venv/*", ".venv/**", "mlruns/*", "mlflow.db"],
    )


if __name__ == "__main__":
    main()
