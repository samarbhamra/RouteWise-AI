# RouteWise AI

**Multi-model traffic intelligence and congestion-aware routing.**

RouteWise predicts **road-segment travel times** with LightGBM (tabular) and a PyTorch LSTM (hourly sequences + neighbor features), then routes on OSMnx/NetworkX using those predictions as edge costs. A LangChain / Hugging Face layer turns natural-language constraints into routing objectives. Experiments are logged with MLflow; inference is served with FastAPI and a Leaflet map (street, satellite, and predicted-congestion overlay).

## Stack

Python, Pandas, LightGBM, PyTorch, MLflow, OSMnx, NetworkX, LangChain, Hugging Face Transformers, FastAPI, Leaflet, SQL / PostgreSQL (optional), Docker.

On macOS, if LightGBM cannot load OpenMP (`libomp`), training **falls back** to scikit-learn `HistGradientBoostingRegressor` (same boosting family). Install `brew install libomp` to use native LightGBM.

## Pipeline

1. `scripts/generate_data.py` — OSM network + 100K+ synthetic trips (hour, day of week, rain/event proxies).
2. Feature table — segment-hour rows: time, road class, historical speed, congestion, neighbor conditions (`src/features/build.py`).
3. `scripts/train_lightgbm.py` / `scripts/train_lstm.py` — time-aware split, MAE/RMSE vs free-flow, MLflow (SQLite tracking DB by default).
4. Routing — variants **A static**, **B LightGBM**, **C LSTM**, plus constraints (avoid highways, fewer turns, delay risk, max time).
5. `scripts/evaluate.py` — model metrics + route-level A/B/C (writes `data/eval_report.json`).
6. FastAPI — `/route`, `/query`, `/api/traffic`, `/drift`, `/health`, plus the map UI.

Resume-style lifts (**~15–25% lower MAE**, **~10–20% shorter route time**) are **outputs of evaluation**, not hardcoded.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Laptop-friendly first run (smaller graph/trips):

```bash
export OSM_PLACE="Piedmont, California, USA"
export N_SYNTHETIC_TRIPS=5000
python scripts/generate_data.py
python scripts/train_lightgbm.py
python scripts/train_lstm.py
python scripts/evaluate.py
python -m src.app.main
# http://127.0.0.1:8000
```

Use places **inside the downloaded map** (default Piedmont, CA), e.g. `Highland Ave, Piedmont, CA` → `Piedmont Park, Piedmont, CA`. Vague names like “downtown” / “airport” often geocode outside the graph.

Full 100K trips: omit `N_SYNTHETIC_TRIPS` or set `100000`. Override place with `OSM_PLACE`.

### Optional env (`.env`)

```
OPENAI_API_KEY=
USE_HF_PARSER=0
HF_ZERO_SHOT_MODEL=typeform/distilbert-base-uncased-mnli
DATABASE_URL=postgresql://user:pass@localhost:5432/routewise
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

- Hugging Face zero-shot constraint tagging: `USE_HF_PARSER=1` (downloads the MNLI model on first use).
- Postgres is optional; the default store is Parquet under `data/`.
- MLflow uses SQLite (`mlflow.db`) unless you override the tracking URI.

### Tests

```bash
pytest tests/ -q
```

### Docker

```bash
docker compose up --build
```

Needs `data/` and `models/` from local train (mounted in compose).

## Map UI

At `http://127.0.0.1:8000`:

- **Street** — OpenStreetMap tiles (Leaflet; not Google Maps).
- **Satellite** — Esri World Imagery.
- **Traffic** — overlay of **predicted** segment congestion (green → red), not live Google traffic.
- Successful routes draw as a polyline with origin/destination markers.

## API

- `POST /route` — origin/destination, `model` (`static` | `lightgbm` | `lstm`), constraints.
- `GET /api/route?...` — same for the UI.
- `POST /query` — `{ "query": "from Highland Ave to Piedmont Park avoid highways" }`.
- `GET /api/traffic` — segment congestion for the traffic overlay.
- `GET /drift` — PSI / KS vs training snapshot.
- `GET /health`

## Project layout

```
src/data/          OSM + trips + optional Postgres
src/features/      segment-hour features (Pandas + NetworkX neighbors)
src/models/        LightGBM, LSTM, MLflow helpers
src/routing/       constraint-aware shortest path
src/nlp/           LangChain / Hugging Face / regex parser
src/monitoring/    drift (PSI, KS)
src/app/           FastAPI + Leaflet UI
scripts/           generate, train, evaluate
tests/             unit tests (tiny graph, no live OSM)
```
