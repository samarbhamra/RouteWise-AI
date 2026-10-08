# RouteWise AI

**Multi-model traffic intelligence and congestion-aware routing.**

RouteWise predicts **road-segment travel times** with LightGBM (tabular) and a PyTorch LSTM (hourly sequences + neighbor features), then routes on OSMnx/NetworkX using those predictions as edge costs. A LangChain / Hugging Face layer turns natural-language constraints into routing objectives. Experiments are logged with MLflow; inference is served with FastAPI.

## Stack

Python, Pandas, LightGBM, PyTorch, MLflow, OSMnx, NetworkX, LangChain, Hugging Face Transformers, FastAPI, SQL / PostgreSQL (optional), Docker.

## Pipeline

1. `scripts/generate_data.py` — OSM network + 100K+ synthetic trips (hour, day of week, rain/event proxies).
2. Feature table — segment-hour rows: time, road class, historical speed, congestion, neighbor conditions (`src/features/build.py`).
3. `scripts/train_lightgbm.py` / `scripts/train_lstm.py` — time-aware split, MAE/RMSE vs free-flow, MLflow.
4. Routing — variants **A static**, **B LightGBM**, **C LSTM**, plus constraints (avoid highways, fewer turns, delay risk, max time).
5. `scripts/evaluate.py` — model metrics + route-level A/B/C (writes `data/eval_report.json`).
6. FastAPI — `/route`, `/query`, `/drift`, `/health`.

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

Full 100K trips: omit `N_SYNTHETIC_TRIPS` or set `100000`. Override place with `OSM_PLACE`.

### Optional env (`.env`)

```
OPENAI_API_KEY=
USE_HF_PARSER=0
HF_ZERO_SHOT_MODEL=typeform/distilbert-base-uncased-mnli
DATABASE_URL=postgresql://user:pass@localhost:5432/routewise
MLFLOW_TRACKING_URI=file:./mlruns
```

Hugging Face zero-shot constraint tagging: `USE_HF_PARSER=1` (downloads the MNLI model on first use).

### Tests

```bash
pytest tests/ -q
```

### Docker

```bash
docker compose up --build
```

Needs `data/` and `models/` from local train (mounted in compose).

## API

- `POST /route` — origin/destination, `model` (`static` | `lightgbm` | `lstm`), constraints.
- `GET /api/route?...` — same for the UI.
- `POST /query` — `{ "query": "from downtown to the airport avoid highways within 30 min" }`.
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
src/app/           FastAPI + UI
scripts/           generate, train, evaluate
tests/             unit tests (tiny graph, no live OSM)
```
