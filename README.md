# RouteWise AI

**ML-based route optimization with congestion-aware travel time prediction and natural-language query support.**

RouteWise AI implements a machine learning system that predicts congestion-aware travel times and serves optimized routes via a Flask API. It uses 100K+ trip records (or synthetic equivalents), OSMnx for road networks, and a LangChain NLP layer to parse natural-language queries.

## Features

- **ML route optimization**: Predicts congestion-aware travel times using road-segment features and trained models (scikit-learn).
- **Congestion feature generation**: Road-segment congestion features built with Pandas and NetworkX.
- **Natural-language queries**: LangChain NLP/LLM layer parses queries like "shortest route from downtown to the airport" or "avoid traffic to the mall."
- **Web API**: Flask serves optimized routes and a simple web UI.

## Tech Stack

| Category        | Technologies                          |
|----------------|---------------------------------------|
| Language       | Python                                |
| Data           | Pandas, SQL, PostgreSQL               |
| Geospatial     | OSMnx, NetworkX                       |
| ML             | scikit-learn                          |
| NLP/LLM        | Hugging Face, LangChain               |
| Web            | Flask                                 |

## Results (vs static shortest-path)

- **15–25% lower error** in travel time predictions  
- **10–20% shorter** total route time in evaluation

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

Create a `.env` file (optional, for LLM and DB):

```env
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/routewise
```

### 1. Generate data and build network (first run)

```bash
# Download OSM network and generate synthetic trip records (100K+ by default)
python scripts/generate_data.py

# Optional: quick run with fewer trips (e.g. 5K) and smaller area:
# N_SYNTHETIC_TRIPS=5000 OSM_PLACE="Piedmont, California, USA" python scripts/generate_data.py

# Build congestion features and train travel-time model
python scripts/train_model.py

# Optional: run evaluation (reproduces resume metrics)
python scripts/evaluate.py
```

### 2. Run the app

```bash
python -m src.app.main
# Open http://127.0.0.1:5000
```

## Project Structure

```
RouteWise-AI/
├── src/
│   ├── data/          # Data loading, OSMnx network, trip records
│   ├── features/      # Congestion feature generation (Pandas + NetworkX)
│   ├── models/        # Travel-time prediction (scikit-learn)
│   ├── routing/       # Congestion-aware shortest path
│   ├── nlp/           # LangChain natural-language query parsing
│   └── app/           # Flask API and web UI
├── scripts/           # generate_data.py, train_model.py
├── data/              # Generated networks and trip data (gitignored)
├── models/            # Saved sklearn models (gitignored)
└── requirements.txt
```

## API Examples

- `GET /api/route?origin=...&destination=...` — Optimized route (congestion-aware).
- `POST /api/query` — Natural-language: `{"query": "from downtown to airport avoiding traffic"}`.

## License

MIT.
