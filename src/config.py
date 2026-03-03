"""Configuration for RouteWise AI."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# OSM / Network
OSM_PLACE = os.getenv("OSM_PLACE", "San Francisco, California, USA")
NETWORK_PATH = DATA_DIR / "network.graphml"
TRIPS_PATH = DATA_DIR / "trips.parquet"
SEGMENT_FEATURES_PATH = DATA_DIR / "segment_features.parquet"

# Model
TRAVEL_TIME_MODEL_PATH = MODELS_DIR / "travel_time_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Data generation
N_SYNTHETIC_TRIPS = int(os.getenv("N_SYNTHETIC_TRIPS", "100000"))  # 100K+ for resume alignment
