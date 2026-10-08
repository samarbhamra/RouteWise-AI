"""Configuration for RouteWise AI."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

OSM_PLACE = os.getenv("OSM_PLACE", "Piedmont, California, USA")
NETWORK_PATH = DATA_DIR / "network.graphml"
TRIPS_PATH = DATA_DIR / "trips.parquet"
SEGMENT_FEATURES_PATH = DATA_DIR / "segment_features.parquet"
TRAIN_SNAPSHOT_PATH = DATA_DIR / "train_feature_snapshot.parquet"
EVAL_REPORT_PATH = DATA_DIR / "eval_report.json"
LSTM_SEQ_PATH = DATA_DIR / "lstm_sequences.npz"

LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_travel_time.txt"
LIGHTGBM_META_PATH = MODELS_DIR / "lightgbm_meta.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_travel_time.pt"
LSTM_META_PATH = MODELS_DIR / "lstm_meta.joblib"

# Backward-compatible aliases
TRAVEL_TIME_MODEL_PATH = LIGHTGBM_MODEL_PATH
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_ZERO_SHOT_MODEL = os.getenv("HF_ZERO_SHOT_MODEL", "typeform/distilbert-base-uncased-mnli")
USE_HF_PARSER = os.getenv("USE_HF_PARSER", "0") == "1"
DATABASE_URL = os.getenv("DATABASE_URL", "")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)

N_SYNTHETIC_TRIPS = int(os.getenv("N_SYNTHETIC_TRIPS", "100000"))
LSTM_SEQ_LEN = int(os.getenv("LSTM_SEQ_LEN", "12"))
DEFAULT_ROUTE_MODEL = os.getenv("DEFAULT_ROUTE_MODEL", "lightgbm")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
