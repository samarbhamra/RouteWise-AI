"""MLflow helpers (SQLite tracking; never fail training if logging breaks)."""
import os
from contextlib import contextmanager

from src.config import MLFLOW_TRACKING_URI

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


@contextmanager
def start_run(experiment: str, run_name: str):
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
        yield None
