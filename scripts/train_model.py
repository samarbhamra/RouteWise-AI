"""Deprecated entry: trains LightGBM (use scripts/train_lightgbm.py)."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_lightgbm.py")), run_name="__main__")
