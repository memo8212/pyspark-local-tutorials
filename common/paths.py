"""Shared helpers for the local PySpark tutorials."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
WAREHOUSE_DIR = REPO_ROOT / "spark-warehouse"

for _d in (DATA_DIR, OUTPUT_DIR, WAREHOUSE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
