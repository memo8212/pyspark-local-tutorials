"""Deterministic sample-data generators (written to ./data as CSV/JSON).

Used by the I/O, SQL, Delta, streaming and ML tutorials so every script is
self-contained and reproducible.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running this file directly: python common/data.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.paths import DATA_DIR  # noqa: E402

REGIONS = ["NORTH", "SOUTH", "EAST", "WEST"]
CATEGORIES = ["Electronics", "Books", "Home", "Toys", "Grocery"]
CHANNELS = ["web", "mobile", "store"]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def customers(n: int = 200, seed: int = 1) -> Path:
    rng = random.Random(seed)
    rows = [[f"C{i:05d}", f"Customer {i}", rng.choice(REGIONS),
             rng.randint(18, 75), round(rng.uniform(20_000, 180_000), 2)]
            for i in range(1, n + 1)]
    p = DATA_DIR / "customers.csv"
    _write_csv(p, ["customer_id", "name", "region", "age", "income"], rows)
    return p


def products(n: int = 100, seed: int = 2) -> Path:
    rng = random.Random(seed)
    rows = [[f"P{i:04d}", f"Product {i}", rng.choice(CATEGORIES),
             round(rng.uniform(5, 500), 2)]
            for i in range(1, n + 1)]
    p = DATA_DIR / "products.csv"
    _write_csv(p, ["product_id", "name", "category", "unit_price"], rows)
    return p


def orders(n: int = 5000, seed: int = 3) -> Path:
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    rows = []
    for i in range(1, n + 1):
        d = start + timedelta(days=rng.randrange(540))
        rows.append([f"O{i:06d}", f"C{rng.randint(1, 200):05d}",
                     f"P{rng.randint(1, 100):04d}", d.strftime("%Y-%m-%d"),
                     rng.randint(1, 10), rng.choice(CHANNELS)])
    p = DATA_DIR / "orders.csv"
    _write_csv(p, ["order_id", "customer_id", "product_id", "order_date", "qty", "channel"], rows)
    return p


def events(n: int = 300, seed: int = 4) -> Path:
    """Newline-delimited JSON (NDJSON) events."""
    rng = random.Random(seed)
    start = date(2024, 1, 1)
    p = DATA_DIR / "events.json"
    with open(p, "w") as f:
        for i in range(1, n + 1):
            rec = {
                "event_id": i,
                "customer_id": f"C{rng.randint(1, 200):05d}",
                "event_type": rng.choice(["view", "add_to_cart", "purchase"]),
                "event_ts": (start + timedelta(days=rng.randrange(120))).strftime("%Y-%m-%d"),
                "amount": round(rng.uniform(0, 400), 2) if rng.random() > 0.6 else 0.0,
            }
            f.write(json.dumps(rec) + "\n")
    return p


def generate_all() -> dict:
    return {
        "customers": customers(),
        "products": products(),
        "orders": orders(),
        "events": events(),
    }


if __name__ == "__main__":
    paths = generate_all()
    for name, p in paths.items():
        print(f"{name:10s} -> {p}")
