#!/usr/bin/env python3
"""Run every tutorial script in order, each in its own process.

    python run_all.py                 # generate data + run all tutorials
    python run_all.py 01 05 09        # run only tutorials whose name starts with these

Exits non-zero if any tutorial fails.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TUT = REPO / "tutorials"


def main() -> int:
    prefixes = sys.argv[1:]
    scripts = sorted(TUT.glob("[0-9][0-9]_*.py"))
    if prefixes:
        scripts = [s for s in scripts if any(s.name.startswith(p) for p in prefixes)]

    # ensure sample data exists
    subprocess.run([sys.executable, str(REPO / "common" / "data.py")], check=True)

    failures: list[str] = []
    for s in scripts:
        print("\n" + "#" * 78)
        print(f"# RUNNING {s.name}")
        print("#" * 78, flush=True)
        rc = subprocess.run([sys.executable, str(s)], cwd=str(REPO)).returncode
        if rc != 0:
            failures.append(s.name)

    print("\n" + "=" * 78)
    print(f"Ran {len(scripts)} tutorial(s).")
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
