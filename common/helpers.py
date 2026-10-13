"""Small printing helpers so tutorial output is readable."""
from __future__ import annotations


def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 8)
    print(f"\n{line}\n  {title}\n{line}")


def show(df, n: int = 20, title: str | None = None) -> None:
    if title:
        print(f"\n-- {title}")
    df.show(n, truncate=False)


def section(title: str) -> None:
    print(f"\n### {title}")
