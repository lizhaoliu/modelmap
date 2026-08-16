"""Shared helpers for the Playwright acceptance suites.

Run against a live server:  BASE=http://127.0.0.1:7860 uv run python tests/e2e/test_explore.py
Screenshots go to $SHOTS (default: tests/e2e/shots, git-ignored).
"""
import os
import sys

BASE = os.environ.get("BASE", "http://127.0.0.1:7860")
SHOTS = os.environ.get("SHOTS", os.path.join(os.path.dirname(__file__), "shots"))
os.makedirs(SHOTS, exist_ok=True)

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(name)


def finish() -> None:
    print("\n" + ("ALL PASS" if not _failures else f"FAILURES: {_failures}"))
    sys.exit(1 if _failures else 0)
