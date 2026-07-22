"""Shared helpers for subprocess-driven tests (demo pipeline, CLI)."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "examples" / "demo_pipeline.py"


def _env(store, extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["FLOWR_DIR"] = str(store)
    env.update(extra or {})
    return env


def run_demo(store, *args, extra_env=None, cwd=None, demo=DEMO):
    return subprocess.run(
        [sys.executable, str(demo), *map(str, args)],
        env=_env(store, extra_env), capture_output=True, text=True, cwd=cwd,
    )


def run_cli(store, *args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "flowr", *map(str, args)],
        env=_env(store), capture_output=True, text=True, cwd=cwd,
    )


def counter_lines(path):
    p = Path(path)
    if not p.exists():
        return 0
    return p.read_bytes().count(b"\n")


def demo_node_count(n):
    # generate + clean + analyze per item, plus aggregate and figure
    return 3 * n + 2
