"""Phase 4: CLI subcommands, run as subprocesses from a foreign cwd."""

import json
import sqlite3

import pytest

from flowr.cli import _human_bytes
from helpers import run_cli, run_demo


@pytest.mark.parametrize("n,text", [
    (0, "0 B"), (1023, "1023 B"), (1024, "1.0 kB"),
    (1536, "1.5 kB"), (1024 ** 2, "1.0 MB"), (1024 ** 4, "1.0 TB"),
    (1024 ** 5, "1024.0 TB"),
])
def test_human_bytes(n, text):
    assert _human_bytes(n) == text


def _seed(store, tmp_path, n=3):
    """Populate a store by running the demo from an unrelated cwd."""
    foreign = tmp_path / "elsewhere"
    foreign.mkdir(exist_ok=True)
    p = run_demo(store, "--n", n, cwd=foreign)
    assert p.returncode == 0, p.stderr
    return foreign


def test_status(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    p = run_cli(store, "status", cwd=cwd)
    assert p.returncode == 0, p.stderr
    for name in ("generate", "clean", "analyze", "aggregate", "figure"):
        assert name in p.stdout
    assert "objects" in p.stdout


def test_runs(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    run_demo(store, "--n", 3, cwd=cwd)  # second run: all hits
    p = run_cli(store, "runs", cwd=cwd)
    assert p.returncode == 0, p.stderr
    lines = [l for l in p.stdout.splitlines() if l.startswith("#")]
    assert len(lines) == 2
    assert "ok" in p.stdout


def test_why_unique_prefix_and_provenance(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    db = sqlite3.connect(store / "index.db")
    (requested,) = db.execute(
        "SELECT requested_keys FROM runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    fig_key = json.loads(requested)[0]
    db.close()
    p = run_cli(store, "why", fig_key[:12], cwd=cwd)
    assert p.returncode == 0, p.stderr
    out = p.stdout
    assert fig_key in out
    assert "stage: figure" in out and "aggregate" in out and "generate" in out
    assert "params" in out and "code_hash" in out and "run: #" in out


def test_why_no_match_exits_3(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    p = run_cli(store, "why", "ffffffffffff", cwd=cwd)
    assert p.returncode == 3


def test_why_ambiguous_exits_1(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    p = run_cli(store, "why", "", cwd=cwd)   # empty prefix matches everything
    assert p.returncode == 1
    assert "ambiguous" in p.stderr


def test_no_store_exits_1(tmp_path):
    p = run_cli(tmp_path / "definitely-missing", "status", cwd=tmp_path)
    assert p.returncode == 1
    assert "no store" in p.stderr


def test_no_command_exits_1(tmp_path):
    store = tmp_path / "s"
    _seed(store, tmp_path)
    p = run_cli(store)
    assert p.returncode == 1


def test_gc_dry_run_then_delete(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    n_before = len(list((store / "objects").iterdir()))
    assert n_before > 0

    # everything is reachable from the recent run: nothing to collect
    p = run_cli(store, "gc", "--older-than", "7d", cwd=cwd)
    assert p.returncode == 0 and "deleted 0 object(s)" in p.stdout

    # age the run below the cutoff -> everything unreachable
    db = sqlite3.connect(store / "index.db")
    db.execute("UPDATE runs SET started_at = started_at - 8*86400")
    db.commit()
    db.close()
    p = run_cli(store, "gc", "--older-than", "7d", "--dry-run", cwd=cwd)
    assert p.returncode == 0 and f"would delete {n_before} object(s)" in p.stdout
    assert len(list((store / "objects").iterdir())) == n_before  # dry: intact

    p = run_cli(store, "gc", "--older-than", "7d", cwd=cwd)
    assert p.returncode == 0 and f"deleted {n_before} object(s)" in p.stdout
    assert list((store / "objects").iterdir()) == []
    db = sqlite3.connect(store / "index.db")
    assert db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
    db.close()


def test_gc_keeps_objects_reachable_from_recent_runs(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    # second, recent run reuses everything as hits
    run_demo(store, "--n", 3, cwd=cwd)
    db = sqlite3.connect(store / "index.db")
    db.execute("UPDATE runs SET started_at = started_at - 8*86400 WHERE run_id = 1")
    db.commit()
    db.close()
    n_before = len(list((store / "objects").iterdir()))
    p = run_cli(store, "gc", "--older-than", "7d", cwd=cwd)
    assert p.returncode == 0 and "deleted 0 object(s)" in p.stdout
    assert len(list((store / "objects").iterdir())) == n_before


def test_gc_bad_duration(tmp_path):
    store = tmp_path / "s"
    cwd = _seed(store, tmp_path)
    p = run_cli(store, "gc", "--older-than", "fortnight", cwd=cwd)
    assert p.returncode == 1
    assert "duration" in p.stderr
