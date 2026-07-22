"""Phases 5–6 behaviors: incremental correctness, kill-resume, determinism,
AST-normalization behavior, cwd-independence — all via the demo pipeline run
as a real subprocess."""

import json
import re
import shutil
import sqlite3

import pytest

from helpers import DEMO, counter_lines, demo_node_count, run_cli, run_demo

N = 6  # keep the in-suite runs quick; verification runs the full sizes


def _misses(stdout):
    m = re.search(r"HITS=(\d+) MISSES=(\d+)", stdout)
    assert m, stdout
    return int(m.group(1)), int(m.group(2))


def test_incremental_correctness(tmp_path):
    store = tmp_path / "s"
    counter = tmp_path / "counter"
    total = demo_node_count(N)

    p = run_demo(store, "--n", N, extra_env={"FLOWR_DEMO_COUNTER": str(counter)})
    assert p.returncode == 0, p.stderr
    assert counter_lines(counter) == total

    # second run: 100% HIT, zero stage code executed
    p = run_demo(store, "--n", N, extra_env={"FLOWR_DEMO_COUNTER": str(counter)})
    assert p.returncode == 0, p.stderr
    assert counter_lines(counter) == total

    # change one param on the aggregate stage: dry-run plans exactly 2 misses
    p = run_demo(store, "--n", N, "--top-k", 3, "--dry")
    assert p.returncode == 0, p.stderr
    hits, misses = _misses(p.stdout)
    assert (hits, misses) == (total - 2, 2)
    assert "aggregate" in p.stdout and "figure" in p.stdout
    assert "param top_k: 5 -> 3" in p.stdout
    assert "upstream miss" in p.stdout

    # the real run matches the plan exactly
    p = run_demo(store, "--n", N, "--top-k", 3,
                 extra_env={"FLOWR_DEMO_COUNTER": str(counter)})
    assert p.returncode == 0, p.stderr
    assert counter_lines(counter) == total + 2


def test_kill_resume(tmp_path):
    store = tmp_path / "s"
    counter = tmp_path / "counter"
    total = demo_node_count(N)
    kill_at = 7

    p = run_demo(store, "--n", N, "--counter", counter, "--kill-after", kill_at)
    assert p.returncode == 1                      # os._exit(1) mid-run
    assert counter_lines(counter) == kill_at      # exactly K executions happened

    # resume: completes; nothing recomputed, nothing corrupted
    p = run_demo(store, "--n", N, "--counter", counter)
    assert p.returncode == 0, p.stderr
    assert counter_lines(counter) == total        # sum across runs == one cold run

    db = sqlite3.connect(store / "index.db")
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    n_nodes = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    db.close()
    assert n_nodes == total

    # third run: pure cache
    p = run_demo(store, "--n", N, "--counter", counter)
    assert p.returncode == 0
    assert counter_lines(counter) == total


def _figure_result_hash(store):
    db = sqlite3.connect(store / "index.db")
    (requested,) = db.execute(
        "SELECT requested_keys FROM runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    key = json.loads(requested)[0]
    (rh,) = db.execute(
        "SELECT result_hash FROM nodes WHERE node_key=?", (key,)
    ).fetchone()
    db.close()
    return key, rh


def test_determinism_across_fresh_stores_parallel(tmp_path):
    stores = [tmp_path / "a", tmp_path / "b"]
    for s in stores:
        p = run_demo(s, "--n", N, "--workers", 4)
        assert p.returncode == 0, p.stderr
    (k1, r1), (k2, r2) = (_figure_result_hash(s) for s in stores)
    assert k1 == k2 and r1 == r2
    names = [sorted(f.name for f in (s / "objects").iterdir()) for s in stores]
    assert names[0] == names[1]
    # `flowr why` reports the identical result hash from both stores
    outs = [run_cli(s, "why", k1[:16]) for s in stores]
    assert all(o.returncode == 0 for o in outs)
    assert r1 in outs[0].stdout and r1 in outs[1].stdout


def test_ast_norm_behavioral(tmp_path):
    """Appending a comment -> all HIT; changing a body constant -> that node
    and its descendants MISS, siblings HIT."""
    demo = tmp_path / "demo_copy.py"
    shutil.copy(DEMO, demo)
    store = tmp_path / "s"
    total = demo_node_count(N)

    assert run_demo(store, "--n", N, demo=demo).returncode == 0

    # comment edit inside analyze()
    src = demo.read_text()
    assert "bias = 0.0" in src
    demo.write_text(src.replace("bias = 0.0", "bias = 0.0  # tweaked comment"))
    p = run_demo(store, "--n", N, "--dry", demo=demo)
    assert p.returncode == 0, p.stderr
    assert _misses(p.stdout) == (total, 0)

    # constant edit inside analyze(): analyze nodes + aggregate + figure miss
    demo.write_text(src.replace("bias = 0.0", "bias = 1e-9"))
    p = run_demo(store, "--n", N, "--dry", demo=demo)
    assert p.returncode == 0, p.stderr
    hits, misses = _misses(p.stdout)
    assert misses == N + 2 and hits == 2 * N
    assert "code changed" in p.stdout

    p = run_demo(store, "--n", N, demo=demo)
    assert p.returncode == 0, p.stderr


def test_cwd_independence(tmp_path):
    """Demo and every CLI command run as subprocesses from a foreign cwd,
    with FLOWR_DIR pointing at the store."""
    store = tmp_path / "the-store"
    foreign = tmp_path / "some" / "other" / "place"
    foreign.mkdir(parents=True)

    p = run_demo(store, "--n", N, cwd=foreign)
    assert p.returncode == 0, p.stderr
    assert not (foreign / ".flowr").exists()      # FLOWR_DIR won, not cwd

    for cmd in (["status"], ["runs"], ["gc", "--older-than", "7d", "--dry-run"]):
        p = run_cli(store, *cmd, cwd=foreign)
        assert p.returncode == 0, (cmd, p.stderr)
    key, _ = _figure_result_hash(store)
    p = run_cli(store, "why", key[:12], cwd=foreign)
    assert p.returncode == 0, p.stderr


@pytest.mark.slow
def test_parallel_stress_small(tmp_path):
    """Scaled-down stress (full N=200 x workers=8 runs in verification)."""
    store = tmp_path / "s"
    n = 40
    p = run_demo(store, "--n", n, "--workers", 8, "--n-points", 60)
    assert p.returncode == 0, p.stderr
    p = run_demo(store, "--n", n, "--workers", 8, "--n-points", 60, "--dry")
    assert p.returncode == 0
    assert _misses(p.stdout) == (demo_node_count(n), 0)
