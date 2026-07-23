"""DAG execution: ready-set scheduling over a ProcessPoolExecutor, dry-run
planning with per-miss reasons, retries, and blocked-subtree semantics.

Only the parent process touches SQLite. Workers execute the stage function,
serialize the result (a pure function), and hand bytes back; the parent
hashes, writes the object file, and commits the index row — in that order,
so a crash at any instant leaves only fully-valid nodes.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from .errors import FlowrError, RunError
from .graph import HIT, MISS, Plan, PlanEntry, closure
from .node import Node
from .store import Store, serialize_value

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _worker_init():
    # Float determinism: BLAS threading can reorder reductions. Overridable
    # with FLOWR_KEEP_THREADS=1.
    if os.environ.get("FLOWR_KEEP_THREADS") != "1":
        for var in _THREAD_VARS:
            os.environ[var] = "1"


def _run_stage(stage, arguments, retries):
    """Executes in a worker (or inline when workers=1). Returns
    (status, codec, blob, duration_s, traceback_str)."""
    import inspect

    start = time.perf_counter()
    tb = None
    for _attempt in range(retries + 1):
        try:
            ba = inspect.BoundArguments(stage.signature, dict(arguments))
            value = stage.func(*ba.args, **ba.kwargs)
            codec, blob = serialize_value(value)
            return ("ok", codec, blob, time.perf_counter() - start, None)
        except Exception:
            tb = traceback.format_exc()
    return ("error", None, None, time.perf_counter() - start, tb)


def _git_commit():
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# dry-run planning

def _short(v):
    s = json.dumps(v, default=str)
    return s if len(s) <= 60 else s[:57] + "..."


def _diff_params(old, new):
    msgs = []
    for k in sorted(set(old) | set(new)):
        o, n = old.get(k, "<absent>"), new.get(k, "<absent>")
        if o == n:
            continue
        if (
            isinstance(o, dict) and isinstance(n, dict)
            and o.get("$file") is not None and o.get("$file") == n.get("$file")
        ):
            msgs.append(f"file content changed: {o['$file']}")
        elif k == "$version":
            msgs.append(f"stage version: {o} -> {n}")
        else:
            msgs.append(f"param {k}: {_short(o)} -> {_short(n)}")
    return "; ".join(msgs) or "params changed"


def _diagnose_miss(store, node, node_key, parent_keys):
    """Human-readable reason for a cache miss, by comparing against the most
    recent index rows for the same stage."""
    if store.get_node_row(node_key) is not None:
        return "cached object missing (removed by gc?)"
    rows = store.nodes_for_stage(node.stage_name)
    if not rows:
        return "new node"
    for row in rows:
        same_code = row["code_hash"] == node.stage.code_hash
        same_edges = store.parent_keys(row["node_key"]) == parent_keys
        row_params = json.loads(row["param_json"])
        same_params = row_params == node.param_json
        if same_edges and same_params and not same_code:
            return "code changed"
        if same_edges and same_code and not same_params:
            return _diff_params(row_params, node.param_json)
    return "new node"


def _build_plan(order, store):
    entries = []
    status = {}
    rhash = {}
    key_of = {}
    for node in order:
        if any(status[id(p)] == MISS for p in node.edges):
            entries.append(PlanEntry(node, None, MISS, "upstream miss"))
            status[id(node)] = MISS
            continue
        ups = [rhash[id(p)] for p in node.edges]
        key = node.key_with(ups)
        key_of[id(node)] = key
        row = store.get_node(key)
        if row is not None:
            entries.append(PlanEntry(node, key, HIT))
            status[id(node)] = HIT
            rhash[id(node)] = row["result_hash"]
        else:
            parent_keys = [key_of[id(p)] for p in node.edges]
            reason = _diagnose_miss(store, node, key, parent_keys)
            entries.append(PlanEntry(node, key, MISS, reason))
            status[id(node)] = MISS
    return Plan(entries)


# --------------------------------------------------------------------------
# execution

class _Abort(Exception):
    pass


def run(target, workers=1, dry=False, fail_fast=False, root=None):
    """Execute (or plan) the closure of the requested Node(s).

    - target: a Node or a list of Nodes (result shape mirrors the input).
    - workers=1 executes in-process (easy debugging); workers>1 uses a
      ProcessPoolExecutor with ready-set scheduling.
    - dry=True returns a Plan (HIT/MISS + reason per node), executes nothing.
    - fail_fast=True aborts at the first failure instead of finishing
      independent branches.
    """
    if workers < 1:
        raise FlowrError(f"workers must be >= 1, got {workers}")

    single = isinstance(target, Node)
    if single:
        targets = [target]
    else:
        try:
            targets = list(target)
        except TypeError:
            raise FlowrError(
                f"flowr.run expects a Node or a list of Nodes; got "
                f"{type(target).__name__}. Did you call the stage function?"
            ) from None
    for t in targets:
        if not isinstance(t, Node):
            raise FlowrError(
                f"flowr.run expects a Node or a list of Nodes; got "
                f"{type(t).__name__}. Did you call the stage function?"
            )
    if not targets:
        return Plan([]) if dry else []

    order = closure(targets)
    store = Store(root)
    try:
        if dry:
            return _build_plan(order, store)
        return _execute(order, targets, single, store, workers, fail_fast)
    finally:
        store.close()


def _execute(order, targets, single, store, workers, fail_fast):
    children = {id(n): [] for n in order}
    pending = {}
    for n in order:
        pending[id(n)] = len(n.edges)
        for p in n.edges:
            children[id(p)].append(n)
    target_ids = {id(t) for t in targets}
    refs = {id(n): len(children[id(n)]) for n in order}

    state = {id(n): "pending" for n in order}
    key_of = {}
    rhash = {}
    values = {}
    failures = []  # (stage_name, node_key, traceback)
    hits = misses = 0

    run_id = store.begin_run(git_commit=_git_commit())
    ready = deque(n for n in order if pending[id(n)] == 0)
    inflight = {}       # future -> (node, key)
    inflight_keys = {}  # key -> executing node
    waiters = {}        # key -> [nodes with identical key, awaiting result]
    pool = None

    def value_of(node):
        i = id(node)
        if i not in values:
            values[i] = store.get_value(rhash[i])
        return values[i]

    def release(parent):
        i = id(parent)
        refs[i] -= 1
        if refs[i] <= 0 and i not in target_ids:
            values.pop(i, None)

    def mark_done(node, result_hash):
        state[id(node)] = "done"
        rhash[id(node)] = result_hash
        for c in children[id(node)]:
            pending[id(c)] -= 1
            if pending[id(c)] == 0 and state[id(c)] == "pending":
                ready.append(c)

    def mark_failed(node, key, tb):
        nonlocal failures
        state[id(node)] = "failed"
        failures.append((node.stage_name, key, tb))
        stack = [node]
        while stack:
            cur = stack.pop()
            for c in children[id(cur)]:
                if state[id(c)] == "pending":
                    state[id(c)] = "blocked"
                    stack.append(c)
        if fail_fast:
            raise _Abort()

    def commit(node, key, codec, blob, duration):
        nonlocal hits
        rh = store.put_bytes(codec, blob)
        store.commit_node(
            key, node.stage_name, node.stage.code_hash, node.param_json,
            rh, duration, run_id, [key_of[id(p)] for p in node.edges],
        )
        mark_done(node, rh)
        for w in waiters.pop(key, ()):
            hits += 1
            mark_done(w, rh)

    def process_ready(node):
        nonlocal hits, misses, pool
        if state[id(node)] != "pending":
            return
        ups = [rhash[id(p)] for p in node.edges]
        key = node.key_with(ups)
        key_of[id(node)] = key
        row = store.get_node(key)
        if row is not None:
            hits += 1
            mark_done(node, row["result_hash"])
            return
        if key in inflight_keys:
            waiters.setdefault(key, []).append(node)
            return
        misses += 1
        state[id(node)] = "running"
        args = node.materialized_arguments([value_of(p) for p in node.edges])
        for p in node.edges:
            release(p)
        if workers == 1:
            status_, codec, blob, dur, tb = _run_stage(
                node.stage, args, node.stage.retries)
            if status_ == "ok":
                commit(node, key, codec, blob, dur)
            else:
                mark_failed(node, key, tb)
        else:
            if pool is None:
                pool = ProcessPoolExecutor(workers, initializer=_worker_init)
            fut = pool.submit(_run_stage, node.stage, args, node.stage.retries)
            inflight[fut] = (node, key)
            inflight_keys[key] = node

    status = "crashed"
    try:
        while True:
            while ready:
                process_ready(ready.popleft())
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                node, key = inflight.pop(fut)
                inflight_keys.pop(key, None)
                status_, codec, blob, dur, tb = fut.result()
                if status_ == "ok":
                    commit(node, key, codec, blob, dur)
                else:
                    mark_failed(node, key, tb)
                    for w in waiters.pop(key, ()):
                        mark_failed(w, key, tb)
        status = "failed" if failures else "ok"
    except _Abort:
        status = "failed"
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        store.finish_run(
            run_id, [key_of.get(id(t)) for t in targets], hits, misses, status)

    if failures:
        n_blocked = sum(1 for s in state.values() if s == "blocked")
        head = (
            f"flowr run failed: {len(failures)} node(s) errored, "
            f"{n_blocked} downstream node(s) blocked.\n"
        )
        details = "\n".join(
            f"--- {name}  [{(key or '?')[:16]}] ---\n{tb}"
            for name, key, tb in failures
        )
        raise RunError(head + details, failures=failures, n_blocked=n_blocked)

    results = [value_of(t) for t in targets]
    return results[0] if single else results
