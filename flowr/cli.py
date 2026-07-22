"""Read-only inspection CLI (+ gc). Pipelines run via Python; this is for
looking at the store: `flowr status`, `flowr why KEY`, `flowr runs`,
`flowr gc --older-than 7d`.

Exit codes: 0 ok, 1 usage/error, 3 nothing matched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .errors import FlowrError
from .store import Store, collect_garbage, resolve_root


def _human_bytes(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def _ts(t):
    if t is None:
        return "-"
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _open_store():
    root = resolve_root()
    if not (root / "index.db").exists():
        print(
            f"flowr: no store found at {root} "
            "(run a pipeline first, or point FLOWR_DIR at a store)",
            file=sys.stderr,
        )
        return None
    return Store(root)


def _table(rows, headers):
    widths = [len(h) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    out += [fmt.format(*r) for r in srows]
    return "\n".join(out)


# --------------------------------------------------------------------------

def cmd_status(store, args):
    db = store.db
    stages = db.execute(
        "SELECT stage_name, COUNT(*) AS n FROM nodes GROUP BY stage_name "
        "ORDER BY stage_name"
    ).fetchall()
    last = db.execute(
        "SELECT * FROM runs WHERE finished_at IS NOT NULL "
        "ORDER BY run_id DESC LIMIT 1"
    ).fetchone()

    # Per-stage hit/miss for the last finished run: walk that run's requested
    # closure; nodes created by the run were misses, the rest were hits.
    last_hits, last_misses = {}, {}
    if last is not None:
        seen = set()
        frontier = [k for k in json.loads(last["requested_keys"] or "[]") if k]
        while frontier:
            k = frontier.pop()
            if k in seen:
                continue
            seen.add(k)
            row = store.get_node_row(k)
            if row is None:
                continue
            bucket = last_misses if row["run_id"] == last["run_id"] else last_hits
            bucket[row["stage_name"]] = bucket.get(row["stage_name"], 0) + 1
            frontier.extend(store.parent_keys(k))

    rows = [
        (s["stage_name"], s["n"],
         last_hits.get(s["stage_name"], 0), last_misses.get(s["stage_name"], 0))
        for s in stages
    ]
    print(_table(rows, ["stage", "nodes cached", "last-run hits", "last-run misses"]))
    n_obj = 0
    size = 0
    for f in store.objects_dir.iterdir():
        if not f.name.startswith(".tmp-"):
            n_obj += 1
            size += f.stat().st_size
    print(f"\nstore: {store.root}  —  {n_obj} objects, {_human_bytes(size)}")
    if last is not None:
        print(
            f"last run: #{last['run_id']} at {_ts(last['started_at'])} UTC — "
            f"{last['n_hits']} hits / {last['n_misses']} misses ({last['status']})"
        )
    return 0


def cmd_runs(store, args):
    rows = store.db.execute(
        "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (args.limit,)
    ).fetchall()
    if not rows:
        print("no runs recorded")
        return 0
    table = []
    for r in rows:
        dur = (
            f"{r['finished_at'] - r['started_at']:.2f}s"
            if r["finished_at"] else "-"
        )
        table.append((
            f"#{r['run_id']}", _ts(r["started_at"]), dur,
            r["n_hits"] or 0, r["n_misses"] or 0, r["status"],
            (r["git_commit"] or "-")[:10],
        ))
    print(_table(table, ["run", "started (UTC)", "duration", "hits", "misses",
                         "status", "git"]))
    return 0


def _print_node(store, key, indent, seen):
    pad = "    " * indent
    row = store.get_node_row(key)
    if row is None:
        print(f"{pad}{key[:16]}…  (not in index)")
        return
    obj = store.db.execute(
        "SELECT * FROM objects WHERE result_hash=?", (row["result_hash"],)
    ).fetchone()
    codec = obj["codec"] if obj else "?"
    size = _human_bytes(obj["size"]) if obj else "?"
    print(f"{pad}{row['node_key']}")
    print(f"{pad}  stage: {row['stage_name']}   status: {row['status']}   "
          f"duration: {row['duration_s']:.3f}s   run: #{row['run_id']}")
    run = store.db.execute(
        "SELECT git_commit FROM runs WHERE run_id=?", (row["run_id"],)
    ).fetchone()
    git = (run["git_commit"] if run and run["git_commit"] else "-")
    print(f"{pad}  created: {_ts(row['created_at'])} UTC   git: {git}")
    print(f"{pad}  code_hash: {row['code_hash']}")
    print(f"{pad}  result: {row['result_hash']}  ({codec}, {size})")
    params = json.dumps(json.loads(row["param_json"]), indent=2, sort_keys=True)
    print(f"{pad}  params: " + params.replace("\n", "\n" + pad + "  "))
    parents = store.parent_keys(key)
    if parents:
        print(f"{pad}  parents:")
        for i, pk in enumerate(parents):
            print(f"{pad}    [{i}]")
            if pk in seen:
                print(f"{pad}    {pk[:16]}…  (shown above)")
            else:
                seen.add(pk)
                _print_node(store, pk, indent + 2, seen)


def cmd_why(store, args):
    rows = store.db.execute(
        "SELECT node_key FROM nodes WHERE node_key LIKE ? ORDER BY node_key",
        (args.prefix + "%",),
    ).fetchall()
    if not rows:
        print(f"flowr why: no node matches prefix {args.prefix!r}", file=sys.stderr)
        return 3
    if len(rows) > 1:
        print(
            f"flowr why: prefix {args.prefix!r} is ambiguous "
            f"({len(rows)} matches):", file=sys.stderr,
        )
        for r in rows[:10]:
            print(f"  {r['node_key']}", file=sys.stderr)
        return 1
    _print_node(store, rows[0]["node_key"], 0, {rows[0]["node_key"]})
    return 0


def cmd_gc(store, args):
    store.close()
    stats = collect_garbage(args.older_than, dry_run=args.dry_run)
    verb = "would delete" if stats["dry_run"] else "deleted"
    print(
        f"gc: {verb} {stats['deleted_objects']} object(s), "
        f"reclaiming {_human_bytes(stats['freed_bytes'])}"
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="flowr",
        description="Inspect a flowr store (resolved from $FLOWR_DIR or ./.flowr).",
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("status", help="per-stage cache summary and store size")
    w = sub.add_parser("why", help="full provenance chain for a node")
    w.add_argument("prefix", help="node key prefix (unique-prefix matched, like git)")
    r = sub.add_parser("runs", help="recent runs with hit/miss counts")
    r.add_argument("--limit", type=int, default=20)
    g = sub.add_parser("gc", help="delete objects unreachable from recent runs")
    g.add_argument("--older-than", required=True, metavar="AGE",
                   help="keep everything reachable from runs newer than this "
                        "(e.g. 7d, 12h, 30m)")
    g.add_argument("--dry-run", action="store_true")

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 1
    store = _open_store()
    if store is None:
        return 1
    try:
        handler = {"status": cmd_status, "why": cmd_why,
                   "runs": cmd_runs, "gc": cmd_gc}[args.cmd]
        return handler(store, args)
    except FlowrError as e:
        print(f"flowr: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            store.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
