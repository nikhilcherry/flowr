"""Content-addressed object store + SQLite index.

Layout (under ``.flowr/`` in the cwd, or ``$FLOWR_DIR``):

    objects/<result_hash>   pickled (protocol 5) or .npz-encoded results
    index.db                SQLite, WAL mode

Every path recorded in the index is relative to the store root (objects are
referenced purely by hash), so the store survives its parent directory being
moved.

Crash safety: a node is committed by (1) atomically writing the object file,
then (2) inserting the node row + edges in one transaction. A run killed at
any instant leaves only fully-valid committed nodes; the next run resumes
from the frontier.

Only the parent process may touch SQLite — workers hand serialized bytes
back to the parent, which commits.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import re
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

from .errors import FlowrError

try:
    import numpy as _np
except ImportError:
    _np = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
    node_key    TEXT PRIMARY KEY,
    stage_name  TEXT NOT NULL,
    code_hash   TEXT NOT NULL,
    param_json  TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    duration_s  REAL,
    run_id      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_nodes_stage ON nodes(stage_name, created_at);
CREATE TABLE IF NOT EXISTS edges(
    child_key    TEXT NOT NULL,
    parent_key   TEXT NOT NULL,
    arg_position INTEGER NOT NULL,
    PRIMARY KEY (child_key, arg_position)
);
CREATE TABLE IF NOT EXISTS runs(
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     REAL NOT NULL,
    finished_at    REAL,
    requested_keys TEXT,
    git_commit     TEXT,
    n_hits         INTEGER DEFAULT 0,
    n_misses       INTEGER DEFAULT 0,
    status         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS objects(
    result_hash TEXT PRIMARY KEY,
    codec       TEXT NOT NULL,
    size        INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
"""


def resolve_root(root=None):
    if root is not None:
        return Path(root)
    env = os.environ.get("FLOWR_DIR")
    if env:
        return Path(env)
    return Path.cwd() / ".flowr"


# --------------------------------------------------------------------------
# value serialization (pure functions — safe to call from worker processes)

def _npz_bytes(arrays):
    """Deterministic .npz encoding: fixed zip timestamps, sorted member
    order, so byte-identical arrays always produce byte-identical archives
    (np.savez stamps wall-clock time into the zip, breaking replay)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            payload = io.BytesIO()
            _np.lib.format.write_array(
                payload, _np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload.getvalue())
    return buf.getvalue()


def _npz_key_ok(k):
    return isinstance(k, str) and k and "/" not in k and "\\" not in k


def serialize_value(value):
    """-> (codec, blob). numpy arrays and flat dicts of arrays become real
    .npz archives (inspectable with np.load / any zip tool); everything else
    is pickle protocol 5."""
    if _np is not None:
        if isinstance(value, _np.ndarray) and value.dtype != object:
            return "npz-array", _npz_bytes({"arr": value})
        if (
            isinstance(value, dict)
            and value
            and all(_npz_key_ok(k) for k in value)
            and all(
                isinstance(v, _np.ndarray) and v.dtype != object
                for v in value.values()
            )
        ):
            return "npz-dict", _npz_bytes(value)
    return "pickle", pickle.dumps(value, protocol=5)


def deserialize_value(codec, blob):
    if codec == "pickle":
        return pickle.loads(blob)
    if codec in ("npz-array", "npz-dict"):
        if _np is None:
            raise FlowrError(
                "this cached result was stored as .npz but numpy is not "
                "installed in the current environment"
            )
        with _np.load(io.BytesIO(blob), allow_pickle=False) as f:
            if codec == "npz-array":
                return f["arr"]
            return {k: f[k] for k in f.files}
    raise FlowrError(f"unknown object codec {codec!r} in store")


_DURATION_RE = re.compile(r"^(\d+)([smhdw])$")
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_duration(text):
    """'7d' / '12h' / '30m' / '45s' / '2w' -> seconds."""
    m = _DURATION_RE.match(str(text).strip())
    if not m:
        raise FlowrError(
            f"cannot parse duration {text!r}; expected e.g. 7d, 12h, 30m, 45s, 2w"
        )
    return int(m.group(1)) * _DURATION_UNITS[m.group(2)]


class Store:
    """Object store + index. One instance per process; parent-only."""

    def __init__(self, root=None):
        self.root = resolve_root(root)
        self.objects_dir = self.root / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "index.db", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # -- objects ------------------------------------------------------------

    def put_bytes(self, codec, blob):
        import hashlib

        rh = hashlib.sha256(blob).hexdigest()
        path = self.objects_dir / rh
        if not path.exists():
            fd, tmp = tempfile.mkstemp(dir=self.objects_dir, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(blob)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO objects VALUES (?,?,?,?)",
                (rh, codec, len(blob), time.time()),
            )
        return rh

    def has_object(self, result_hash):
        row = self.db.execute(
            "SELECT 1 FROM objects WHERE result_hash=?", (result_hash,)
        ).fetchone()
        return row is not None and (self.objects_dir / result_hash).exists()

    def get_value(self, result_hash):
        row = self.db.execute(
            "SELECT codec FROM objects WHERE result_hash=?", (result_hash,)
        ).fetchone()
        path = self.objects_dir / result_hash
        if row is None or not path.exists():
            raise FlowrError(
                f"object {result_hash[:16]}… is missing from the store "
                "(deleted by gc, or the store was manually modified)"
            )
        return deserialize_value(row["codec"], path.read_bytes())

    # -- nodes --------------------------------------------------------------

    def get_node(self, node_key):
        """Committed node row, or None. A row whose object file is gone
        counts as absent (treated as a cache miss)."""
        row = self.db.execute(
            "SELECT * FROM nodes WHERE node_key=?", (node_key,)
        ).fetchone()
        if row is None or not self.has_object(row["result_hash"]):
            return None
        return row

    def get_node_row(self, node_key):
        return self.db.execute(
            "SELECT * FROM nodes WHERE node_key=?", (node_key,)
        ).fetchone()

    def commit_node(self, node_key, stage_name, code_hash, param_json,
                    result_hash, duration_s, run_id, parent_keys):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?)",
                (node_key, stage_name, code_hash,
                 json.dumps(param_json, sort_keys=True), result_hash, "ok",
                 time.time(), duration_s, run_id),
            )
            self.db.execute("DELETE FROM edges WHERE child_key=?", (node_key,))
            self.db.executemany(
                "INSERT INTO edges VALUES (?,?,?)",
                [(node_key, pk, i) for i, pk in enumerate(parent_keys)],
            )

    def parent_keys(self, child_key):
        rows = self.db.execute(
            "SELECT parent_key FROM edges WHERE child_key=? ORDER BY arg_position",
            (child_key,),
        ).fetchall()
        return [r["parent_key"] for r in rows]

    def nodes_for_stage(self, stage_name, limit=100):
        return self.db.execute(
            "SELECT * FROM nodes WHERE stage_name=? ORDER BY created_at DESC LIMIT ?",
            (stage_name, limit),
        ).fetchall()

    # -- runs ---------------------------------------------------------------

    def begin_run(self, git_commit=None):
        with self.db:
            cur = self.db.execute(
                "INSERT INTO runs (started_at, git_commit, status) VALUES (?,?,?)",
                (time.time(), git_commit, "running"),
            )
        return cur.lastrowid

    def finish_run(self, run_id, requested_keys, n_hits, n_misses, status):
        with self.db:
            self.db.execute(
                "UPDATE runs SET finished_at=?, requested_keys=?, n_hits=?, "
                "n_misses=?, status=? WHERE run_id=?",
                (time.time(), json.dumps(requested_keys), n_hits, n_misses,
                 status, run_id),
            )


def collect_garbage(older_than, dry_run=False, root=None):
    """Delete objects unreachable from any run newer than the cutoff.

    Reachable = the closure (via edges) of every requested_keys list of runs
    started within the window. Node rows whose object is deleted are removed
    too. Returns {'deleted_objects', 'freed_bytes', 'dry_run'}.
    """
    store = Store(root)
    try:
        cutoff = time.time() - parse_duration(older_than)
        frontier = []
        for row in store.db.execute(
            "SELECT requested_keys FROM runs WHERE started_at >= ?", (cutoff,)
        ):
            frontier.extend(k for k in json.loads(row["requested_keys"] or "[]") if k)
        live_keys = set()
        while frontier:
            k = frontier.pop()
            if k in live_keys:
                continue
            live_keys.add(k)
            frontier.extend(store.parent_keys(k))
        live_hashes = set()
        for k in live_keys:
            row = store.get_node_row(k)
            if row is not None:
                live_hashes.add(row["result_hash"])

        deleted = 0
        freed = 0
        for f in store.objects_dir.iterdir():
            if f.name.startswith(".tmp-"):
                continue
            if f.name not in live_hashes:
                freed += f.stat().st_size
                deleted += 1
                if not dry_run:
                    f.unlink()
        if not dry_run:
            with store.db:
                if live_hashes:
                    marks = ",".join("?" * len(live_hashes))
                    dead = [
                        r["node_key"]
                        for r in store.db.execute(
                            f"SELECT node_key FROM nodes WHERE result_hash NOT IN ({marks})",
                            tuple(live_hashes),
                        )
                    ]
                    store.db.execute(
                        f"DELETE FROM objects WHERE result_hash NOT IN ({marks})",
                        tuple(live_hashes),
                    )
                else:
                    dead = [
                        r["node_key"]
                        for r in store.db.execute("SELECT node_key FROM nodes")
                    ]
                    store.db.execute("DELETE FROM objects")
                for k in dead:
                    store.db.execute("DELETE FROM nodes WHERE node_key=?", (k,))
                    store.db.execute("DELETE FROM edges WHERE child_key=?", (k,))
        return {"deleted_objects": deleted, "freed_bytes": freed, "dry_run": dry_run}
    finally:
        store.close()
