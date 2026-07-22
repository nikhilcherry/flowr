"""Lazy call-graph objects: File inputs, the @stage decorator, and Node.

Calling a decorated stage executes nothing — it binds the arguments, hashes
the parameters, records Node arguments as graph edges, and returns a Node.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import os

from . import hashing
from .errors import FlowrError

_KEY_SALT = b"flowr:node:1\x00"


class File:
    """Declares a filesystem input. The file's *content hash* (streamed
    sha256) enters the cache key — the path itself does not, so renaming a
    file without changing its bytes does not invalidate.

    The stage function receives the plain path string at execution time.

    Paths passed as ordinary ``str`` parameters are opaque values: flowr
    cannot see that a stage reads them, and edits to those files are
    invisible to invalidation (the same contract Make has). Always wrap real
    inputs in ``flowr.File``.
    """

    __slots__ = ("path", "_hash")

    def __init__(self, path):
        self.path = os.fspath(path)
        self._hash = None

    @property
    def content_hash(self):
        if self._hash is None:
            try:
                self._hash = hashing.file_sha256(self.path)
            except FileNotFoundError:
                raise FlowrError(
                    f"flowr.File input does not exist: {self.path!r}"
                ) from None
        return self._hash

    def __repr__(self):
        return f"flowr.File({self.path!r})"


class EdgeRef(int):
    """Placeholder left in the argument structure where a Node edge was.
    The integer is the index into ``Node.edges``."""

    __slots__ = ()

    def __repr__(self):
        return f"EdgeRef({int(self)})"


def _replace_nodes(value, on_node):
    """Rebuild a container structure, mapping every Node through on_node."""
    if isinstance(value, Node):
        return on_node(value)
    t = type(value)
    if t is list:
        return [_replace_nodes(v, on_node) for v in value]
    if t is tuple:
        return tuple(_replace_nodes(v, on_node) for v in value)
    if t is dict:
        return {k: _replace_nodes(v, on_node) for k, v in value.items()}
    return value


class Node:
    """One lazy stage invocation. Immutable once created.

    ``edges`` is the ordered list of unique upstream Nodes discovered in the
    bound arguments (directly or nested in lists/tuples/dicts). Everything
    else in the arguments is a parameter and was canonically hashed at
    construction time, so bad parameter types fail at composition, not at
    run time.
    """

    __slots__ = ("stage", "arguments", "edges", "param_digest", "param_json")

    def __init__(self, stage, arguments):
        self.stage = stage
        self.arguments = arguments  # bound, defaults applied, Nodes in place
        edges = []
        index_of = {}

        def on_node(n):
            j = index_of.get(id(n))
            if j is None:
                j = len(edges)
                index_of[id(n)] = j
                edges.append(n)
            return EdgeRef(j)

        substituted = {
            name: _replace_nodes(v, on_node) for name, v in arguments.items()
        }
        self.edges = edges
        self.param_digest = hashing.params_digest(substituted, stage.name)
        pj = {"$version": stage.version}
        for name, v in substituted.items():
            pj[name] = hashing.json_repr(v)
        self.param_json = pj

    @property
    def stage_name(self):
        return self.stage.name

    def key_with(self, upstream_hashes):
        """node_key given the *result* hashes of each edge, in edge order."""
        h = hashlib.sha256(_KEY_SALT)
        h.update(self.stage.code_hash.encode())
        h.update(b"\x00")
        h.update(self.stage.version.encode())
        h.update(b"\x00")
        h.update(self.param_digest)
        for u in upstream_hashes:
            h.update(b"\x00")
            h.update(u.encode())
        return h.hexdigest()

    def materialized_arguments(self, edge_values):
        """Arguments dict ready to call: Nodes replaced by their computed
        values (``edge_values`` aligned with ``self.edges``), Files replaced
        by their path strings."""
        index_of = {id(e): i for i, e in enumerate(self.edges)}

        def sub(v):
            if isinstance(v, Node):
                return edge_values[index_of[id(v)]]
            if isinstance(v, File):
                return v.path
            t = type(v)
            if t is list:
                return [sub(x) for x in v]
            if t is tuple:
                return tuple(sub(x) for x in v)
            if t is dict:
                return {k: sub(x) for k, x in v.items()}
            return v

        return {name: sub(v) for name, v in self.arguments.items()}

    def __repr__(self):
        return (
            f"<flowr.Node {self.stage_name} edges={len(self.edges)} "
            f"params={self.param_digest.hex()[:8]}>"
        )


def _resolve_stage(module, qualname):
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, Stage):
        raise FlowrError(
            f"{module}.{qualname} is no longer a flowr stage; cannot "
            "reconstruct it in a worker process."
        )
    return obj


class Stage:
    """A pipeline stage: a wrapped top-level function. Calling it returns a
    Node; the underlying function runs only inside flowr.run()."""

    def __init__(self, func, *, version="0", code_deps=(), retries=0):
        if func.__name__ == "<lambda>" or "<locals>" in func.__qualname__:
            raise FlowrError(
                "flowr stages must be top-level named functions (picklable "
                f"and importable); got {func.__qualname__!r}. Lambdas and "
                "closures cannot be stages."
            )
        functools.update_wrapper(self, func)
        self.func = func
        self.name = func.__qualname__
        self.module = func.__module__
        self.version = str(version)
        self.code_deps = tuple(code_deps)
        self.retries = int(retries)
        self.signature = inspect.signature(func)
        deps = tuple(d.func if isinstance(d, Stage) else d for d in self.code_deps)
        self.code_hash = hashing.code_hash(func, deps)

    def __call__(self, *args, **kwargs):
        ba = self.signature.bind(*args, **kwargs)
        ba.apply_defaults()
        return Node(self, dict(ba.arguments))

    def __reduce__(self):
        return (_resolve_stage, (self.module, self.name))

    def __repr__(self):
        return f"<flowr.stage {self.module}.{self.name} v{self.version}>"


def stage(func=None, *, version="0", code_deps=(), retries=0):
    """Decorator turning a top-level function into a flowr stage.

    Usable bare (``@flowr.stage``) or with arguments
    (``@flowr.stage(version="2", code_deps=[helper], retries=1)``).
    """
    def wrap(f):
        return Stage(f, version=version, code_deps=code_deps, retries=retries)

    return wrap(func) if func is not None else wrap
