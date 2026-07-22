"""Cache-key computation: AST-normalized code hashes and canonical parameter hashing.

The node key is:

    node_key = sha256(code_hash + stage_version + param_hash + upstream_result_hashes)

- code_hash ignores comments, docstrings, whitespace and the outer decorator
  (AST-normalize: parse -> strip -> unparse -> sha256).
- param_hash is a canonical, type-tagged serialization of the *bound* call
  arguments (defaults applied), so ``f(x, w=0.5)`` and ``f(x)`` hash
  identically when 0.5 is the default.
- Unsupported parameter types are a hard error naming the argument and stage;
  flowr never silently str()-hashes arbitrary objects.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

from .errors import CacheKeyError, SourceUnavailableError

try:
    import numpy as _np
except ImportError:  # numpy is optional everywhere in flowr
    _np = None

_CHUNK = 1 << 20


def file_sha256(path):
    """Streamed sha256 of a file's content, as hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def normalized_source(obj, *, strip_outer_decorators=False):
    """AST-normalized source of a function or module.

    Comments, docstrings and formatting do not survive normalization; logic
    does. With ``strip_outer_decorators`` the decorator list of top-level
    function definitions is dropped (so editing ``@flowr.stage(...)`` args
    never invalidates by itself — version/retries are folded in separately).
    """
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError) as e:
        raise SourceUnavailableError(
            f"flowr cannot read the source of {obj!r}. Stages and code_deps "
            "must be plain functions or modules defined in importable .py "
            "files — not in a REPL, notebook cell without a file, or exec'd "
            "string."
        ) from e
    tree = ast.parse(textwrap.dedent(src))
    if strip_outer_decorators:
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmt.decorator_list = []
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def code_hash(func, code_deps=()):
    """sha256 over the normalized source of ``func`` plus each explicit dep."""
    h = hashlib.sha256()
    h.update(normalized_source(func, strip_outer_decorators=True).encode())
    for dep in code_deps:
        h.update(b"\x00dep\x00")
        h.update(normalized_source(dep).encode())
    return h.hexdigest()


def _enc(tag, payload):
    return tag + len(payload).to_bytes(8, "big") + payload


def canonical_bytes(value, *, arg, stage):
    """Canonical type-tagged byte encoding of one parameter value.

    Raises CacheKeyError (naming the argument and stage) for any type that is
    not explicitly supported.
    """
    from .node import EdgeRef, File  # local import: node.py imports this module

    if value is None:
        return b"N"
    if value is True:
        return b"T"
    if value is False:
        return b"F"
    t = type(value)
    if t is int:
        return _enc(b"i", repr(value).encode())
    if t is float:
        return _enc(b"f", value.hex().encode())
    if t is str:
        return _enc(b"s", value.encode())
    if t is bytes:
        return _enc(b"y", value)
    if isinstance(value, EdgeRef):
        return _enc(b"E", repr(int(value)).encode())
    if isinstance(value, File):
        return _enc(b"P", value.content_hash.encode())
    if t is list or t is tuple:
        tag = b"l" if t is list else b"t"
        return _enc(tag, b"".join(
            canonical_bytes(v, arg=arg, stage=stage) for v in value))
    if t is dict:
        pairs = []
        for k, v in value.items():
            kb = canonical_bytes(k, arg=arg, stage=stage)
            vb = canonical_bytes(v, arg=arg, stage=stage)
            pairs.append(kb + vb)
        return _enc(b"d", b"".join(sorted(pairs)))
    if _np is not None:
        if isinstance(value, _np.ndarray):
            if value.dtype == object:
                raise CacheKeyError(
                    f"Cannot build a cache key for argument '{arg}' of stage "
                    f"'{stage}': object-dtype numpy arrays are not hashable. "
                    "Use a concrete dtype, or restructure the argument."
                )
            a = _np.ascontiguousarray(value)
            head = a.dtype.str.encode() + b"|" + repr(a.shape).encode() + b"|"
            return _enc(b"a", head + a.tobytes())
        if isinstance(value, _np.generic):
            return _enc(b"g", value.dtype.str.encode() + b"|" + value.tobytes())
    supported = "None, bool, int, float, str, bytes, list, tuple, dict, flowr.File"
    if _np is not None:
        supported += ", numpy.ndarray, numpy scalars"
    raise CacheKeyError(
        f"Cannot build a cache key for argument '{arg}' of stage '{stage}': "
        f"unsupported type '{type(value).__name__}'. Supported parameter "
        f"types: {supported}. Wrap filesystem inputs in flowr.File; convert "
        "anything else explicitly — flowr never guesses how to hash an object."
    )


def params_digest(arguments, stage_name):
    """Digest (raw bytes) over an ordered mapping of bound argument values."""
    h = hashlib.sha256()
    for name, value in arguments.items():
        h.update(_enc(b"k", name.encode()))
        h.update(canonical_bytes(value, arg=name, stage=stage_name))
    return h.digest()


def json_repr(value):
    """JSON-safe display form of a parameter value (used for `flowr why`
    output and dry-run diffing; NOT part of the cache key)."""
    from .node import EdgeRef, File

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, EdgeRef):  # EdgeRef subclasses int
            return {"$node": int(value)}
        return value
    if isinstance(value, bytes):
        return {"$bytes": hashlib.sha256(value).hexdigest(), "len": len(value)}
    if isinstance(value, File):
        return {"$file": value.path, "sha256": value.content_hash}
    if isinstance(value, (list, tuple)):
        return [json_repr(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: json_repr(v) for k, v in value.items()}
        return {"$dict": [[json_repr(k), json_repr(v)] for k, v in value.items()]}
    if _np is not None and isinstance(value, _np.ndarray):
        a = _np.ascontiguousarray(value)
        return {
            "$ndarray": {
                "dtype": a.dtype.str,
                "shape": list(a.shape),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            }
        }
    if _np is not None and isinstance(value, _np.generic):
        return {"$npscalar": {"dtype": value.dtype.str, "repr": repr(value.item())}}
    return {"$unrepresentable": type(value).__name__}
