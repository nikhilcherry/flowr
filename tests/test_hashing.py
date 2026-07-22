"""Phase 1: cache-key rules — AST normalization and param canonicalization."""

import pytest

import flowr
from flowr.errors import CacheKeyError, FlowrError, SourceUnavailableError
from flowr.hashing import canonical_bytes, code_hash


BASE = """
    import flowr

    @flowr.stage
    def f(x, window=0.5):
        y = x * 2
        return y + window
"""


def _key(node):
    return node.key_with([])


# ---------------------------------------------------------------- AST norm

def test_comment_edit_does_not_change_code_hash(make_module):
    a = make_module(BASE, name="m_comment")
    h1 = a.f.code_hash
    a = make_module(BASE.replace("y = x * 2", "y = x * 2  # a comment"),
                    name="m_comment")
    assert a.f.code_hash == h1


def test_docstring_and_whitespace_do_not_change_code_hash(make_module):
    a = make_module(BASE, name="m_doc")
    h1 = a.f.code_hash
    edited = BASE.replace(
        "def f(x, window=0.5):",
        'def f(x, window=0.5):\n        "younger, better docstring"\n',
    )
    a = make_module(edited, name="m_doc")
    assert a.f.code_hash == h1


def test_logic_edit_changes_code_hash(make_module):
    a = make_module(BASE, name="m_logic")
    h1 = a.f.code_hash
    a = make_module(BASE.replace("y = x * 2", "y = x * 3"), name="m_logic")
    assert a.f.code_hash != h1


def test_decorator_args_do_not_change_code_hash(make_module):
    a = make_module(BASE, name="m_dec")
    h1 = a.f.code_hash
    a = make_module(BASE.replace("@flowr.stage", "@flowr.stage(retries=2)"),
                    name="m_dec")
    assert a.f.code_hash == h1


def test_version_changes_node_key_not_code_hash(make_module):
    a = make_module(BASE, name="m_ver")
    k1, h1 = _key(a.f(1)), a.f.code_hash
    a = make_module(BASE.replace("@flowr.stage", '@flowr.stage(version="2")'),
                    name="m_ver")
    assert a.f.code_hash == h1
    assert _key(a.f(1)) != k1


def test_code_deps_fold_helper_source_into_hash(make_module):
    src = """
        import flowr

        def helper(v):
            return v + 1

        @flowr.stage(code_deps=[helper])
        def g(x):
            return helper(x)

        @flowr.stage
        def h(x):
            return helper(x)
    """
    a = make_module(src, name="m_deps")
    g1, h1 = a.g.code_hash, a.h.code_hash
    a = make_module(src.replace("return v + 1", "return v + 2"), name="m_deps")
    assert a.g.code_hash != g1      # declared dep -> invalidates
    assert a.h.code_hash == h1      # undeclared helper -> invisible (documented)


def test_source_unavailable_raises_clear_error():
    ns = {}
    exec("def repl_func(x):\n    return x\n", ns)
    with pytest.raises(SourceUnavailableError, match="importable"):
        flowr.stage(ns["repl_func"])


def test_lambda_and_nested_functions_rejected():
    with pytest.raises(FlowrError, match="top-level"):
        flowr.stage(lambda x: x)

    def outer():
        def inner(x):
            return x
        return inner

    with pytest.raises(FlowrError, match="top-level"):
        flowr.stage(outer())


# ------------------------------------------------------------ param canon

def test_default_vs_explicit_param_hash_identically(make_module):
    a = make_module(BASE, name="m_default")
    assert _key(a.f(3)) == _key(a.f(3, window=0.5))
    assert _key(a.f(3)) == _key(a.f(3, 0.5))          # positional too
    assert _key(a.f(3)) != _key(a.f(3, window=0.6))
    assert _key(a.f(x=3)) == _key(a.f(3))             # kw vs positional


def test_dict_param_order_is_irrelevant(make_module):
    a = make_module(BASE, name="m_dict")
    assert _key(a.f({"a": 1, "b": 2})) == _key(a.f({"b": 2, "a": 1}))


def test_type_distinctions():
    assert canonical_bytes(True, arg="a", stage="s") != canonical_bytes(1, arg="a", stage="s")
    assert canonical_bytes(1, arg="a", stage="s") != canonical_bytes(1.0, arg="a", stage="s")
    assert canonical_bytes([1, 2], arg="a", stage="s") != canonical_bytes((1, 2), arg="a", stage="s")
    assert canonical_bytes("1", arg="a", stage="s") != canonical_bytes(1, arg="a", stage="s")
    assert canonical_bytes(None, arg="a", stage="s") != canonical_bytes(False, arg="a", stage="s")


def test_float_canonical_via_hex():
    assert canonical_bytes(0.1 + 0.2, arg="a", stage="s") == canonical_bytes(
        0.30000000000000004, arg="a", stage="s")
    assert canonical_bytes(0.3, arg="a", stage="s") != canonical_bytes(
        0.1 + 0.2, arg="a", stage="s")


def test_bytes_and_nested_containers():
    v1 = {"k": [b"\x00\x01", (1, 2.5, None)], "z": "txt"}
    v2 = {"z": "txt", "k": [b"\x00\x01", (1, 2.5, None)]}
    assert canonical_bytes(v1, arg="a", stage="s") == canonical_bytes(v2, arg="a", stage="s")


def test_unhashable_param_is_hard_error_naming_arg_and_stage(make_module):
    a = make_module(BASE, name="m_err")

    class Weird:
        pass

    with pytest.raises(CacheKeyError) as ei:
        a.f(Weird())
    msg = str(ei.value)
    assert "'x'" in msg and "'f'" in msg and "Weird" in msg


def test_sets_rejected(make_module):
    a = make_module(BASE, name="m_set")
    with pytest.raises(CacheKeyError):
        a.f({1, 2, 3})


def test_file_param_hashes_by_content_not_path(tmp_path, make_module):
    a = make_module(BASE, name="m_file")
    p1, p2 = tmp_path / "one.dat", tmp_path / "two.dat"
    p1.write_bytes(b"same content")
    p2.write_bytes(b"same content")
    assert _key(a.f(flowr.File(p1))) == _key(a.f(flowr.File(p2)))
    p2.write_bytes(b"different content")
    assert _key(a.f(flowr.File(p1))) != _key(a.f(flowr.File(p2)))


def test_missing_file_raises(make_module):
    a = make_module(BASE, name="m_file_missing")
    with pytest.raises(FlowrError, match="does not exist"):
        a.f(flowr.File("/nonexistent/nowhere.bin"))


def test_plain_string_path_is_opaque(tmp_path, make_module):
    a = make_module(BASE, name="m_opaque")
    p = tmp_path / "data.bin"
    p.write_bytes(b"v1")
    k1 = _key(a.f(str(p)))
    p.write_bytes(b"v2")
    assert _key(a.f(str(p))) == k1  # documented Make-style blind spot


def test_numpy_param_hashing():
    np = pytest.importorskip("numpy")
    a1 = canonical_bytes(np.arange(6, dtype="f8"), arg="a", stage="s")
    a2 = canonical_bytes(np.arange(6, dtype="f8"), arg="a", stage="s")
    a3 = canonical_bytes(np.arange(6, dtype="f4"), arg="a", stage="s")
    a4 = canonical_bytes(np.arange(6, dtype="f8").reshape(2, 3), arg="a", stage="s")
    assert a1 == a2 and a1 != a3 and a1 != a4
    with pytest.raises(CacheKeyError):
        canonical_bytes(np.array([object()]), arg="a", stage="s")


# ------------------------------------------------------------ node keys

def test_upstream_hash_enters_key(make_module):
    a = make_module(BASE, name="m_up")
    n = a.f(1)
    child = a.f(n)
    assert child.edges == [n]
    assert child.key_with(["aaa"]) != child.key_with(["bbb"])


def test_nodes_nested_in_lists_are_edges(make_module):
    a = make_module(BASE, name="m_nest")
    n1, n2 = a.f(1), a.f(2)
    agg = a.f([n1, n2])
    assert agg.edges == [n1, n2]
    # same node passed twice dedupes to one edge
    agg2 = a.f([n1, n1])
    assert agg2.edges == [n1]


def test_code_hash_survives_stage_pickle_roundtrip(make_module):
    import pickle
    a = make_module(BASE, name="m_pickle_rt")
    st = pickle.loads(pickle.dumps(a.f))
    assert st is a.f
