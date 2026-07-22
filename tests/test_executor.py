"""Phase 3b: execution — caching, invalidation, dry-run reasons, retries,
blocked subtrees, early cutoff."""

import pytest

import flowr
from flowr.errors import RunError

PIPE = """
    import flowr, os

    def _count(tag):
        path = os.environ.get("TEST_COUNTER")
        if path:
            with open(path, "a") as f:
                f.write(tag + "\\n")

    @flowr.stage
    def source(i, base=10):
        _count("source")
        return i + base

    @flowr.stage
    def double(v, factor=2):
        _count("double")
        return v * factor

    @flowr.stage
    def total(vs, offset=0):
        _count("total")
        return sum(vs) + offset
"""


@pytest.fixture
def counter(tmp_path, monkeypatch):
    path = tmp_path / "exec-counter"
    monkeypatch.setenv("TEST_COUNTER", str(path))

    def read():
        if not path.exists():
            return []
        return path.read_text().splitlines()

    return read


def test_run_executes_and_returns_value(store_dir, make_module, counter):
    m = make_module(PIPE)
    assert flowr.run(m.double(m.source(1))) == 22
    assert counter() == ["source", "double"]


def test_second_run_is_all_hits_zero_execution(store_dir, make_module, counter):
    m = make_module(PIPE)
    node = m.total([m.double(m.source(i)) for i in range(3)])
    v1 = flowr.run(node)
    n_first = len(counter())
    assert n_first == 7
    # rebuild the graph from scratch — still all hits
    node2 = m.total([m.double(m.source(i)) for i in range(3)])
    assert flowr.run(node2) == v1
    assert len(counter()) == n_first          # zero stage code executed
    plan = flowr.run(node2, dry=True)
    assert plan.n_misses == 0 and plan.n_hits == 7


def test_param_change_invalidates_only_downstream(store_dir, make_module, counter):
    m = make_module(PIPE)
    flowr.run(m.total([m.double(m.source(i)) for i in range(3)]))
    before = len(counter())
    changed = m.total([m.double(m.source(i)) for i in range(3)], offset=5)
    plan = flowr.run(changed, dry=True)
    assert plan.n_misses == 1
    assert plan.misses[0].stage_name == "total"
    assert "param offset: 0 -> 5" in plan.misses[0].reason
    flowr.run(changed)
    assert counter()[before:] == ["total"]    # only total re-ran


def test_list_of_targets_order_preserving(store_dir, make_module):
    m = make_module(PIPE)
    nodes = [m.source(i) for i in (5, 1, 3)]
    assert flowr.run(nodes) == [15, 11, 13]


def test_dry_run_reasons(store_dir, make_module, tmp_path):
    name = "m_reasons"
    src = PIPE + """
    @flowr.stage
    def load(f, scale=1):
        _count("load")
        return scale
    """
    m = make_module(src, name=name)
    data = tmp_path / "in.dat"
    data.write_bytes(b"v1")

    # new node (and its downstream is an upstream miss)
    node = m.double(m.load(flowr.File(data)))
    plan = flowr.run(node, dry=True)
    reasons = {e.stage_name: e.reason for e in plan.misses}
    assert reasons == {"load": "new node", "double": "upstream miss"}
    flowr.run(node)

    # code changed
    m = make_module(src.replace("return v * factor", "return v * factor + 0"),
                    name=name)
    node = m.double(m.load(flowr.File(data)))
    plan = flowr.run(node, dry=True)
    assert [e.stage_name for e in plan.misses] == ["double"]
    assert plan.misses[0].reason == "code changed"

    # file content changed + upstream miss
    m = make_module(src, name=name)
    data.write_bytes(b"v2")
    node = m.double(m.load(flowr.File(data)))
    plan = flowr.run(node, dry=True)
    reasons = {e.stage_name: e.reason for e in plan.misses}
    assert "file content changed" in reasons["load"] and str(data) in reasons["load"]
    assert reasons["double"] == "upstream miss"


def test_early_cutoff_shares_downstream_cache(store_dir, make_module, counter):
    name = "m_cutoff"
    m = make_module(PIPE, name=name)
    flowr.run(m.double(m.source(1)))
    assert counter() == ["source", "double"]
    # different source code, byte-identical result -> double still hits
    m = make_module(PIPE.replace("return i + base", "return base + i"), name=name)
    plan = flowr.run(m.double(m.source(1)), dry=True)
    assert [e.stage_name for e in plan.misses] == ["source", "double"]
    flowr.run(m.double(m.source(1)))
    assert counter() == ["source", "double", "source"]   # double: early cutoff hit


def test_duplicate_equivalent_nodes_execute_once(store_dir, make_module, counter):
    m = make_module(PIPE)
    n1, n2 = m.source(7), m.source(7)     # distinct objects, same key
    assert flowr.run([n1, n2]) == [17, 17]
    assert counter() == ["source"]


def test_retries(store_dir, make_module, counter, tmp_path):
    flaky = tmp_path / "flaky-attempts"
    src = f"""
    import flowr, os

    @flowr.stage(retries=2)
    def flaky(x):
        path = {str(flaky)!r}
        n = 0
        if os.path.exists(path):
            n = len(open(path).read())
        open(path, "a").write("x")
        if n < 2:
            raise RuntimeError("transient")
        return x
    """
    m = make_module(src)
    assert flowr.run(m.flaky(9)) == 9
    assert flaky.read_text() == "xxx"     # failed twice, third attempt won


def test_failure_blocks_subtree_but_independent_branch_completes(
        store_dir, make_module, counter):
    src = PIPE + """
    @flowr.stage
    def boom(v):
        _count("boom")
        raise ValueError("kaput")
    """
    m = make_module(src)
    bad = m.double(m.boom(m.source(1)))
    good = m.double(m.source(100))
    with pytest.raises(RunError) as ei:
        flowr.run([bad, good])
    err = ei.value
    assert len(err.failures) == 1
    assert err.failures[0][0] == "boom"
    assert "kaput" in err.failures[0][2]
    assert err.n_blocked == 1                       # the double() above boom
    # independent branch finished and is cached:
    assert flowr.run(good, dry=True).n_misses == 0


def test_fail_fast(store_dir, make_module):
    src = PIPE + """
    @flowr.stage
    def boom(v):
        raise ValueError("kaput")
    """
    m = make_module(src)
    with pytest.raises(RunError):
        flowr.run(m.double(m.boom(m.source(1))), fail_fast=True)


def test_run_rejects_non_nodes():
    with pytest.raises(flowr.FlowrError, match="expects a Node"):
        flowr.run(42)


def test_run_empty_list(store_dir):
    assert flowr.run([]) == []


def test_map_builds_one_node_per_item(store_dir, make_module):
    m = make_module(PIPE)
    nodes = flowr.map(m.source, [1, 2, 3], base=100)
    assert [n.stage_name for n in nodes] == ["source"] * 3
    assert flowr.run(nodes) == [101, 102, 103]
    with pytest.raises(flowr.FlowrError, match="flowr.map expects"):
        flowr.map(lambda x: x, [1])


def test_version_bump_forces_miss(store_dir, make_module):
    name = "m_bump"
    m = make_module(PIPE, name=name)
    flowr.run(m.source(1))
    m = make_module(
        PIPE.replace("@flowr.stage\n    def source", '@flowr.stage(version="2")\n    def source'),
        name=name)
    plan = flowr.run(m.source(1), dry=True)
    assert plan.n_misses == 1
    assert "stage version: 0 -> 2" in plan.misses[0].reason
