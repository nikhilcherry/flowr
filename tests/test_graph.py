"""Phase 3a: topo ordering, dedup, cycle detection."""

import pytest

from flowr.errors import GraphCycleError
from flowr.graph import closure

SRC = """
    import flowr

    @flowr.stage
    def s(x):
        return x
"""


def test_parents_before_children_diamond(make_module):
    m = make_module(SRC)
    root = m.s(0)
    left, right = m.s(root), m.s(root)
    top = m.s([left, right])
    order = closure([top])
    pos = {id(n): i for i, n in enumerate(order)}
    assert len(order) == 4                       # root deduplicated
    assert pos[id(root)] < pos[id(left)] < pos[id(top)]
    assert pos[id(root)] < pos[id(right)] < pos[id(top)]


def test_multiple_targets_share_closure(make_module):
    m = make_module(SRC)
    root = m.s(0)
    a, b = m.s(root), m.s(root)
    order = closure([a, b])
    assert len(order) == 3


def test_cycle_detection_names_stages(make_module):
    m = make_module(SRC)
    a = m.s(1)
    b = m.s(a)
    a.edges.append(b)  # sabotage: graphs built via the API cannot do this
    with pytest.raises(GraphCycleError, match="s -> s"):
        closure([b])
