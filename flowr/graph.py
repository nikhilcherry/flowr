"""Graph utilities: closure/topological ordering, cycle detection, and the
dry-run Plan object."""

from __future__ import annotations

from .errors import GraphCycleError
from .node import Node

HIT = "HIT"
MISS = "MISS"


def closure(targets):
    """Topologically ordered closure of the target Nodes (parents strictly
    before children, deduplicated by object identity).

    Nodes are immutable so cycles cannot normally form, but detection is
    kept as a hard guarantee — the error names the stages on the cycle.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    order = []
    for root in targets:
        if color.get(id(root), WHITE) == BLACK:
            continue
        path = []
        stack = [(root, 0)]
        while stack:
            node, idx = stack.pop()
            if idx == 0:
                if color.get(id(node), WHITE) == BLACK:
                    continue
                color[id(node)] = GRAY
                path.append(node)
            if idx < len(node.edges):
                stack.append((node, idx + 1))
                child = node.edges[idx]
                c = color.get(id(child), WHITE)
                if c == GRAY:
                    i = next(j for j, p in enumerate(path) if p is child)
                    names = " -> ".join(p.stage_name for p in path[i:])
                    raise GraphCycleError(
                        f"the pipeline graph contains a cycle: "
                        f"{names} -> {child.stage_name}"
                    )
                if c == WHITE:
                    stack.append((child, 0))
            else:
                color[id(node)] = BLACK
                path.pop()
                order.append(node)
    return order


class PlanEntry:
    """One node's planned fate: HIT, or MISS with a human-readable reason."""

    __slots__ = ("node", "stage_name", "node_key", "status", "reason")

    def __init__(self, node, node_key, status, reason=None):
        self.node = node
        self.stage_name = node.stage_name
        self.node_key = node_key
        self.status = status
        self.reason = reason

    def __repr__(self):
        key = (self.node_key or "?")[:12]
        tail = f"  ({self.reason})" if self.reason else ""
        return f"{self.status:<4} {self.stage_name}  {key}{tail}"


class Plan:
    """Result of ``flowr.run(..., dry=True)``: what would execute and why."""

    def __init__(self, entries):
        self.entries = list(entries)

    @property
    def hits(self):
        return [e for e in self.entries if e.status == HIT]

    @property
    def misses(self):
        return [e for e in self.entries if e.status == MISS]

    @property
    def n_hits(self):
        return len(self.hits)

    @property
    def n_misses(self):
        return len(self.misses)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __str__(self):
        lines = [
            f"flowr plan: {len(self.entries)} nodes — "
            f"{self.n_hits} HIT, {self.n_misses} MISS"
        ]
        per_stage_hits = {}
        for e in self.hits:
            per_stage_hits[e.stage_name] = per_stage_hits.get(e.stage_name, 0) + 1
        for name in sorted(per_stage_hits):
            lines.append(f"  HIT   {name} × {per_stage_hits[name]}")
        for e in self.misses:
            key = (e.node_key or "?")[:12]
            lines.append(f"  MISS  {e.stage_name}  {key}  — {e.reason}")
        return "\n".join(lines)
