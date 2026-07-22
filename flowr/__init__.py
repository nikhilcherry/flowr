"""flowr — deterministic DAG pipeline runner with content-addressed caching.

Stages are plain Python functions; composing them lazily defines a DAG;
every stage result is cached by a content hash of (code + params + inputs).

    import flowr

    @flowr.stage
    def detrend(path, window=0.5):
        ...

    lc = detrend(flowr.File("data/tic123.npz"))   # lazy: returns a Node
    result = flowr.run(lc)                        # executes cache misses only

Install: pip install git+https://github.com/nikhilcherry/flowr
"""

from .errors import (
    CacheKeyError,
    FlowrError,
    GraphCycleError,
    RunError,
    SourceUnavailableError,
)
from .executor import run
from .graph import Plan, PlanEntry
from .node import File, Node, Stage, stage
from .store import Store, collect_garbage as gc

__version__ = "0.1.0"


def map(stage, items, **kwargs):
    """Fan out: one lazily-cached Node per item.

    ``flowr.map(clean, series_nodes, window=9)`` is exactly
    ``[clean(item, window=9) for item in series_nodes]`` — each item gets its
    own node, cache entry, and invalidation.
    """
    if not isinstance(stage, Stage):
        raise FlowrError(
            "flowr.map expects a @flowr.stage-decorated function as its "
            f"first argument; got {type(stage).__name__}"
        )
    return [stage(item, **kwargs) for item in items]


__all__ = [
    "CacheKeyError", "File", "FlowrError", "GraphCycleError", "Node", "Plan",
    "PlanEntry", "RunError", "SourceUnavailableError", "Stage", "Store",
    "gc", "map", "run", "stage", "__version__",
]
