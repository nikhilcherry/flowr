"""flowr exception hierarchy."""


class FlowrError(Exception):
    """Base class for every error flowr raises deliberately."""


class CacheKeyError(FlowrError, TypeError):
    """A stage argument cannot be canonically hashed into a cache key."""


class SourceUnavailableError(FlowrError):
    """A stage or code_dep has no readable source file (REPL, exec'd string)."""


class GraphCycleError(FlowrError):
    """The composed graph contains a cycle."""


class RunError(FlowrError):
    """One or more nodes failed during flowr.run().

    Attributes:
        failures: list of (stage_name, node_key, traceback_str)
        n_blocked: number of downstream nodes skipped because an ancestor failed
    """

    def __init__(self, message, failures=(), n_blocked=0):
        super().__init__(message)
        self.failures = list(failures)
        self.n_blocked = n_blocked
