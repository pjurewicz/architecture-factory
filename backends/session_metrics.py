"""Per-connection call metrics (1.8-14).

``AuditMiddleware`` records every top-level tool call of a connection here -
the call, its tool, the bytes of text it returned, its error kind and its
latency - and ``system_status(metrics=True)`` reads the connection's snapshot.
Nothing is reported unless asked: the default status payload and the idle
catalog pay nothing for it.

Everything is bounded and in memory: at most :data:`MAX_SESSIONS` connections
(the oldest is forgotten first) and :data:`MAX_SAMPLES` latencies per
connection. Recording never fails a call.
"""

from __future__ import annotations

import collections
import math
import threading
from collections.abc import Iterable
from typing import Any

#: Connections remembered at once; a server with more forgets the oldest.
MAX_SESSIONS = 64

#: Latency samples kept per connection for p50/p95.
MAX_SAMPLES = 2048


def percentile(values: Iterable[float], q: float) -> float | None:
    """The nearest-rank ``q``-th percentile, or None with no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def result_text_bytes(result: Any) -> int:
    """UTF-8 bytes of the text blocks a tool result hands the client."""
    total = 0
    for block in getattr(result, "content", None) or ():
        text = getattr(block, "text", None)
        if isinstance(text, str):
            total += len(text.encode("utf-8"))
    return total


class SessionMetrics:
    """One connection's counters."""

    __slots__ = ("calls", "by_tool", "result_bytes", "errors_by_kind", "latencies")

    def __init__(self) -> None:
        self.calls = 0
        self.by_tool: collections.Counter[str] = collections.Counter()
        self.result_bytes = 0
        self.errors_by_kind: collections.Counter[str] = collections.Counter()
        self.latencies: collections.deque[float] = collections.deque(maxlen=MAX_SAMPLES)

    def record(
        self, tool: str, elapsed_ms: float, result_bytes: int, error_kind: str | None
    ) -> None:
        self.calls += 1
        self.by_tool[tool] += 1
        self.result_bytes += int(result_bytes)
        if error_kind:
            self.errors_by_kind[error_kind] += 1
        self.latencies.append(float(elapsed_ms))

    def snapshot(self) -> dict[str, Any]:
        samples = list(self.latencies)
        p50 = percentile(samples, 50)
        p95 = percentile(samples, 95)
        return {
            "calls": self.calls,
            "by_tool": dict(sorted(self.by_tool.items())),
            "result_bytes": self.result_bytes,
            "errors_by_kind": dict(sorted(self.errors_by_kind.items())),
            "p50_ms": None if p50 is None else round(p50, 3),
            "p95_ms": None if p95 is None else round(p95, 3),
            "window": len(samples),
        }


class SessionMetricsRegistry:
    """Connection id -> :class:`SessionMetrics`, least recently used forgotten first."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max = max(1, int(max_sessions))
        self._sessions: collections.OrderedDict[str, SessionMetrics] = collections.OrderedDict()
        self._lock = threading.Lock()

    def record(
        self,
        session: str,
        tool: str,
        elapsed_ms: float,
        result_bytes: int,
        error_kind: str | None = None,
    ) -> None:
        with self._lock:
            metrics = self._sessions.get(session)
            if metrics is None:
                metrics = self._sessions[session] = SessionMetrics()
                while len(self._sessions) > self._max:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session)
            metrics.record(tool, elapsed_ms, result_bytes, error_kind)

    def snapshot(self, session: str) -> dict[str, Any]:
        """The connection's counters; an unknown connection reads as all zero."""
        with self._lock:
            metrics = self._sessions.get(session)
            return (metrics or SessionMetrics()).snapshot()

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


#: The server's registry (one per process).
SESSION_METRICS = SessionMetricsRegistry()
