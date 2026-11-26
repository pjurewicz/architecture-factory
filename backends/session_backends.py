"""One headless engine per MCP session (SESSION_ISOLATION).

The headless engine keeps its open documents, the active one, undo and
transactions on the backend object, and a server used to build exactly one.
Every connection then shared one workspace: two clients drawing at once wrote
into whichever document the other had made active, with no error. With
isolation on, the lifespan puts a ``SessionBackends`` router where the backend
used to be; ``SessionBackendMiddleware`` binds the calling session's own engine
for each request, and every ``lifespan_context["backend"]`` read resolves to
it. The first session gets the engine the lifespan built, which is also what
a read outside any request sees, so a server with one client behaves exactly
as before; every later session gets a new engine.

A session's engine is released with its unsaved documents when the client
ends the session (an HTTP ``DELETE`` carrying its ``mcp-session-id``,
``EndSessionOnDelete``); the first session's engine is emptied instead, and
the next session starts on it. A client that vanishes without ending its
session leaves its engine until ``SESSION_IDLE_TIMEOUT`` seconds pass without a
request (``0`` keeps it until shutdown); the first session's engine is never
released for idleness, as it always was.
"""

from __future__ import annotations

import contextvars
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

#: The engine bound for the current request (``SessionBackendMiddleware``).
CURRENT_BACKEND: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "acadmcp_session_backend", default=None
)


class _Entry:
    __slots__ = ("backend", "last_used", "in_flight", "is_default")

    def __init__(self, backend, is_default=False):
        self.backend = backend
        self.is_default = is_default
        self.last_used = time.monotonic()
        self.in_flight = 0


class SessionBackends:
    """Routes backend access to the calling session's own engine."""

    def __init__(
        self,
        default,
        factory: Callable[[], Awaitable[Any]],
        release: Callable[[Any], Awaitable[None]],
        idle_timeout: float,
    ):
        object.__setattr__(self, "_default", default)
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_release", release)
        object.__setattr__(self, "_idle_timeout", float(idle_timeout))
        object.__setattr__(self, "_sessions", {})

    # -- the router's own interface ------------------------------------------------

    async def acquire(self, session: str | None):
        """The engine of ``session``, created on its first request; marks it in use."""
        await self._evict_idle()
        if session is None:
            return self._default, None
        entry = self._sessions.get(session)
        if entry is None:
            if not any(e.is_default for e in self._sessions.values()):
                entry = _Entry(self._default, is_default=True)
            else:
                entry = _Entry(await self._factory())
            self._sessions[session] = entry
            log.info("session %s: new headless engine (%d open)", session, len(self._sessions))
        entry.in_flight += 1
        entry.last_used = time.monotonic()
        return entry.backend, entry

    def release_entry(self, entry: _Entry | None) -> None:
        if entry is not None:
            entry.in_flight -= 1
            entry.last_used = time.monotonic()

    def engines(self) -> list[tuple[str | None, Any, float]]:
        """``(session, engine, last_used)`` per session, the default engine's session first.

        The default engine is listed once with session None when no session holds it.
        """
        rows = [(key, e.backend, e.last_used) for key, e in self._sessions.items()]
        rows.sort(key=lambda row: row[1] is not self._default)
        if not any(row[1] is self._default for row in rows):
            rows.insert(0, (None, self._default, 0.0))
        return rows

    def session_count(self) -> int:
        return len(self._sessions)

    async def _evict_idle(self) -> None:
        if self._idle_timeout <= 0:
            return
        now = time.monotonic()
        stale = [
            key
            for key, entry in self._sessions.items()
            if not entry.is_default
            and entry.in_flight == 0
            and now - entry.last_used > self._idle_timeout
        ]
        for key in stale:
            entry = self._sessions.pop(key)
            log.info("session %s: idle for %.0fs, engine released", key, now - entry.last_used)
            await self._release(entry.backend)

    async def end_session(self, session: str) -> bool:
        """The client ended ``session``: release its engine now; True when one was held.

        The default engine is emptied, not released, and goes to the next new
        session. An engine still inside a call is left to the idle sweep.
        """
        entry = self._sessions.get(session)
        if entry is None or entry.in_flight:
            return False
        del self._sessions[session]
        await self._release(entry.backend)
        if entry.is_default:
            await entry.backend.connect()
        log.info("session %s: ended by the client, engine released", session)
        return True

    async def disconnect(self) -> None:
        """Shutdown: release every session's engine, then the default one."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for entry in sessions:
            if not entry.is_default:
                await self._release(entry.backend)
        await self._release(self._default)

    # -- everything else is the bound engine's ---------------------------------------

    def _bound(self):
        return CURRENT_BACKEND.get() or self._default

    def __getattr__(self, name):
        return getattr(self._bound(), name)

    def __setattr__(self, name, value):
        setattr(self._bound(), name, value)


class EndSessionOnDelete:
    """ASGI wrapper: a successful ``DELETE`` of an MCP session ends its engine too.

    ``router()`` returns the server's ``SessionBackends`` or None (isolation
    off, a live engine, no lifespan yet). The session id is the
    ``mcp-session-id`` header, the same key ``SessionBackendMiddleware`` used.
    """

    def __init__(self, app, router: Callable[[], Any]):
        self.app = app
        self._router = router

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "DELETE":
            await self.app(scope, receive, send)
            return
        session = None
        for name, value in scope.get("headers") or ():
            if name.lower() == b"mcp-session-id":
                session = value.decode("latin-1")
        status: list[int] = []

        async def _send(message):
            if message.get("type") == "http.response.start":
                status.append(int(message.get("status", 0)))
            await send(message)

        await self.app(scope, receive, _send)
        router = self._router()
        if session and status and 200 <= status[0] < 300 and router is not None:
            try:
                await router.end_session(session)
            except Exception as exc:  # the client's DELETE already succeeded
                log.warning("session %s: engine release failed: %s", session, exc)
