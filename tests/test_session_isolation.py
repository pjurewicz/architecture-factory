"""SESSION_ISOLATION: each MCP session of a headless server has its own engine.

Two clients connected at once used to share one workspace: the second
client's ``drawing_new`` became the active document for both, and the first
client's next line landed in it without an error.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastmcp import Client

import config
import server

LINE = {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0}


def _data(result) -> dict:
    return result.structured_content or json.loads(result.content[0].text)


async def _entities(client) -> int:
    return _data(await client.call_tool("drawing_info", {}))["entity_count"]


async def _documents(client) -> list[str]:
    return [d["key"] for d in _data(await client.call_tool("document_list", {}))["documents"]]


@pytest.mark.asyncio
async def test_sessions_draw_into_their_own_documents():
    async with Client(server.mcp) as a, Client(server.mcp) as b:
        await a.call_tool("drawing_new", {})
        assert await _documents(b) == []
        await b.call_tool("drawing_new", {})
        await a.call_tool("entity_create_line", LINE)
        await a.call_tool("entity_create_line", LINE)
        await b.call_tool("entity_create_circle", {"cx": 0.0, "cy": 0.0, "radius": 1.0})
        assert await _entities(a) == 2
        assert await _entities(b) == 1
        assert len(await _documents(a)) == 1


@pytest.mark.asyncio
async def test_isolation_off_shares_one_workspace(monkeypatch):
    monkeypatch.setattr(config.settings, "session_isolation", False)
    async with Client(server.mcp) as a, Client(server.mcp) as b:
        await a.call_tool("drawing_new", {})
        await b.call_tool("drawing_new", {})
        await a.call_tool("entity_create_line", LINE)
        assert len(await _documents(a)) == 2
        assert await _entities(b) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_timeout, kept", [(60.0, True), (0.2, False)])
async def test_an_idle_session_engine_is_released(monkeypatch, idle_timeout, kept):
    monkeypatch.setattr(config.settings, "session_idle_timeout", idle_timeout)
    async with Client(server.mcp) as first, Client(server.mcp) as a, Client(server.mcp) as b:
        await first.call_tool("drawing_new", {})  # the first session's engine is never released
        await a.call_tool("drawing_new", {})
        await a.call_tool("entity_create_line", LINE)
        await asyncio.sleep(0.4)
        await b.call_tool("document_list", {})  # B's request sweeps idle engines
        assert (await _documents(a) != []) is kept
        assert await _documents(first) != []


@pytest.mark.asyncio
async def test_ending_an_http_session_releases_its_drawings(monkeypatch):
    """A client's DELETE of its session frees its engine at once, not after the idle timeout."""
    import httpx
    from fastmcp.client.transports import StreamableHttpTransport

    import http_guard

    monkeypatch.setattr(http_guard, "_bound", {"host": "127.0.0.1", "port": 8000})
    app = server.mcp.http_app()

    def factory(**kwargs):
        kwargs.pop("base_url", None)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **kwargs)

    def client():
        url = "http://127.0.0.1:8000/mcp"
        return Client(StreamableHttpTransport(url, httpx_client_factory=factory))

    async with app.router.lifespan_context(app):
        router = server._session_router()
        async with client() as first:
            await first.call_tool("drawing_new", {})
            async with client() as second:
                await second.call_tool("drawing_new", {})
                await second.call_tool("entity_create_line", LINE)
                assert router.session_count() == 2
            assert router.session_count() == 1
            assert await _entities(first) == 0  # the first session keeps its drawing
        assert router.session_count() == 0
        async with client() as third:  # starts on the emptied server engine
            assert await _documents(third) == []
