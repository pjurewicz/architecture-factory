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
