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


# ── round 3 (A1): recovering an untitled drawing after its session ended ────


async def _lost_untitled_drawing() -> tuple[str, str]:
    """Session A draws three lines on an untitled drawing, checkpoints it and ends."""
    async with Client(server.mcp) as a:
        made = _data(await a.call_tool("drawing_new", {}))
        for _ in range(3):
            await a.call_tool("entity_create_line", LINE)
        await a.call_tool("checkpoint_create", {"label": "before the break"})
        return made["document_id"], made["recovery_token"]


async def _recover(client, document_id: str, token: str):
    return await client.call_tool(
        "drawing_recover",
        {"document_id": document_id, "recovery_token": token},
        raise_on_error=False,
    )


@pytest.mark.asyncio
async def test_only_the_token_recovers_an_untitled_drawing_into_the_callers_session():
    async with Client(server.mcp) as first:  # the default engine is someone else's
        await first.call_tool("drawing_new", {})
        document_id, token = await _lost_untitled_drawing()
        async with Client(server.mcp) as b:
            refusals = [
                await _recover(b, document_id, token[:-1] + ("A" if token[-1] != "A" else "B")),
                await _recover(b, "not-a-document", token),
                await _recover(b, document_id, "nolineage.secret"),
            ]
            texts = {r.content[0].text for r in refusals}
            assert all(r.is_error for r in refusals) and len(texts) == 1, texts
            assert all(r.structured_content["kind"] == "recovery_refused" for r in refusals)
            assert await _documents(b) == []

            recovered = _data(await _recover(b, document_id, token))
            assert recovered["ok"] is True and await _entities(b) == 3
            assert recovered["recovered"]["label"] == "before the break"


@pytest.mark.asyncio
async def test_a_token_expires_with_its_checkpoints(monkeypatch):
    from backends.checkpoints import open_store

    async with Client(server.mcp) as first:
        await first.call_tool("drawing_new", {})
        document_id, token = await _lost_untitled_drawing()
        store = open_store()
        lineage_id = token.split(".", 1)[0]
        index = store.load(lineage_id)
        for row in index["checkpoints"]:
            store._evict_row(lineage_id, row, "test")
        (store.lineage_dir(lineage_id) / "index.json").write_text(json.dumps(index))
        async with Client(server.mcp) as b:
            refused = await _recover(b, document_id, token)
            assert refused.is_error and refused.structured_content["kind"] == "recovery_refused"


@pytest.mark.asyncio
async def test_the_journal_keeps_no_recovery_token():
    import os
    import sqlite3
    from pathlib import Path

    async with Client(server.mcp) as first:
        await first.call_tool("drawing_new", {})
        document_id, token = await _lost_untitled_drawing()
        async with Client(server.mcp) as b:
            await _recover(b, document_id, token)
    secret = token.split(".", 1)[1]
    db = Path(os.environ["ACADMCP_STATE_DIR"]) / "journal.sqlite3"
    dump = "\n".join(sqlite3.connect(db).iterdump())
    assert secret not in dump and "recovery_token" in dump
    folder = Path(os.environ["ACADMCP_STATE_DIR"]) / "docs" / token.split(".", 1)[0]
    assert all(secret not in p.read_text(errors="ignore") for p in folder.glob("*.json"))
