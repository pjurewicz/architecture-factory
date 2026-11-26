"""preview.py - the read-only /preview page: token check and per-session state."""

from __future__ import annotations

import contextlib

import httpx
import pytest
from fastmcp import Client

import config
import http_guard
import server

BASE = "http://127.0.0.1:8000"
USER, PASSWORD = "architect", "preview-pass-0123456789"


@contextlib.asynccontextmanager
async def _served(monkeypatch, password=PASSWORD):
    monkeypatch.setattr(config.settings, "mcp_auth_token", "mcp-token-0123456789")
    monkeypatch.setattr(config.settings, "preview_user", USER)
    monkeypatch.setattr(config.settings, "preview_password", password)
    monkeypatch.setattr(http_guard, "_bound", {"host": "127.0.0.1", "port": 8000})
    app = server.mcp.http_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE) as client:
            yield client


@pytest.mark.asyncio
async def test_the_page_needs_its_basic_credentials(monkeypatch):
    async with _served(monkeypatch) as http:
        refused = await http.get("/preview")
        assert refused.status_code == 401 and refused.headers["www-authenticate"].startswith(
            "Basic"
        )
        for auth in ((USER, "wrong"), ("someone", PASSWORD)):
            assert (await http.get("/preview/state", auth=auth)).status_code == 401
        mcp_token = {"authorization": "Bearer mcp-token-0123456789"}
        assert (await http.get("/preview/state", headers=mcp_token)).status_code == 401
        assert (await http.get("/preview", auth=(USER, PASSWORD))).status_code == 200
        assert (await http.get("/preview/state", auth=(USER, PASSWORD))).status_code == 200


@pytest.mark.asyncio
async def test_the_active_drawing_downloads_as_pdf_dxf_and_dwg(monkeypatch):
    import preview
    from backends.ezdxf_backend import _odafc_available

    async with _served(monkeypatch) as http:
        engine = preview._engine(server, "server")
        await engine.drawing_new()
        await engine.entity_create_line(0, 0, 50, 20)
        formats = {"pdf": b"%PDF", "dxf": b"  0\nSECTION"}
        if _odafc_available():
            formats["dwg"] = b"AC10"
        for fmt, magic in formats.items():
            got = await http.get(
                "/preview/download",
                params={"session": "server", "format": fmt},
                auth=(USER, PASSWORD),
            )
            assert got.status_code == 200 and got.content.startswith(magic), fmt
            assert got.headers["content-disposition"].endswith(f'.{fmt}"')


@pytest.mark.asyncio
async def test_the_page_is_off_without_a_password(monkeypatch):
    async with _served(monkeypatch, password="") as http:
        assert (await http.get("/preview", auth=(USER, ""))).status_code == 404


@pytest.mark.asyncio
async def test_state_shows_each_session_drawing_and_renders_it(monkeypatch):
    monkeypatch.setattr(config.settings, "session_isolation", True)
    async with Client(server.mcp) as a, Client(server.mcp) as b:
        await a.call_tool("drawing_new", {})
        await b.call_tool("drawing_new", {})
        for _ in range(3):
            await b.call_tool("entity_create_line", {"x1": 0, "y1": 0, "x2": 5, "y2": 5})
        import preview

        sessions = (await preview._state(server))["sessions"]
        counts = sorted(s["documents"][0]["entity_count"] for s in sessions if s["documents"])
        assert counts == [0, 3]
        busy = next(s for s in sessions if s["documents"] and s["documents"][0]["entity_count"])
        assert [row["tool"] for row in busy["history"]][:1] == ["entity_create_line"]
        engine = preview._engine(server, busy["id"])
        name, data = await engine.dxf_bytes()
        assert name.endswith(".dxf") and data.count(b"\nLINE\n") == 3
