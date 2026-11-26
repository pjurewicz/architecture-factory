"""preview.py - the read-only /preview page: token check and per-session state."""

from __future__ import annotations

import base64
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
        for route in ("/preview", "/preview/state", "/preview/tools", "/preview/tools.json"):
            assert (await http.get(route)).status_code == 401, route
            assert (await http.get(route, auth=(USER, PASSWORD))).status_code == 200, route


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
        # every open document is viewable, not only the active one, and viewing
        # one never activates it
        await engine.drawing_new()
        for _ in range(2):
            await engine.entity_create_circle(0, 0, 5)
        state = (await http.get("/preview/state", auth=(USER, PASSWORD))).json()
        docs = state["sessions"][0]["documents"]
        first = next(d for d in docs if not d["active"] and d["entity_count"] == 1)
        query = {"session": "server", "document": first["document_id"]}
        got = await http.get("/preview/download", params=query, auth=(USER, PASSWORD))
        assert got.content.count(b"\nLINE\n") == 1 and b"\nCIRCLE\n" not in got.content
        got = await http.get("/preview/render.png", params=query, auth=(USER, PASSWORD))
        assert got.status_code == 200 and got.content.startswith(b"\x89PNG")
        # zoom and pan: the page asks for a window and is told the one drawn
        window = {**query, "region": "10,0,30,5", "max_px": "400"}
        got = await http.get("/preview/render.png", params=window, auth=(USER, PASSWORD))
        x0, y0, x1, y1 = map(float, got.headers["x-render-region"].split(","))
        assert x0 <= 10 and x1 >= 30 and y0 <= 0 and y1 >= 5 and x1 - x0 < 20.1
        assert int.from_bytes(got.content[16:20], "big") == 400  # IHDR width
        bad = {**query, "region": "1,2,x,4"}
        got = await http.get("/preview/render.png", params=bad, auth=(USER, PASSWORD))
        assert got.status_code == 400
        got = await http.get(
            "/preview/download", params={**query, "format": "pdf"}, auth=(USER, PASSWORD)
        )
        assert got.status_code == 200 and got.content.startswith(b"%PDF")
        active = next(d for d in await engine.document_list() if d["active"])
        assert active["entity_count"] == 2
        missing = {"session": "server", "document": "no-such-id"}
        got = await http.get("/preview/render.png", params=missing, auth=(USER, PASSWORD))
        assert got.status_code == 409


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
        recent = busy["calls"]["recent"]
        assert [row["tool"] for row in recent][:1] == ["entity_create_line"]
        assert recent[0]["ok"] and recent[0]["ms"] > 0 and recent[0]["revision"] is not None
        # the arguments as JSON, byte for byte; a text result back and no image
        assert recent[0]["bytes_in"] == len('{"x1": 0, "y1": 0, "x2": 5, "y2": 5}')
        assert recent[0]["bytes_out_text"] > 0 and recent[0]["bytes_out_image"] == 0
        assert busy["calls"]["totals"]["calls"] == 4  # drawing_new + three lines
        # through the search proxy, a call is still logged once, under its real tool
        previous = "search" if server._discovery_transform is not None else "off"
        server._apply_discovery_mode("search")
        try:
            await b.call_tool("call_tool", {"name": "drawing_info", "arguments": {}})
        finally:  # the mode the suite was in, not the configured default
            server._apply_discovery_mode(previous)
        calls = (await preview._state(server))["sessions"]
        busy = next(s for s in calls if s["id"] == busy["id"])
        assert busy["calls"]["totals"]["calls"] == 5
        assert busy["calls"]["recent"][0]["tool"] == "drawing_info"
        # a screenshot's PNG is counted as image bytes: its real decoded size
        shot = await b.call_tool("view_screenshot", {"max_px": 256})
        png = next(block for block in shot.content if block.type == "image")
        calls = (await preview._state(server))["sessions"]
        row = next(s for s in calls if s["id"] == busy["id"])["calls"]["recent"][0]
        assert row["tool"] == "view_screenshot"
        assert row["bytes_out_image"] == len(base64.b64decode(png.data))
        engine = preview._engine(server, busy["id"])
        name, data = await engine.dxf_bytes()
        assert name.endswith(".dxf") and data.count(b"\nLINE\n") == 3


@pytest.mark.asyncio
async def test_tool_usage_reads_the_journal_under_the_tool_that_ran(monkeypatch):
    monkeypatch.setattr(config.settings, "session_isolation", True)
    monkeypatch.setattr(config.settings, "hide_live_only_tools", True)
    async with _served(monkeypatch) as http:

        async def usage(window="all"):
            got = await http.get(
                "/preview/tools.json", params={"range": window}, auth=(USER, PASSWORD)
            )
            assert got.status_code == 200
            return got.json()

        before = {t["tool"]: t for t in (await usage())["tools"]}
        previous = "search" if server._discovery_transform is not None else "off"
        server._apply_discovery_mode("search")
        try:
            async with Client(server.mcp) as a, Client(server.mcp) as b:
                await a.call_tool("call_tool", {"name": "drawing_new", "arguments": {}})
                await b.call_tool("drawing_new", {})
                line = {"x1": 0, "y1": 0, "x2": 5, "y2": 5}
                await b.call_tool("call_tool", {"name": "entity_create_line", "arguments": line})
                for handle in ("FFFFFF", "FFFFFE"):
                    await b.call_tool(
                        "call_tool",
                        {"name": "entity_get", "arguments": {"handle": handle}},
                        raise_on_error=False,
                    )
                await b.call_tool(
                    "call_tool",
                    {"name": "system_run_lisp", "arguments": {"code": "(+ 1 2)"}},
                    raise_on_error=False,
                )
        finally:  # the mode the suite was in, not the configured default
            server._apply_discovery_mode(previous)
        # the sessions have ended; the journal still holds their calls
        state = await usage()
        assert (await usage("24h"))["calls"] <= state["calls"]
        bad = await http.get("/preview/tools.json", params={"range": "1y"}, auth=(USER, PASSWORD))
        assert bad.status_code == 400
    after = {t["tool"]: t for t in state["tools"]}

    def delta(tool, key="calls"):
        return after[tool][key] - before.get(tool, {}).get(key, 0)

    def kinds(tool, kind):
        return after[tool]["error_kinds"].get(kind, 0) - before.get(tool, {}).get(
            "error_kinds", {}
        ).get(kind, 0)

    assert "call_tool" not in after
    assert delta("drawing_new") == 2 and delta("entity_create_line") == 1
    assert delta("entity_get") == 2 and kinds("entity_get", "not_found") == 2
    assert delta("drawing_new", "errors") == 0
    # a call the tool policy turned away is counted too (live-only on this engine)
    assert kinds("system_run_lisp", "policy_refused") == 1
    assert state["calls"] == sum(t["calls"] for t in state["tools"])
    ranked = [t["calls"] for t in state["tools"]]
    assert ranked == sorted(ranked, reverse=True)
