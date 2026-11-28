"""artifact_links.py - download_url for files written during an HTTP call."""

from __future__ import annotations

import contextlib
import io
import zipfile

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

import artifact_links
import config
import http_guard
import server

BASE = "http://127.0.0.1:8000"


@contextlib.asynccontextmanager
async def _served(monkeypatch, tmp_path, ttl=3600.0):
    monkeypatch.setattr(config.settings, "artifact_link_ttl_s", ttl)
    monkeypatch.setattr(config.settings, "allowed_paths", [str(tmp_path)], raising=False)
    monkeypatch.setattr(http_guard, "_bound", {"host": "127.0.0.1", "port": 8000})
    app = server.mcp.http_app()
    async with app.router.lifespan_context(app):

        def factory(**kwargs):
            kwargs.pop("transport", None)
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **kwargs)

        transport = StreamableHttpTransport(f"{BASE}/mcp", httpx_client_factory=factory)
        async with Client(transport) as mcp, factory(base_url=BASE) as http:
            yield mcp, http


async def _call(mcp, tool, args):
    result = await mcp.call_tool("call_tool", {"name": tool, "arguments": args})
    return result.structured_content.get("result", result.structured_content)


@pytest.mark.asyncio
async def test_a_written_pdf_and_dxf_download_through_their_links(monkeypatch, tmp_path):
    async with _served(monkeypatch, tmp_path) as (mcp, http):
        await _call(mcp, "drawing_new", {})
        await _call(mcp, "entity_create_line", {"x1": 0, "y1": 0, "x2": 50, "y2": 20})
        pdf = await _call(mcp, "drawing_export_pdf", {"path": str(tmp_path / "QA_TEST_R3.pdf")})
        url = pdf["download_url"]
        assert url.startswith(f"{BASE}/artifacts/") and url.endswith("/QA_TEST_R3.pdf")
        got = await http.get(url)
        assert got.status_code == 200 and got.content == (tmp_path / "QA_TEST_R3.pdf").read_bytes()
        assert got.headers["content-type"] == "application/pdf"
        assert got.headers["content-disposition"] == 'inline; filename="QA_TEST_R3.pdf"'
        # the token is the credential: another name, or a guessed token, is 404
        token = url.split("/")[-2]
        assert (await http.get(f"/artifacts/{token}/other.pdf")).status_code == 404
        assert (await http.get("/artifacts/guess/QA_TEST_R3.pdf")).status_code == 404

        saved = await _call(mcp, "drawing_save", {"path": str(tmp_path / "part.dxf")})
        got = await http.get(saved["download_url"])
        assert got.headers["content-type"] == "application/dxf"
        assert got.content == (tmp_path / "part.dxf").read_bytes()

        plotted = await _call(
            mcp, "batch_plot", {"output_dir": str(tmp_path / "plots"), "layouts": ["Model"]}
        )
        sheet = plotted["sheets"][0]
        got = await http.get(sheet["download_url"])
        assert got.content.startswith(b"%PDF") and sheet["download_url"].endswith(".pdf")

        # round 6: a screenshot saved as a file downloads too
        shot = await mcp.call_tool(
            "call_tool",
            {"name": "view_screenshot", "arguments": {"save_path": str(tmp_path / "v/a.png")}},
        )
        shot = shot.structured_content.get("result", shot.structured_content)
        got = await http.get(shot["download_url"])
        assert got.headers["content-type"] == "image/png"
        assert got.content == (tmp_path / "v/a.png").read_bytes()
        assert got.content.startswith(b"\x89PNG") and shot["path"] == str(tmp_path / "v/a.png")

        # an expired link is gone
        artifact_links._links[token]["expires"] = 0
        assert (await http.get(url)).status_code == 404


@pytest.mark.asyncio
async def test_a_delivery_downloads_as_one_zip(monkeypatch, tmp_path):
    async with _served(monkeypatch, tmp_path) as (mcp, http):
        await _call(mcp, "drawing_new", {})
        await _call(mcp, "entity_create_line", {"x1": 0, "y1": 0, "x2": 50, "y2": 20})
        out = tmp_path / "bundle"
        delivered = await _call(
            mcp,
            "drawing_deliver",
            {
                "output_dir": str(out),
                "formats": ["dxf", "pdf"],
                "min_score": 0,
                "strict_critique": False,
            },
        )
        assert delivered["download_url"].endswith("/bundle.zip")
        got = await http.get(delivered["download_url"])
        assert got.headers["content-type"] == "application/zip"
        names = set(zipfile.ZipFile(io.BytesIO(got.content)).namelist())
        created = {a["filename"] for a in delivered["artifacts"] if a["status"] == "created"}
        assert created | {"manifest.json", "validation.json"} <= names
        for record in delivered["artifacts"]:
            if record["status"] == "created":
                one = await http.get(record["download_url"])
                assert one.content == (out / record["filename"]).read_bytes()


@pytest.mark.asyncio
async def test_no_link_over_stdio_or_with_links_off(monkeypatch, tmp_path):
    async with Client(server.mcp) as local:
        await local.call_tool("drawing_new", {})
        result = await local.call_tool("drawing_save", {"path": str(tmp_path / "a.dxf")})
        assert "download_url" not in result.structured_content
    async with _served(monkeypatch, tmp_path, ttl=0) as (mcp, _http):
        await _call(mcp, "drawing_new", {})
        saved = await _call(mcp, "drawing_save", {"path": str(tmp_path / "b.dxf")})
        assert saved["ok"] and "download_url" not in saved
