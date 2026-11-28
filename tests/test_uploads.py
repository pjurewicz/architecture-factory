"""uploads.py - POST /files puts a client's file on the server and answers its path.

Driven in-process through the guarded app ``mcp.http_app()`` builds (httpx's
ASGITransport), with both tokens configured: the returned path is then passed,
unchanged, to ``view_screenshot(overlay_image=)`` and ``image_attach`` over the
same HTTP MCP endpoint.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

import config
import http_guard
import server
import uploads

Image = pytest.importorskip("PIL.Image", reason="the scan is written with Pillow")

BASE = "http://127.0.0.1:8000"
RW, RO = "rw-upload-token-0123456789", "ro-upload-token-0123456789"
ALIGN = [[0, 0, 0, 0], [100, 0, 50, 0]]


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


@contextlib.asynccontextmanager
async def _served(monkeypatch, tmp_path, token=RW):
    auth = http_guard.build_auth(RW, RO)
    monkeypatch.setattr(server.mcp, "auth", auth)
    monkeypatch.setattr(server, "_auth", auth)
    monkeypatch.setattr(config.settings, "allowed_paths", [tmp_path.resolve()], raising=False)
    monkeypatch.setattr(config.settings, "upload_dir", "", raising=False)
    monkeypatch.setattr(config.settings, "upload_max_bytes", 64 * 1024, raising=False)
    monkeypatch.setattr(http_guard, "_bound", {"host": "127.0.0.1", "port": 8000})
    app = server.mcp.http_app()
    async with app.router.lifespan_context(app):

        def factory(**kwargs):
            kwargs.pop("transport", None)
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **kwargs)

        transport = StreamableHttpTransport(
            f"{BASE}/mcp",
            headers={"authorization": f"Bearer {token}"},
            httpx_client_factory=factory,
        )
        async with Client(transport) as mcp, factory(base_url=BASE) as http:
            yield mcp, http


def _auth(token: str = RW) -> dict:
    return {"authorization": f"Bearer {token}"}


async def _post(http, name, body, token=RW, **headers):
    return await http.post(
        "/files", params={"name": name}, content=body, headers={**_auth(token), **headers}
    )


@pytest.mark.asyncio
async def test_an_uploaded_scan_works_in_overlay_image_and_image_attach(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    data = _png()
    async with _served(monkeypatch, tmp_path) as (mcp, http):
        response = await _post(http, "my scan.png", data)
        assert response.status_code == 201, response.text
        stored = response.json()
        path = stored["path"]
        assert stored["bytes"] == len(data) and stored["format"] == "png"
        assert os.path.dirname(path) == str(tmp_path.resolve() / "uploads")
        assert os.path.basename(path).startswith("my_scan-") and path.endswith(".png")
        with open(path, "rb") as f:
            assert f.read() == data
        import hashlib

        assert stored["sha256"] == hashlib.sha256(data).hexdigest()

        await mcp.call_tool("call_tool", {"name": "drawing_new", "arguments": {}})
        shot = await mcp.call_tool(
            "call_tool",
            {
                "name": "view_screenshot",
                "arguments": {
                    "region": [-10, -40, 60, 10],
                    "max_px": 128,
                    "overlay_image": path,
                    "overlay_align": ALIGN,
                },
            },
        )
        report = next(
            json.loads(block.text) for block in shot.content if getattr(block, "text", None)
        )
        assert report["overlay"]["path"] == path and report["overlay"]["pixels"] == [100, 60]

        attached = await mcp.call_tool(
            "call_tool", {"name": "image_attach", "arguments": {"path": path}}
        )
        assert not attached.is_error
        payload = attached.structured_content.get("result", attached.structured_content)
        assert payload.get("handle")


@pytest.mark.asyncio
async def test_every_bad_upload_is_refused_and_stores_nothing(monkeypatch, tmp_path):
    data = _png()
    folder = tmp_path / "uploads"
    async with _served(monkeypatch, tmp_path) as (_mcp, http):
        # no token, a wrong token, a read-only token
        assert (await http.post("/files?name=a.png", content=data)).status_code == 401
        assert (await _post(http, "a.png", data, token="nope")).status_code == 401
        ro = await _post(http, "a.png", data, token=RO)
        assert ro.status_code == 403 and ro.json()["kind"] == "policy_refused"
        # oversized: refused on Content-Length before the body is read
        big = await _post(http, "a.png", data + b"\0" * (64 * 1024))
        assert big.status_code == 413 and big.json()["error"] == "body_too_large"

        # a chunked body with no length
        async def chunks():
            yield data

        chunked = await http.post("/files?name=a.png", content=chunks(), headers=_auth())
        assert chunked.status_code == 411
        # a disallowed extension, mismatched content
        txt = await _post(http, "notes.txt", b"hello")
        assert txt.status_code == 415 and txt.json()["reason"] == "extension_not_allowed"
        fake = await _post(http, "scan.png", b"\xff\xd8\xff\xe0 a jpeg really")
        assert fake.status_code == 422 and fake.json()["reason"] == "content_mismatch"
        assert fake.json()["found"] == "jpeg"
        notdxf = await _post(http, "part.dxf", b"<html>not a drawing</html>")
        assert notdxf.status_code == 422
        # a path in the name
        for name in ("../../etc/evil.png", "..\\evil.png", "sub/evil.png", "/abs/evil.png"):
            refused = await _post(http, name, data)
            assert refused.status_code == 400 and refused.json()["reason"] == "traversal", name
        # loadable names, even with content that sniffs as an image
        for name in ("acaddoc.lsp", "acad.pgp", "run.scr", "evil.lsp. ", "x.png:acad.lsp"):
            refused = await _post(http, name, data)
            assert refused.status_code == 400, name
            assert refused.json()["kind"] == "path_refused", name
    assert not folder.exists() or [p.name for p in folder.iterdir()] == []


@pytest.mark.asyncio
async def test_drawings_upload_and_open(monkeypatch, tmp_path):
    import ezdxf

    doc = ezdxf.new()
    doc.modelspace().add_line((0, 0), (10, 5))
    stream = io.StringIO()
    doc.write(stream)
    async with _served(monkeypatch, tmp_path) as (mcp, http):
        response = await _post(http, "client.dxf", stream.getvalue().encode())
        assert response.status_code == 201, response.text
        opened = await mcp.call_tool(
            "call_tool", {"name": "drawing_open", "arguments": {"path": response.json()["path"]}}
        )
        assert not opened.is_error
        dwg = await _post(http, "client.dwg", b"AC1032" + b"\0" * 200)
        assert dwg.status_code == 201 and dwg.json()["format"] == "dwg"


def test_the_sweep_deletes_only_old_uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "upload_retain_s", 3600.0, raising=False)
    old = tmp_path / "scan-0123456789abcdef.png"
    fresh = tmp_path / "scan-fedcba9876543210.png"
    foreign = tmp_path / "client.png"
    for file in (old, fresh, foreign):
        file.write_bytes(b"x")
    past = time.time() - 7200
    os.utime(old, (past, past))
    os.utime(foreign, (past, past))
    assert uploads.sweep(tmp_path) == 1
    assert not old.exists() and fresh.exists() and foreign.exists()


def test_sniff_reads_each_format():
    assert uploads.sniff(b"\x89PNG\r\n\x1a\n...") == "png"
    assert uploads.sniff(b"II*\x00....") == "tiff"
    assert uploads.sniff(b"AC1027\0\0") == "dwg"
    assert uploads.sniff(b"999\nmade by x\n  0\nSECTION\n  2\nHEADER\n") == "dxf"
    assert uploads.sniff(b"  0\r\nSECTION\r\n") == "dxf"
    assert uploads.sniff(b"  0\nLINE\n") is None
    assert uploads.sniff(b"MZ\x90\x00") is None


def _age(path) -> None:
    past = time.time() - 2 * 86400
    os.utime(path, (past, past))


@pytest.mark.asyncio
async def test_work_on_an_uploaded_drawing_never_lives_in_the_expiring_folder(
    monkeypatch, tmp_path
):
    """(a) An in-place save or a close-and-save onto an upload is refused; saved elsewhere, it survives."""
    import ezdxf

    monkeypatch.setattr(config.settings, "allowed_paths", [tmp_path.resolve()], raising=False)
    monkeypatch.setattr(config.settings, "upload_dir", "", raising=False)
    folder = uploads.upload_dir()
    folder.mkdir(parents=True)
    upload = folder / "client-0123456789abcdef.dxf"
    ezdxf.new().saveas(upload)
    work = tmp_path / "work.dxf"
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_open", {"path": str(upload)})
        await client.call_tool("entity_create_line", {"x1": 0, "y1": 0, "x2": 9, "y2": 9})
        for tool, args in (
            ("drawing_save", {}),
            ("document_close", {"save": True}),
            ("drawing_save", {"path": str(folder / "mine.dxf")}),
        ):
            refused = await client.call_tool(tool, args, raise_on_error=False)
            assert refused.is_error, tool
            body = refused.structured_content
            assert body["kind"] == "path_refused" and body["reason"] == "upload_folder", tool
        assert "drawing_save(path=" in refused.content[0].text
        saved = await client.call_tool("drawing_save", {"path": str(work)})
        assert not saved.is_error
    assert len(ezdxf.readfile(upload).modelspace()) == 0  # the upload was never written
    _age(upload)
    _age(work)
    assert uploads.sweep() == 1 and not upload.exists()
    assert len(ezdxf.readfile(work).modelspace()) == 1


@pytest.mark.asyncio
async def test_an_attached_upload_survives_the_sweep(monkeypatch, tmp_path):
    """(b) image_attach references a kept copy, so the drawing keeps its image after the sweep."""
    import ezdxf

    monkeypatch.setattr(config.settings, "allowed_paths", [tmp_path.resolve()], raising=False)
    monkeypatch.setattr(config.settings, "upload_dir", "", raising=False)
    folder = uploads.upload_dir()
    folder.mkdir(parents=True)
    upload = folder / "logo-0123456789abcdef.png"
    upload.write_bytes(_png())
    async with Client(server.mcp) as client:
        await client.call_tool("drawing_new", {})
        attached = await client.call_tool("image_attach", {"path": str(upload), "plot": True})
        assert attached.structured_content["kept_from_upload"] == str(upload)
        await client.call_tool("drawing_save", {"path": str(tmp_path / "sheet.dxf")})
    _age(upload)
    assert uploads.sweep() == 1 and not upload.exists()
    defs = list(ezdxf.readfile(tmp_path / "sheet.dxf").objects.query("IMAGEDEF"))
    assert len(defs) == 1
    kept = defs[0].dxf.filename
    assert kept == str(folder / "kept" / upload.name)
    assert open(kept, "rb").read() == _png()


async def _link(mcp, name: str) -> str:
    result = await mcp.call_tool("file_upload_url", {"name": name})
    payload = result.structured_content
    keys = {"ok", "upload_url", "method", "expires_in_s", "max_bytes"}
    # round 4 (B3): a raster's answer points at scan_read, where measuring is
    raster = name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    assert set(payload) == (keys | {"measure"} if raster else keys)
    assert not raster or "scan_read" in payload["measure"]
    assert RW not in json.dumps(payload) and RO not in json.dumps(payload), "no bearer token"
    return payload["upload_url"].removeprefix(BASE)


@pytest.mark.asyncio
async def test_a_one_time_link_uploads_once_and_its_path_opens(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib", reason="rendering needs the [pdf] extra")
    data = _png()
    async with _served(monkeypatch, tmp_path) as (mcp, http):
        url = await _link(mcp, "my scan.png")
        assert url.startswith("/files/") and "name=" not in url
        stored = await http.post(url, content=data)  # no Authorization header
        assert stored.status_code == 201, stored.text
        path = stored.json()["path"]
        assert path.startswith(str(tmp_path / "uploads")) and path.endswith(".png")
        again = await http.post(url, content=data)
        assert again.status_code == 404, "a link uploads once"
        await mcp.call_tool("drawing_new", {})
        shot = await mcp.call_tool(
            "view_screenshot", {"overlay_image": path, "overlay_align": ALIGN}
        )
        assert shot.structured_content["overlay"]["pixels"] == [100, 60]
        # the link opens nothing else: no GET, no other path under it
        url2 = await _link(mcp, "b.png")
        assert (await http.get(url2)).status_code in (404, 405)
        assert (await http.post(url2 + "/x", content=data)).status_code == 404
        assert (await http.post(url2, content=data)).status_code == 201


@pytest.mark.asyncio
async def test_a_link_skips_none_of_the_upload_checks_and_a_refusal_spends_it(
    monkeypatch, tmp_path
):
    data = _png()
    folder = tmp_path / "uploads"
    async with _served(monkeypatch, tmp_path) as (mcp, http):
        # the name rules apply when the link is asked for
        for name in ("notes.txt", "acaddoc.lsp", "acad.pgp", "../evil.png", "x.png:acad.lsp"):
            refused = await mcp.call_tool("file_upload_url", {"name": name}, raise_on_error=False)
            assert refused.is_error and refused.structured_content["kind"] == "path_refused", name
        # size: refused before a byte is read, and the link is spent
        url = await _link(mcp, "a.png")
        big = await http.post(url, content=data + b"\0" * (64 * 1024))
        assert big.status_code == 413
        assert (await http.post(url, content=data)).status_code == 404, "spent by the 413"

        # no length (chunked)
        async def chunks():
            yield data

        url = await _link(mcp, "a.png")
        assert (await http.post(url, content=chunks())).status_code == 411
        assert (await http.post(url, content=data)).status_code == 404, "spent by the 411"
        # content: the extension the link was issued for decides
        url = await _link(mcp, "scan.png")
        fake = await http.post(url, content=b"\xff\xd8\xff\xe0 a jpeg really")
        assert fake.status_code == 422 and fake.json()["reason"] == "content_mismatch"
        assert (await http.post(url, content=data)).status_code == 404, "spent by the 422"
        # the client cannot rename what the link stores
        url = await _link(mcp, "part.dxf")
        renamed = await http.post(url, params={"name": "evil.png"}, content=data)
        assert renamed.status_code == 422 and renamed.json()["expected"] == "dxf"
        # an unknown token
        assert (await http.post("/files/" + "A" * 43, content=data)).status_code == 404
        # the write policy is read again at upload time
        url = await _link(mcp, "a.png")
        monkeypatch.setattr(config.settings, "read_only", True, raising=False)
        ro = await http.post(url, content=data)
        assert ro.status_code == 403 and ro.json()["kind"] == "policy_refused"
    assert not folder.exists() or [p.name for p in folder.iterdir()] == []


@pytest.mark.asyncio
async def test_a_read_only_token_gets_no_link(monkeypatch, tmp_path):
    async with _served(monkeypatch, tmp_path, token=RO) as (mcp, _http):
        refused = await mcp.call_tool("file_upload_url", {"name": "a.png"}, raise_on_error=False)
    assert refused.is_error and refused.structured_content["kind"] == "policy_refused"
    assert uploads._tickets == {} or all(t["name"] != "a.png" for t in uploads._tickets.values())


def test_the_access_log_masks_the_token():
    import logging

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4", "POST", "/files/s3cr3t-T0ken_x", "1.1", 201),
        None,
    )
    uploads._MaskTokens().filter(record)
    assert "s3cr3t" not in record.getMessage() and "/files/***" in record.getMessage()
