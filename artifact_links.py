"""Download links for files a tool wrote (``GET /artifacts/<token>/<name>``).

A tool called over HTTP that writes a file - ``drawing_export_pdf``,
``batch_plot``, ``drawing_save``, ``drawing_deliver`` - adds ``download_url``
to its result, so the client hands its user a link instead of pushing the
file's bytes through the model. The link names a random token (32 bytes from
``secrets``) registered for ``ARTIFACT_LINK_TTL_S`` seconds (default 3600,
at most 24 h, 0 = no links); the token is the credential, like a signed URL,
because a browser opening the link carries no MCP bearer token. The route
still sits behind the HTTP request guard (Host / Origin).

The file is served from disk as it is when the link is opened, inline with
its media type; a delivery bundle is one zip of the files the delivery wrote,
built when it is downloaded. A stdio call gets no link (there is no server
address to name). The registry is in memory: a restart forgets every link.
"""

from __future__ import annotations

import io
import logging
import secrets
import time
import zipfile
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

import config

log = logging.getLogger(__name__)

#: Media types by suffix; anything else is application/octet-stream.
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".dxf": "application/dxf",
    ".dwg": "application/acad",
    ".dwt": "application/acad",
    ".zip": "application/zip",
    ".png": "image/png",
    ".json": "application/json",
    ".md": "text/markdown; charset=utf-8",
}

#: The most links kept at once; registering past it drops the oldest.
MAX_LINKS = 2048

#: token -> {"name", "files": [Path], "zip": bool, "expires"}
_links: dict[str, dict] = {}


def _base_url() -> str | None:
    """The address the calling client reached this server at, or None off HTTP."""
    try:
        from fastmcp.server.dependencies import get_http_request

        request = get_http_request()
    except Exception:  # stdio, or no request in this context
        return None
    if config.settings.artifact_public_url:
        return config.settings.artifact_public_url
    return str(request.base_url).rstrip("/")


def _register(name: str, files: list[Path], *, as_zip: bool) -> str | None:
    ttl = config.settings.artifact_link_ttl_s
    base = _base_url()
    if ttl <= 0 or base is None:
        return None
    now = time.monotonic()
    for token in [t for t, link in _links.items() if link["expires"] <= now]:
        del _links[token]
    while len(_links) >= MAX_LINKS:
        del _links[next(iter(_links))]
    token = secrets.token_urlsafe(32)
    _links[token] = {"name": name, "files": files, "zip": as_zip, "expires": now + ttl}
    return f"{base}/artifacts/{token}/{name}"


def link_file(path: str | Path | None) -> str | None:
    """A download URL for one written file, or None (off HTTP, links off, no file)."""
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        return None
    return _register(file.name, [file.resolve()], as_zip=False)


def link_bundle(name: str, paths: list[str | Path]) -> str | None:
    """A download URL for a zip of ``paths`` (the files that exist), or None."""
    files = [Path(p).resolve() for p in paths if Path(p).is_file()]
    if not files:
        return None
    return _register(name, files, as_zip=True)


def add_link(result, path: str | Path | None = None):
    """``result`` with ``download_url`` for the file it wrote, when there is one."""
    if not isinstance(result, dict) or result.get("ok") is False:
        return result
    url = link_file(path or result.get("path"))
    if url:
        result["download_url"] = url
        result["download_expires_in_s"] = int(config.settings.artifact_link_ttl_s)
    return result


def _zip(files: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if file.is_file():
                archive.write(file, arcname=file.name)
    return buffer.getvalue()


def register(server) -> None:
    """Add ``GET /artifacts/{token}/{filename}`` to ``server.mcp``."""
    import anyio

    @server.mcp.custom_route("/artifacts/{token}/{filename}", methods=["GET"])
    async def artifact_download(request: Request) -> Response:
        token = request.path_params["token"]
        link = _links.get(token)
        if (
            link is None
            or link["expires"] <= time.monotonic()
            or request.path_params["filename"] != link["name"]
        ):
            return JSONResponse({"error": "not_found"}, status_code=404)
        name = link["name"]
        headers = {"cache-control": "private, no-store", "x-content-type-options": "nosniff"}
        media_type = MEDIA_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")
        if link["zip"]:
            data = await anyio.to_thread.run_sync(_zip, link["files"])
            headers["content-disposition"] = f'attachment; filename="{name}"'
            return Response(data, media_type=media_type, headers=headers)
        file = link["files"][0]
        if not file.is_file():
            return JSONResponse({"error": "gone"}, status_code=410)
        return FileResponse(
            file,
            media_type=media_type,
            filename=name,
            content_disposition_type="inline",
            headers=headers,
        )
