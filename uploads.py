"""Uploading a client's file to the server (``POST /files?name=<file name>``).

MCP has no way to send a file from the client to the server, and every tool
that reads one - ``image_attach``, ``view_screenshot(overlay_image=)``,
``drawing_open``, ``xref_attach``, ``drawing_new(template=)`` - takes a path on
the server. Over the HTTP transport a client (or its user, with curl) sends the
file's bytes as the request body and gets back the server path to pass to
those tools::

    curl -H "Authorization: Bearer $MCP_AUTH_TOKEN" --data-binary @scan.png \\
         "https://cad.example.com/files?name=scan.png"
    -> 201 {"ok": true, "path": ".../uploads/scan-3f9c0a1b2c4d5e6f.png",
            "sha256": "...", "bytes": 48213, "format": "png", "expires_in_s": 86400}

A stdio server runs on the client's machine and opens its files directly; it
has no endpoint.

Rules (each refusal names its reason):

* **The MCP bearer token.** The same ``MCP_AUTH_TOKEN`` the MCP endpoint
  checks, through the same ``ConstantTimeTokenVerifier``; no token is 401. An
  upload writes a file, so a token without ``mcp:write``
  (``MCP_AUTH_TOKEN_READONLY``) is 403 ``policy_refused``, and so is a server
  whose policy forbids writes (``READ_ONLY``, ``MAX_TOOL_COST`` below
  ``mutate``). A server with no token at all has no endpoint (404): its files
  are local to whoever can reach it.
* **The server names the file.** The client's ``name`` gives only the
  extension and a readable stem; a name holding a path separator, ``..``, a
  ``:`` or a control character is refused, a loadable name (``*.lsp``,
  ``acad.pgp``, ...: ``security.WRITE_REFUSED_SUFFIXES``) is refused, and the
  extension must be one of :data:`security.KINDS_UPLOAD`. The stored name is
  ``<stem>-<16 random hex><ext>`` in :func:`upload_dir`, written through
  ``security.validate_path(allow_write=True, kinds=KINDS_UPLOAD)``, so it is
  inside ``ALLOWED_PATHS`` and every tool accepts it unchanged.
* **The content must be what the extension says**: PNG / JPEG / TIFF by their
  magic bytes, a DWG by its ``AC10xx`` version string, a DXF by its first
  ``0 / SECTION`` pair (or the binary DXF sentinel), a PDF (for ``pdf_read``)
  by ``%PDF-`` at byte 0, and a ``.json`` (a ``cad_batch(steps_path=)`` /
  ``entity_batch_create(entities_path=)`` list) by parsing the whole file
  within ``engineering.batch_file``'s ``MAX_BYTES`` (16 MiB) and ``MAX_DEPTH``.
  Anything else is 422 ``invalid_args``, nothing kept.
* **Bounded.** The body must declare its ``Content-Length`` (411 without), at
  most ``UPLOAD_MAX_BYTES`` (256 MiB, clamped to 2 GiB; ``0`` turns uploads
  off); ``http_guard.HttpGuard`` applies that limit to this path instead of
  ``MAX_HTTP_BODY_BYTES`` and refuses a larger one 413 before a byte is read.
  The body is streamed to disk, never held in memory.
* **Kept for ``UPLOAD_RETAIN_S``** (24 h): every upload first deletes the
  uploads older than that (only names this module wrote, only directly in
  the folder). No user work can live there: every other write into the
  folder is refused (``refuse_write``: ``validate_path(allow_write=True)``
  and every DXF save, the in-place one included), and ``image_attach`` /
  ``xref_attach`` reference a copy in ``kept/`` (:func:`keep`), never the
  upload itself.

**One-time links** (``file_upload_url``): an agent whose MCP client does not
hand it the bearer token asks the tool, under its own authenticated MCP call,
for ``POST /files/<token>``. The token (32 random bytes) stands in for the
``Authorization`` header for exactly one request: it is bound to the name the
tool was given (the client cannot change it), lives ``UPLOAD_LINK_TTL_S``
(600 s), and is spent by the first request that names it, a refused one
included. Every other rule above still applies to it - the write policy is
read again at upload time - and it opens no other route. The access log
shows ``/files/***`` instead of the token.

Refusals the HTTP layer makes (404, 401, 411, 413) answer the guard's
``{ok: false, error, message}``; the others are the tool error envelope's
fields (``engineering/errors.py``: ``kind``, ``category``, ``retryable``,
``hint``, ``message``, ``reason``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response

import config
import http_guard
import security
from policy import PolicyRefusedError

log = logging.getLogger("autocad_mcp.uploads")

#: The route; ``http_guard.HttpGuard`` gives it ``UPLOAD_MAX_BYTES``.
UPLOAD_PATH = "/files"

#: Extension -> the format its content must be.
FORMATS: dict[str, str] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".dxf": "dxf",
    ".dwg": "dwg",
    ".pdf": "pdf",
    ".json": "json",
}

#: A name this module wrote (the only files the retention sweep deletes).
_STORED = re.compile(r"^[A-Za-z0-9_-]{1,40}-[0-9a-f]{16}(\.[a-z]{3,4})(\.part)?$")
_STEM_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_DWG = re.compile(rb"^AC(\d{4}|\d\.\d{1,2})")
_BINARY_DXF = b"AutoCAD Binary DXF\r\n\x1a\x00"
_SNIFF_BYTES = 4096
#: The subfolder an upload a drawing references is copied into; the sweep
#: reads only the files directly in the uploads folder, so it is never swept.
KEPT_DIR = "kept"
#: A ``.part`` file older than this is a dead upload's leftover.
_PART_TTL_S = 3600.0
#: How long a one-time upload link lives, and how many may be open at once.
UPLOAD_LINK_TTL_S = 600
MAX_TICKETS = 256
#: token -> {"name", "expires"}; in memory, so a restart spends every link.
_tickets: dict[str, dict] = {}


def enabled() -> bool:
    return int(getattr(config.settings, "upload_max_bytes", 0) or 0) > 0


def upload_dir() -> Path:
    """``UPLOAD_DIR``, else ``<first ALLOWED_PATHS entry>/uploads``, else a temp folder."""
    raw = getattr(config.settings, "upload_dir", "")
    if raw:
        return Path(raw).expanduser().resolve()
    allowed = list(config.settings.allowed_paths or ())
    if allowed:
        return Path(allowed[0]).resolve() / "uploads"
    return Path(tempfile.gettempdir()).resolve() / "autocad-mcp-pro-uploads"


def in_upload_folder(path) -> bool:
    """True when ``path`` is a file directly in :func:`upload_dir` - what the sweep deletes."""
    try:
        return Path(path).resolve().parent == upload_dir()
    except (OSError, ValueError, TypeError):
        return False


def refuse_write(path, op: str = "a write") -> None:
    """Refuse ``op`` onto a file the sweep would delete (``path_refused``, reason ``upload_folder``).

    Only ``upload()`` writes there; a drawing opened from an upload is saved
    elsewhere, so no user work ever lives in the expiring folder.
    """
    if in_upload_folder(path):
        raise security.PathRefusedError(
            f"{op}: '{Path(path).name}' is in the uploads folder, whose files are deleted "
            f"after UPLOAD_RETAIN_S; save your work elsewhere with "
            f"drawing_save(path='<folder>/<name>.dxf').",
            reason="upload_folder",
        )


def keep(path) -> tuple[Path, bool]:
    """``path``, or - for an upload - its copy in ``<upload_dir>/kept`` (never swept).

    A drawing that references a file (an IMAGE, an xref) must not lose it when
    the upload expires, so the reference names the kept copy. The second value
    is True when ``path`` was an upload.
    """
    source = Path(path).resolve()
    if not in_upload_folder(source):
        return source, False
    target = security.validate_path(
        str(source.parent / KEPT_DIR / source.name),
        allow_write=True,
        kinds=security.KINDS_UPLOAD,
        create_parents=True,
    )
    if not target.exists():
        part = target.with_name(target.name + ".part")
        shutil.copyfile(source, part)
        os.replace(part, target)
    return target, True


def sweep(folder: Path | None = None, now: float | None = None) -> int:
    """Delete this module's uploads older than ``UPLOAD_RETAIN_S``; the count deleted."""
    folder = folder or upload_dir()
    if not folder.is_dir():
        return 0
    now = time.time() if now is None else now
    retain = float(config.settings.upload_retain_s)
    removed = 0
    for entry in folder.iterdir():
        match = _STORED.match(entry.name)
        if match is None or not entry.is_file():
            continue
        ttl = _PART_TTL_S if match.group(2) else retain
        try:
            if now - entry.stat().st_mtime > ttl:
                entry.unlink()
                removed += 1
        except OSError:  # deleted meanwhile, or not ours to delete
            continue
    return removed


def sniff(head: bytes) -> str | None:
    """The format ``head`` (a file's first bytes) starts like, or None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if _DWG.match(head):
        return "dwg"
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.removeprefix(b"\xef\xbb\xbf").lstrip()[:1] in (b"[", b"{"):
        return "json"  # the head only; upload() parses the whole file
    if head.startswith(_BINARY_DXF):
        return "dxf"
    text = head.removeprefix(b"\xef\xbb\xbf").decode("latin-1")
    lines = text.splitlines()
    index = 0
    while index + 1 < len(lines):
        code, value = lines[index].strip(), lines[index + 1].strip()
        if code == "999":  # a DXF comment before the first section
            index += 2
            continue
        return "dxf" if code == "0" and value.upper() == "SECTION" else None
    return None


def _check_json(part: Path, total: int) -> str | None:
    """ "json" when the stored ``part`` parses as JSON within the batch file's limits."""
    from engineering import batch_file

    if total > batch_file.MAX_BYTES:
        return None
    try:
        batch_file.parse(part.read_bytes(), "upload")
    except ValueError:
        return None
    return "json"


def _transport_refusal(status: int, error: str, message: str, **headers) -> Response:
    return JSONResponse(
        {"ok": False, "error": error, "message": message}, status_code=status, headers=headers
    )


def _refusal(status: int, kind: str, message: str, **details) -> Response:
    from engineering.errors import describe

    spec = describe(kind)
    body = {
        "ok": False,
        "error": kind,
        "kind": kind,
        "category": spec.category,
        "retryable": False,
        "hint": spec.hint,
        "message": message,
        **{key: value for key, value in details.items() if value is not None},
    }
    return JSONResponse(body, status_code=status)


def _name_refusal(name: str) -> Response | None:
    """Why the client's ``name`` is refused, or None (only its stem and extension are used)."""
    if not name.strip():
        return _refusal(400, "invalid_args", "Name the file: POST /files?name=<file name>.")
    if (
        len(name) > 255
        or any(sep in name for sep in ("/", "\\", "\x00"))
        or ".." in name
        or any(ord(c) < 32 for c in name)
    ):
        return _refusal(
            400,
            "path_refused",
            "The name may not hold a path, '..' or a control character; send the bare file "
            "name. The server chooses where the file goes.",
            reason="traversal",
        )
    if ":" in name:
        return _refusal(
            400,
            "path_refused",
            "The name may not hold ':' (an alternate data stream).",
            reason="alternate_data_stream",
        )
    written = name.rstrip(" .").lower()
    suffix = Path(written).suffix
    if suffix in security.WRITE_REFUSED_SUFFIXES or written in security.WRITE_REFUSED_NAMES:
        return _refusal(
            400,
            "path_refused",
            f"'{suffix or written}' is a file AutoCAD or Windows loads or runs; it is never "
            "stored.",
            reason="executable_name",
            extension=suffix or written,
        )
    if suffix not in FORMATS:
        return _refusal(
            415,
            "path_refused",
            f"Uploads end in {', '.join(security.KINDS_UPLOAD)} (got {suffix or 'no extension'!r}).",
            reason="extension_not_allowed",
            extension=suffix,
            allowed=list(security.KINDS_UPLOAD),
        )
    return None


def issue_ticket(name: str) -> str:
    """A one-time token for uploading ``name`` (refused here by the same name rules)."""
    refused = _name_refusal(name)
    if refused is not None:
        body = json.loads(refused.body)
        details = {k: body[k] for k in ("reason", "extension", "allowed") if k in body}
        if body["kind"] == "path_refused":
            raise security.PathRefusedError(body["message"], **details)
        from engineering.errors import InvalidArgsError

        raise InvalidArgsError(f"invalid_args: {body['message']}")
    now = time.monotonic()
    for token in [t for t, ticket in _tickets.items() if ticket["expires"] <= now]:
        del _tickets[token]
    while len(_tickets) >= MAX_TICKETS:
        del _tickets[next(iter(_tickets))]
    token = secrets.token_urlsafe(32)
    _tickets[token] = {"name": name, "expires": now + UPLOAD_LINK_TTL_S}
    return token


def spend_ticket(token: str) -> dict | None:
    """Remove ``token`` and return its ticket while it is still valid, else None.

    A dict lookup on the token (no comparison loop to time); the ticket is
    gone after this call whatever happens next.
    """
    ticket = _tickets.pop(token, None)
    if ticket is None or ticket["expires"] <= time.monotonic():
        return None
    return ticket


def mask_token(text: str) -> str:
    """``text`` with any one-time upload token replaced by ``***``."""
    return _TOKEN_IN_PATH.sub(UPLOAD_PATH + "/***", text)


_TOKEN_IN_PATH = re.compile(re.escape(UPLOAD_PATH) + r"/[^\s?\"' ]+")


class _MaskTokens(logging.Filter):
    """Keeps one-time upload tokens out of uvicorn's access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(mask_token(a) if isinstance(a, str) else a for a in record.args)
        if isinstance(record.msg, str):
            record.msg = mask_token(record.msg)
        return True


async def _authorize(server, request: Request) -> Response | None:
    verifier = getattr(server.mcp, "auth", None)
    if verifier is None:
        return _transport_refusal(
            404,
            "not_found",
            "Uploads need MCP_AUTH_TOKEN; a server without a token reads its files locally.",
        )
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    access = None
    if scheme.lower() == "bearer" and token.strip():
        access = await verifier.verify_token(token.strip())
    if access is None:
        return _transport_refusal(
            401,
            "unauthorized",
            "Send the MCP bearer token: Authorization: Bearer <MCP_AUTH_TOKEN>.",
            **{"www-authenticate": 'Bearer realm="autocad-mcp-pro"'},
        )
    if http_guard.WRITE_SCOPE not in set(access.scopes or ()):
        return _refusal(
            403,
            "policy_refused",
            "An upload writes a file; this token has only mcp:read. Use MCP_AUTH_TOKEN.",
            reason="read_only_token",
        )
    return None


async def upload(server, request: Request, ticket: str | None = None) -> Response:
    if ticket is not None:
        # spent first: a refusal below must not leave the link usable
        granted = spend_ticket(ticket)
        if not enabled():
            return _transport_refusal(404, "not_found", "Uploads are off (UPLOAD_MAX_BYTES=0).")
        if granted is None or getattr(server.mcp, "auth", None) is None:
            return _transport_refusal(
                404,
                "not_found",
                "This upload link is unknown, expired or already used; ask file_upload_url "
                "for a new one.",
            )
        name = granted["name"]
    else:
        if not enabled():
            return _transport_refusal(404, "not_found", "Uploads are off (UPLOAD_MAX_BYTES=0).")
        refused = await _authorize(server, request)
        if refused is not None:
            return refused
        name = request.query_params.get("name", "")
    refused = _name_refusal(name)
    if refused is not None:
        return refused
    suffix = Path(name.rstrip(" .")).suffix.lower()
    stem = _STEM_CHARS.sub("_", Path(name).stem).strip("_")[:40] or "upload"
    folder = upload_dir()
    target = folder / f"{stem}-{secrets.token_hex(8)}{suffix}"
    try:
        # Inside ALLOWED_PATHS, an upload extension, not a loadable name, and the
        # server's write policy (READ_ONLY / MAX_TOOL_COST) - what every writer passes.
        target = security.validate_path(
            str(target),
            allow_write=True,
            kinds=security.KINDS_UPLOAD,
            create_parents=True,
            into_uploads=True,
        )
    except security.PathRefusedError as exc:
        return _refusal(403, "path_refused", str(exc), **exc.details)
    except PolicyRefusedError as exc:
        return _refusal(403, "policy_refused", str(exc), reason="write_policy")
    sweep(folder)

    limit = int(config.settings.upload_max_bytes)
    try:
        declared = int(request.headers.get("content-length", "-1"))  # the guard checked it
    except ValueError:
        declared = -1
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    total = 0
    head = b""
    try:
        with open(part, "xb") as out:
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit or total > declared:
                    break
                if len(head) < _SNIFF_BYTES:
                    head += chunk[: _SNIFF_BYTES - len(head)]
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    except ClientDisconnect:
        part.unlink(missing_ok=True)
        return _transport_refusal(400, "incomplete_body", "The client ended the upload early.")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if total > limit:
        part.unlink(missing_ok=True)
        return _transport_refusal(
            413, "body_too_large", f"The upload is over UPLOAD_MAX_BYTES={limit}."
        )
    if total != declared:
        part.unlink(missing_ok=True)
        return _transport_refusal(
            400, "incomplete_body", f"Content-Length said {declared} bytes; {total} arrived."
        )
    expected = FORMATS[suffix]
    found = sniff(head)
    if found == "json" == expected:
        found = _check_json(part, total)
    if found != expected:
        part.unlink(missing_ok=True)
        return _refusal(
            422,
            "invalid_args",
            f"'{name}' does not hold {expected} content (it reads as {found or 'unknown'}); "
            "nothing was stored.",
            reason="content_mismatch",
            expected=expected,
            found=found,
        )
    os.replace(part, target)
    log.info("upload stored: %s (%d bytes, %s)", target, total, expected)
    return JSONResponse(
        {
            "ok": True,
            "path": str(target),
            "sha256": digest.hexdigest(),
            "bytes": total,
            "format": expected,
            "expires_in_s": int(config.settings.upload_retain_s),
        },
        status_code=201,
    )


def register(server) -> None:
    """Add ``POST /files`` to ``server.mcp``."""

    @server.mcp.custom_route(UPLOAD_PATH, methods=["POST"])
    async def file_upload(request: Request) -> Response:
        return await upload(server, request)

    @server.mcp.custom_route(UPLOAD_PATH + "/{token}", methods=["POST"])
    async def file_upload_ticket(request: Request) -> Response:
        return await upload(server, request, ticket=request.path_params["token"])

    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _MaskTokens) for f in access.filters):
        access.addFilter(_MaskTokens())
