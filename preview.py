"""A read-only browser page showing what each MCP session is drawing (``/preview``).

A headless server has no AutoCAD window to watch. ``/preview`` lists the
server's sessions (one per engine, see backends/session_backends.py), each
session's open documents, a render of its active document that refreshes when
the document's revision moves, its recent tool calls with their duration and data size, and
PDF / DXF / DWG downloads of the active document. Nothing on the page changes a
drawing.

The routes sit beside ``/mcp`` but outside its bearer-token check, and the
MCP token is not a browser credential, so the page has its own: HTTP Basic
auth with ``PREVIEW_USER`` / ``PREVIEW_PASSWORD``. Without ``PREVIEW_PASSWORD``
every preview route answers 404, as if it did not exist.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import tempfile
import time
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

import config

log = logging.getLogger(__name__)

_HISTORY_ROWS = 15


def _enabled() -> bool:
    return bool(config.settings.preview_password)


def _authorized(request: Request) -> bool:
    """HTTP Basic credentials equal to PREVIEW_USER / PREVIEW_PASSWORD, compared in constant time."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        user, _, password = base64.b64decode(header[6:].strip()).decode("utf-8").partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    settings = config.settings
    user_ok = hmac.compare_digest(user.encode(), settings.preview_user.encode())
    password_ok = hmac.compare_digest(password.encode(), settings.preview_password.encode())
    return user_ok and password_ok


def _guard(request: Request) -> Response | None:
    """The refusal for ``request``, or None when it may see the page."""
    if not _enabled():
        return JSONResponse({"error": "not_found"}, status_code=404)
    if not _authorized(request):
        return JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
            headers={"www-authenticate": 'Basic realm="AutoCAD MCP preview", charset="UTF-8"'},
        )
    return None


def _session_id(session: str | None) -> str:
    """A short public id: the MCP session id itself never leaves the server."""
    return "server" if session is None else hashlib.sha256(session.encode()).hexdigest()[:12]


def _engines(server) -> list[tuple[str, object, float]]:
    from backends.session_backends import SessionBackends

    lifespan = getattr(server.mcp, "_lifespan_result", None) or {}
    backend = lifespan.get("backend") if isinstance(lifespan, dict) else None
    if backend is None:
        return []
    if isinstance(backend, SessionBackends):
        return [(_session_id(key), engine, used) for key, engine, used in backend.engines()]
    return [("server", backend, 0.0)]


def _engine(server, wanted: str):
    for sid, engine, _ in _engines(server):
        if sid == wanted:
            return engine
    return None


def _calls(engine) -> dict:
    """The engine's recent tool calls, newest first, and their totals."""
    from backends.session_backends import call_log

    calls = call_log(engine)
    totals = {
        "calls": len(calls),
        "ms": round(sum(c["ms"] for c in calls), 1),
        "bytes_in": sum(c["bytes_in"] for c in calls),
        "bytes_out_text": sum(c["bytes_out_text"] for c in calls),
        "bytes_out_image": sum(c["bytes_out_image"] for c in calls),
    }
    return {"recent": list(reversed(calls))[:_HISTORY_ROWS], "totals": totals}


async def _state(server) -> dict:
    sessions = []
    now = time.monotonic()
    for sid, engine, used in _engines(server):
        try:
            documents = await engine.document_list()
        except Exception as exc:
            documents, error = [], f"{type(exc).__name__}: {exc}"
        else:
            error = None
        active = next((d for d in documents if d.get("active")), None)
        sessions.append(
            {
                "id": sid,
                "idle_s": None if not used else round(now - used),
                "documents": [
                    {k: d.get(k) for k in ("name", "active", "saved", "entity_count", "revision")}
                    for d in documents
                ],
                "active": None if active is None else active["name"],
                "revision": None if active is None else active["revision"],
                "calls": _calls(engine),
                "error": error,
            }
        )
    return {"sessions": sessions, "time": time.time()}


def register(server) -> None:
    """Add the ``/preview`` routes to ``server.mcp``."""
    mcp = server.mcp

    @mcp.custom_route("/preview", methods=["GET"])
    async def preview_page(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        return HTMLResponse(_PAGE)

    @mcp.custom_route("/preview/state", methods=["GET"])
    async def preview_state(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        return JSONResponse(await _state(server), headers={"cache-control": "no-store"})

    @mcp.custom_route("/preview/render.png", methods=["GET"])
    async def preview_render(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        engine = _engine(server, request.query_params.get("session", ""))
        if engine is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        try:
            report = await engine.view_render(region=None, max_px=1600)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=409)
        png = report.get("png") if isinstance(report, dict) else None
        if png is None:
            return JSONResponse({"error": str(report.get("error") or "no render")}, status_code=409)
        return Response(png, media_type="image/png", headers={"cache-control": "no-store"})

    @mcp.custom_route("/preview/download", methods=["GET"])
    async def preview_download(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        engine = _engine(server, request.query_params.get("session", ""))
        fmt = request.query_params.get("format", "dxf")
        if engine is None or fmt not in _FORMATS:
            return JSONResponse({"error": "unknown session or format"}, status_code=404)
        if not hasattr(engine, "dxf_bytes"):
            return JSONResponse({"error": "downloads need the headless engine"}, status_code=501)
        try:
            name, data = await engine.dxf_bytes()
            if fmt != "dxf":
                with tempfile.TemporaryDirectory(prefix="acadmcp-preview-") as folder:
                    path = Path(folder) / f"{Path(name).stem}.{fmt}"
                    exporter = (
                        engine.drawing_export_dwg if fmt == "dwg" else engine.drawing_export_pdf
                    )
                    report = await exporter(str(path))
                    if isinstance(report, dict) and report.get("ok") is False:
                        raise RuntimeError(report.get("error") or f"{fmt} export failed")
                    name, data = path.name, path.read_bytes()
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=409)
        return Response(
            data,
            media_type=_FORMATS[fmt],
            headers={"content-disposition": f'attachment; filename="{name}"'},
        )


#: Download formats and their media types: DXF straight from memory, DWG
#: through ODA File Converter, PDF through ezdxf's renderer (model space).
_FORMATS = {"dxf": "image/vnd.dxf", "dwg": "application/acad", "pdf": "application/pdf"}


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoCAD MCP preview</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--text:#e6e8ee;--dim:#8b93a7;--accent:#4c8dff;--ok:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh}
aside{width:280px;border-right:1px solid var(--line);overflow:auto;padding:12px}
main{flex:1;display:flex;flex-direction:column;min-width:0}
h1{font-size:15px;margin:4px 4px 12px}h2{font-size:13px;color:var(--dim);font-weight:600;margin:16px 0 6px;text-transform:uppercase;letter-spacing:.04em}
.s{padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin-bottom:8px;cursor:pointer;background:var(--panel)}
.s.sel{border-color:var(--accent)}.s .id{font-family:ui-monospace,monospace;font-size:12px;color:var(--dim)}
.s .doc{font-weight:600;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line)}
.bar .t{font-weight:600}.bar .m{color:var(--dim)}.sp{flex:1}
a.b{color:var(--text);text-decoration:none;border:1px solid var(--line);padding:5px 10px;border-radius:6px;background:var(--panel)}a.b:hover{border-color:var(--accent)}
.view{flex:1;display:flex;min-height:0}.canvas{flex:1;display:flex;align-items:center;justify-content:center;background:#212830;margin:12px;border-radius:8px;overflow:hidden;position:relative}
.canvas img{max-width:100%;max-height:100%;object-fit:contain}.empty{color:var(--dim)}
.side{width:420px;border-left:1px solid var(--line);overflow:auto;padding:0 12px 12px}
table{width:100%;border-collapse:collapse;font-size:12px}td{padding:4px 2px;border-bottom:1px solid var(--line);vertical-align:top}
td.n{font-family:ui-monospace,monospace;white-space:nowrap;padding-right:8px}td.r{text-align:right;white-space:nowrap}#tot{margin:-2px 0 6px;font-family:ui-monospace,monospace;font-size:12px;color:var(--dim)}.committed{color:var(--ok)}.failed,.refused{color:var(--bad)}.partial,.unknown_commit{color:var(--warn)}
.pill{font-size:11px;padding:1px 6px;border-radius:10px;border:1px solid var(--line);color:var(--dim)}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:6px}
</style></head><body>
<aside><h1>AutoCAD MCP preview</h1><h2>Sessions</h2><div id="sessions"></div></aside>
<main>
 <div class="bar"><span class="t" id="title">No session selected</span><span class="m" id="meta"></span><span class="sp"></span>
  <a class="b" id="pdf" href="#">PDF</a><a class="b" id="dxf" href="#">DXF</a><a class="b" id="dwg" href="#">DWG</a></div>
 <div class="view"><div class="canvas"><img id="img" alt="Render of the active document" hidden><div class="empty" id="empty">Nothing drawn yet</div></div>
  <div class="side"><h2>Open documents</h2><table id="docs"></table><h2>Recent calls</h2><div class="id" id="tot"></div><table id="hist"></table></div></div>
</main>
<script>
let sel=null,shown=null;const $=id=>document.getElementById(id);
const fmtB=b=>b>=1048576?(b/1048576).toFixed(1)+' MB':b>=1024?(b/1024).toFixed(1)+' kB':b+' B';
const fmtMs=m=>m>=1000?(m/1000).toFixed(1)+' s':Math.round(m)+' ms';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function tick(){
 let st;try{const r=await fetch('preview/state',{cache:'no-store'});if(!r.ok)throw 0;st=await r.json()}catch(e){$('meta').textContent='server unreachable';return}
 const ss=st.sessions;if(!ss.find(s=>s.id===sel))sel=(ss.find(s=>s.active)||ss[0]||{}).id||null;
 $('sessions').innerHTML=ss.map((s,i)=>`<div class="s ${s.id===sel?'sel':''}" data-id="${s.id}"><div class="id"><span class="dot" style="opacity:${s.idle_s!=null&&s.idle_s<60?1:.25}"></span>${s.id==='server'?'server engine':'session '+esc(s.id)}</div><div class="doc">${esc(s.active||'no document')}</div><div class="id">${s.documents.length} open · ${fmtB(s.calls.totals.bytes_in+s.calls.totals.bytes_out_text+s.calls.totals.bytes_out_image)}${s.idle_s!=null?' · idle '+s.idle_s+'s':''}</div></div>`).join('')||'<div class="empty" style="color:var(--dim)">No sessions yet</div>';
 document.querySelectorAll('.s').forEach(e=>e.onclick=()=>{sel=e.dataset.id;shown=null;tick()});
 const s=ss.find(x=>x.id===sel);
 if(!s){$('title').textContent='No session';return}
 $('title').textContent=s.active||'No document open';
 $('meta').textContent=s.active?`revision ${s.revision}`:'';
 for(const f of['pdf','dxf','dwg']){$(f).href=`preview/download?session=${s.id}&format=${f}`;$(f).style.visibility=s.active?'visible':'hidden'}
 const key=s.id+':'+s.active+':'+s.revision;
 if(s.active&&key!==shown){const img=new Image();img.onload=()=>{$('img').src=img.src;$('img').hidden=false;$('empty').hidden=true;shown=key};img.onerror=()=>{shown=key};img.src=`preview/render.png?session=${s.id}&r=${encodeURIComponent(key)}`}
 if(!s.active){$('img').hidden=true;$('empty').hidden=false;shown=key}
 $('docs').innerHTML=s.documents.map(d=>`<tr><td>${esc(d.name)} ${d.active?'<span class="pill">active</span>':''}</td><td class="n">${d.entity_count} ent</td><td>${d.saved?'saved':'unsaved'}</td></tr>`).join('')||'<tr><td>none</td></tr>';
 const t=s.calls.totals,tb=t.bytes_in+t.bytes_out_text+t.bytes_out_image;$('tot').textContent=t.calls?`${t.calls} calls · ${fmtMs(t.ms)} · ${fmtB(tb)} (in ${fmtB(t.bytes_in)} / out text ${fmtB(t.bytes_out_text)}${t.bytes_out_image?' + images '+fmtB(t.bytes_out_image):''})`:'';
 $('hist').innerHTML=s.calls.recent.map(h=>`<tr title="in ${fmtB(h.bytes_in)} · out text ${fmtB(h.bytes_out_text)}${h.bytes_out_image?' + image '+fmtB(h.bytes_out_image):''}"><td class="n">${new Date(h.started*1000).toLocaleTimeString('en-GB',{hour12:false})}</td><td class="n">${esc(h.tool)}</td><td class="n r">${fmtMs(h.ms)}</td><td class="n r">${fmtB(h.bytes_in+h.bytes_out_text+h.bytes_out_image)}${h.bytes_out_image?' 🖼':''}</td><td class="${h.ok?'committed':'failed'}">${h.ok?'ok':'error'}</td><td class="n">${h.revision!=null?'r'+h.revision:''}</td></tr>`).join('')||'<tr><td>none yet</td></tr>';
}
tick();setInterval(tick,2000);
</script></body></html>
"""
