"""A read-only browser page showing what each MCP session is drawing (``/preview``).

A headless server has no AutoCAD window to watch. ``/preview`` lists the
server's sessions (one per engine, see backends/session_backends.py), each
session's open documents, a render of its active document that refreshes when
the document's revision moves, its recent tool calls with their duration and data size, and
PDF / DXF / DWG downloads of the active document. ``/preview/tools`` counts the
calls of every tool and their errors by kind, across all sessions, read from the
operation journal (backends/journal_store.py): it outlives a restart and keeps
``JOURNAL_RETAIN_DAYS``. Nothing on either page changes a drawing.

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
                    {
                        k: d.get(k)
                        for k in (
                            "name",
                            "active",
                            "saved",
                            "entity_count",
                            "revision",
                            "document_id",
                        )
                    }
                    for d in documents
                ],
                "active": None if active is None else active["name"],
                "revision": None if active is None else active["revision"],
                "calls": _calls(engine),
                "error": error,
            }
        )
    return {"sessions": sessions, "time": time.time()}


#: ``/preview/tools`` ranges: seconds back from now, None for every kept row.
_RANGES = {"24h": 86400.0, "7d": 7 * 86400.0, "all": None}


async def _tool_usage(server, window: str) -> dict:
    """The journal's calls per tool over ``window`` (a key of ``_RANGES``)."""
    import asyncio

    journal = server._operations_journal(getattr(server.mcp, "_lifespan_result", None))
    back = _RANGES[window]
    since = None if back is None else time.time() - back
    tools = await asyncio.to_thread(journal.tool_usage, since)
    settings = getattr(journal, "settings", None)
    durable = getattr(journal, "_connection", None) is not None
    return {
        "range": window,
        "since": since,
        "durable": durable,
        "retain_days": getattr(settings, "retain_days", None) if durable else None,
        "calls": sum(t["calls"] for t in tools),
        "errors": sum(t["errors"] for t in tools),
        "tools": tools,
    }


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

    @mcp.custom_route("/preview/tools", methods=["GET"])
    async def preview_tools_page(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        return HTMLResponse(_TOOLS_PAGE)

    @mcp.custom_route("/preview/tools.json", methods=["GET"])
    async def preview_tools(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        window = request.query_params.get("range", "all")
        if window not in _RANGES:
            return JSONResponse({"error": f"range is one of {list(_RANGES)}"}, status_code=400)
        return JSONResponse(
            await _tool_usage(server, window), headers={"cache-control": "no-store"}
        )

    @mcp.custom_route("/preview/render.png", methods=["GET"])
    async def preview_render(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        engine = _engine(server, request.query_params.get("session", ""))
        if engine is None:
            return JSONResponse({"error": "no such session"}, status_code=404)
        document = request.query_params.get("document") or None
        if document is not None and not hasattr(engine, "dxf_bytes"):
            return JSONResponse(
                {"error": "another document needs the headless engine"}, status_code=501
            )
        # The page zooms and pans by asking for the window it shows: ``region``
        # is x0,y0,x1,y1 in drawing units, ``max_px`` the long side it needs.
        try:
            region = request.query_params.get("region")
            region = None if not region else [float(v) for v in region.split(",")]
            max_px = min(2048, max(64, int(request.query_params.get("max_px") or 1600)))
        except ValueError:
            return JSONResponse({"error": "region must be x0,y0,x1,y1"}, status_code=400)
        try:
            extra = {} if document is None else {"document": document}
            report = await engine.view_render(region=region, max_px=max_px, **extra)
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=409)
        png = report.get("png") if isinstance(report, dict) else None
        if png is None:
            return JSONResponse({"error": str(report.get("error") or "no render")}, status_code=409)
        headers = {"cache-control": "no-store"}
        if report.get("region"):  # the window actually drawn, for the page's pan and zoom
            headers["x-render-region"] = ",".join(f"{v:.9g}" for v in report["region"])
        return Response(png, media_type="image/png", headers=headers)

    @mcp.custom_route("/preview/download", methods=["GET"])
    async def preview_download(request: Request) -> Response:
        if (refused := _guard(request)) is not None:
            return refused
        engine = _engine(server, request.query_params.get("session", ""))
        fmt = request.query_params.get("format", "dxf")
        document = request.query_params.get("document") or None
        if engine is None or fmt not in _FORMATS:
            return JSONResponse({"error": "unknown session or format"}, status_code=404)
        if not hasattr(engine, "dxf_bytes"):
            return JSONResponse({"error": "downloads need the headless engine"}, status_code=501)
        try:
            name, data = await engine.dxf_bytes(document)
            if fmt != "dxf":
                with tempfile.TemporaryDirectory(prefix="acadmcp-preview-") as folder:
                    path = Path(folder) / f"{Path(name).stem}.{fmt}"
                    exporter = (
                        engine.drawing_export_dwg if fmt == "dwg" else engine.drawing_export_pdf
                    )
                    report = await exporter(str(path), document=document)
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
.canvas{cursor:grab;touch-action:none}.canvas.drag{cursor:grabbing}.canvas img{position:absolute;pointer-events:none;user-select:none}.canvas img.fit{position:static;max-width:100%;max-height:100%;object-fit:contain}
.hint{position:absolute;left:10px;bottom:8px;font-size:11px;color:var(--dim)}.zb{position:absolute;right:10px;top:10px;cursor:pointer}.empty{color:var(--dim)}
.side{width:420px;border-left:1px solid var(--line);overflow:auto;padding:0 12px 12px}
table{width:100%;border-collapse:collapse;font-size:12px}td{padding:4px 2px;border-bottom:1px solid var(--line);vertical-align:top}
td.n{font-family:ui-monospace,monospace;white-space:nowrap;padding-right:8px}td.r{text-align:right;white-space:nowrap}#tot{margin:-2px 0 6px;font-family:ui-monospace,monospace;font-size:12px;color:var(--dim)}.committed{color:var(--ok)}.failed,.refused{color:var(--bad)}.partial,.unknown_commit{color:var(--warn)}
tr.d{cursor:pointer}tr.d:hover td{background:var(--panel)}tr.d.sel td{background:var(--panel);box-shadow:inset 0 -1px var(--accent)}
.pill{font-size:11px;padding:1px 6px;border-radius:10px;border:1px solid var(--line);color:var(--dim)}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:6px}
</style></head><body>
<aside><h1>AutoCAD MCP preview</h1><a class="b" href="preview/tools">Tool usage</a><h2>Sessions</h2><div id="sessions"></div></aside>
<main>
 <div class="bar"><span class="t" id="title">No session selected</span><span class="m" id="meta"></span><span class="sp"></span>
  <a class="b" id="pdf" href="#">PDF</a><a class="b" id="dxf" href="#">DXF</a><a class="b" id="dwg" href="#">DWG</a></div>
 <div class="view"><div class="canvas" id="cv"><img id="img" class="fit" alt="Render of the selected document" draggable="false" hidden><div class="empty" id="empty">Nothing drawn yet</div><a class="b zb" id="fitb" hidden>Fit</a><div class="hint" id="hint" hidden>wheel or pinch to zoom · drag or two-finger scroll to pan · double-click to fit</div></div>
  <div class="side"><h2>Open documents</h2><div class="id" style="margin:-2px 0 6px;color:var(--dim);font-size:12px">click a document to view it</div><table id="docs"></table><h2>Recent calls</h2><div class="id" id="tot"></div><table id="hist"></table></div></div>
</main>
<script>
let sel=null,shown=null,pick=null,cur=null,view=null,imgR=null,moved=false,seq=0,timer=null;const $=id=>document.getElementById(id);
function place(){const c=$('cv'),img=$('img');if(!imgR||!view){img.className='fit';img.style.cssText='';return}img.className='';const W=c.clientWidth,H=c.clientHeight,s=view.s,x0=view.cx-W/2/s,y1=view.cy+H/2/s;Object.assign(img.style,{left:(imgR[0]-x0)*s+'px',top:(y1-imgR[3])*s+'px',width:(imgR[2]-imgR[0])*s+'px',height:(imgR[3]-imgR[1])*s+'px'})}
function fit(r){const c=$('cv');view={cx:(r[0]+r[2])/2,cy:(r[1]+r[3])/2,s:Math.min(c.clientWidth/(r[2]-r[0]),c.clientHeight/(r[3]-r[1]))*0.95}}
async function render(){if(!cur)return;const n=++seq,c=$('cv'),W=c.clientWidth,H=c.clientHeight;let u=`preview/render.png?${cur.q}&r=${encodeURIComponent(cur.key)}`;const zoomed=moved&&view;
 if(zoomed){const s=view.s;u+=`&region=${[view.cx-W/2/s,view.cy-H/2/s,view.cx+W/2/s,view.cy+H/2/s].join(',')}&max_px=${Math.min(2048,Math.round(Math.max(W,H)*(devicePixelRatio||1)))}`}
 try{const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw 0;const url=URL.createObjectURL(await r.blob());const im=new Image();im.src=url;await im.decode().catch(()=>{});
  if(n!==seq){URL.revokeObjectURL(url);return}const old=$('img').src;$('img').src=url;if(old.startsWith('blob:'))URL.revokeObjectURL(old);
  const h=r.headers.get('x-render-region');imgR=h?h.split(',').map(Number):null;if(imgR&&!zoomed)fit(imgR);
  $('img').hidden=false;$('empty').hidden=true;$('hint').hidden=$('fitb').hidden=!imgR;place()}catch(e){}}
function later(){clearTimeout(timer);timer=setTimeout(render,250)}
function reset(){moved=false;render()}
const fmtB=b=>b>=1048576?(b/1048576).toFixed(1)+' MB':b>=1024?(b/1024).toFixed(1)+' kB':b+' B';
const fmtMs=m=>m>=1000?(m/1000).toFixed(1)+' s':Math.round(m)+' ms';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function tick(){
 let st;try{const r=await fetch('preview/state',{cache:'no-store'});if(!r.ok)throw 0;st=await r.json()}catch(e){$('meta').textContent='server unreachable';return}
 const ss=st.sessions;if(!ss.find(s=>s.id===sel))sel=(ss.find(s=>s.active)||ss[0]||{}).id||null;
 $('sessions').innerHTML=ss.map((s,i)=>`<div class="s ${s.id===sel?'sel':''}" data-id="${s.id}"><div class="id"><span class="dot" style="opacity:${s.idle_s!=null&&s.idle_s<60?1:.25}"></span>${s.id==='server'?'server engine':'session '+esc(s.id)}</div><div class="doc">${esc(s.active||'no document')}</div><div class="id">${s.documents.length} open · ${fmtB(s.calls.totals.bytes_in+s.calls.totals.bytes_out_text+s.calls.totals.bytes_out_image)}${s.idle_s!=null?' · idle '+s.idle_s+'s':''}</div></div>`).join('')||'<div class="empty" style="color:var(--dim)">No sessions yet</div>';
 document.querySelectorAll('.s').forEach(e=>e.onclick=()=>{sel=e.dataset.id;pick=null;shown=null;tick()});
 const s=ss.find(x=>x.id===sel);
 if(!s){$('title').textContent='No session';return}
 const d=s.documents.find(x=>x.document_id===pick)||s.documents.find(x=>x.active)||null;pick=d&&!d.active?d.document_id:null;
 const q=`session=${s.id}`+(pick?`&document=${encodeURIComponent(pick)}`:'');
 $('title').textContent=d?d.name:'No document open';
 $('meta').textContent=d?`revision ${d.revision}`+(d.active?'':' · not active'):'';
 for(const f of['pdf','dxf','dwg']){$(f).href=`preview/download?${q}&format=${f}`;$(f).style.visibility=d?'visible':'hidden'}
 const doc=s.id+':'+(d&&d.document_id),key=doc+':'+(d&&d.revision);
 if(d&&key!==shown){shown=key;if(!cur||cur.doc!==doc){moved=false;view=imgR=null}cur={q,key,doc};render()}
 if(!d){$('img').hidden=$('hint').hidden=$('fitb').hidden=true;$('empty').hidden=false;shown=key;cur=null}
 $('docs').innerHTML=s.documents.map(x=>`<tr class="d ${x===d?'sel':''}" data-id="${esc(x.document_id)}"><td>${esc(x.name)} ${x.active?'<span class="pill">active</span>':''}</td><td class="n">${x.entity_count} ent</td><td>${x.saved?'saved':'unsaved'}</td></tr>`).join('')||'<tr><td>none</td></tr>';
 document.querySelectorAll('tr.d').forEach(e=>e.onclick=()=>{pick=e.dataset.id;tick()});
 const t=s.calls.totals,tb=t.bytes_in+t.bytes_out_text+t.bytes_out_image;$('tot').textContent=t.calls?`${t.calls} calls · ${fmtMs(t.ms)} · ${fmtB(tb)} (in ${fmtB(t.bytes_in)} / out text ${fmtB(t.bytes_out_text)}${t.bytes_out_image?' + images '+fmtB(t.bytes_out_image):''})`:'';
 $('hist').innerHTML=s.calls.recent.map(h=>`<tr title="in ${fmtB(h.bytes_in)} · out text ${fmtB(h.bytes_out_text)}${h.bytes_out_image?' + image '+fmtB(h.bytes_out_image):''}"><td class="n">${new Date(h.started*1000).toLocaleTimeString('en-GB',{hour12:false})}</td><td class="n">${esc(h.tool)}</td><td class="n r">${fmtMs(h.ms)}</td><td class="n r">${fmtB(h.bytes_in+h.bytes_out_text+h.bytes_out_image)}${h.bytes_out_image?' 🖼':''}</td><td class="${h.ok?'committed':'failed'}">${h.ok?'ok':'error'}</td><td class="n">${h.revision!=null?'r'+h.revision:''}</td></tr>`).join('')||'<tr><td>none yet</td></tr>';
}
{const c=$('cv');let drag=null;
 // A pinch (ctrlKey) or a mouse wheel zooms; a two-finger trackpad scroll pans. A trackpad
 // shows itself by sideways or fractional/small pixel deltas, and stays one for the gesture.
 let pad=0;c.onwheel=e=>{if(!view||!imgR)return;e.preventDefault();const now=performance.now();
  if(!e.ctrlKey&&e.deltaMode===0&&(e.deltaX!==0||Math.abs(e.deltaY)<50||!Number.isInteger(e.deltaY)||now-pad<300))pad=now;
  if(!e.ctrlKey&&now===pad){view.cx+=e.deltaX/view.s;view.cy-=e.deltaY/view.s;moved=true;place();later();return}
  const r=c.getBoundingClientRect(),W=c.clientWidth,H=c.clientHeight,mx=e.clientX-r.left-W/2,my=e.clientY-r.top-H/2;
  const px=view.cx+mx/view.s,py=view.cy-my/view.s,dy=e.deltaY*(e.deltaMode===1?33:1);view.s*=Math.exp(-Math.max(-300,Math.min(300,dy))*(e.ctrlKey?0.01:0.0015));
  view.cx=px-mx/view.s;view.cy=py+my/view.s;moved=true;place();later()};
 c.onpointerdown=e=>{if(!view||!imgR||e.button!==0||e.target.id==='fitb')return;drag={x:e.clientX,y:e.clientY};c.setPointerCapture(e.pointerId);c.classList.add('drag')};
 c.onpointermove=e=>{if(!drag)return;view.cx-=(e.clientX-drag.x)/view.s;view.cy+=(e.clientY-drag.y)/view.s;drag={x:e.clientX,y:e.clientY};moved=true;place()};
 c.onpointerup=c.onpointercancel=()=>{if(!drag)return;drag=null;c.classList.remove('drag');if(moved)later()};
 c.ondblclick=reset;$('fitb').onclick=reset;window.onresize=()=>{place();later()}}
tick();setInterval(tick,2000);
</script></body></html>
"""


_TOOLS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AutoCAD MCP tool usage</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--text:#e6e8ee;--dim:#8b93a7;--accent:#4c8dff;--bad:#f85149}
*{box-sizing:border-box}body{margin:0;padding:20px 28px;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--text)}
h1{font-size:17px;margin:0}.bar{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
a.b,button{color:var(--text);text-decoration:none;border:1px solid var(--line);padding:5px 10px;border-radius:6px;background:var(--panel);font:13px system-ui,sans-serif;cursor:pointer}a.b:hover,button:hover{border-color:var(--accent)}
button.on{border-color:var(--accent);background:#1d2a44}.seg{display:flex;gap:4px;margin-left:auto}
#tot{color:var(--dim);margin-bottom:16px}table{border-collapse:collapse;width:100%}
th{text-align:right;color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em;padding:6px 10px;border-bottom:1px solid var(--line)}th:first-child,th:nth-child(5){text-align:left}
td{padding:7px 12px;border-bottom:1px solid var(--line);font-family:ui-monospace,monospace;font-size:14px;text-align:right;white-space:nowrap;vertical-align:top}td:first-child,td:nth-child(5){text-align:left}
th:nth-child(3){width:30%}td:nth-child(5){white-space:normal}.k{display:inline-block;white-space:nowrap;margin:0 6px 2px 0;padding:0 6px;border-radius:9px;border:1px solid #5a2a2a;color:#ffb3ad;font-size:12px}
.share{display:flex;align-items:center;gap:10px}.share b{flex:1;display:block}.share i{display:block;height:8px;background:var(--accent);border-radius:2px}.share span{width:44px}.bad{color:var(--bad)}.empty{color:var(--dim)}
</style></head><body>
<div class="bar"><h1>Tool usage</h1><a class="b" href="../preview">&larr; Sessions</a>
<div class="seg"><button data-r="24h">24 h</button><button data-r="7d">7 days</button><button data-r="all">All</button></div></div>
<div id="tot"></div>
<table><thead><tr><th>Tool</th><th>Calls</th><th>Share</th><th>Errors</th><th>Error kinds</th><th>Avg</th><th>Total time</th></tr></thead><tbody id="rows"></tbody></table>
<script>
const $=id=>document.getElementById(id);let range=new URLSearchParams(location.search).get('range')||'all';
const fmtMs=m=>m>=60000?(m/60000).toFixed(1)+' min':m>=1000?(m/1000).toFixed(1)+' s':Math.round(m)+' ms';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>{range=b.dataset.r;history.replaceState(null,'','?range='+range);tick()});
async function tick(){
 document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on',b.dataset.r===range));
 let st;try{const r=await fetch('tools.json?range='+range,{cache:'no-store'});if(!r.ok)throw 0;st=await r.json()}catch(e){$('tot').textContent='server unreachable';return}
 const span=st.range==='all'?(st.durable?`the journal's last ${st.retain_days} days`:'this run (no durable journal)'):(st.range==='24h'?'the last 24 hours':'the last 7 days');
 $('tot').textContent=`${st.calls} calls of ${st.tools.length} tools, ${st.errors} errors, all sessions, over ${span}`;
 const top=st.tools.length?st.tools[0].calls:1;
 $('rows').innerHTML=st.tools.map(t=>`<tr><td>${esc(t.tool)}</td><td>${t.calls}</td><td><span class="share"><b><i style="width:${Math.max(1,100*t.calls/top)}%"></i></b><span>${(100*t.calls/st.calls).toFixed(1)}%</span></span></td><td class="${t.errors?'bad':''}">${t.errors||''}</td><td>${Object.entries(t.error_kinds).map(([k,n])=>`<span class="k">${esc(k)} ${n}</span>`).join('')}</td><td>${fmtMs(t.ms/t.calls)}</td><td>${fmtMs(t.ms)}</td></tr>`).join('')||'<tr><td class="empty" colspan="7">No calls in this range</td></tr>';
}
tick();setInterval(tick,5000);
</script></body></html>
"""
