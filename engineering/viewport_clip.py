"""The ``viewport_clip`` critique focus: model lettering a viewport's frame cuts.

A sheet of several views shows each through its own viewport. When a window
is framed a little too tight it cuts the start of a view's title, or a little
too wide it lets in the tail of a neighbouring view's callout - and nothing
in the drawing is wrong, so every other focus stays silent. This focus is the
2D test of that framing: for each model viewport of each paper layout, every
model-space TEXT, MTEXT, ATTRIB, DIMENSION and MULTILEADER is either inside
its window, outside it, or **cut** by its frame - only the last is reported.

* A TEXT / ATTRIB is its oriented text box (``ezdxf.disassemble``); an MTEXT
  is the box :func:`ezdxf.tools.text_size.mtext_size` measures, placed by its
  attachment point and rotation (``ezdxf.bbox`` wraps an MTEXT with no width
  at every word, which made "ABS 2 mm" 14 x 23 mm instead of 36 x 5).
* A DIMENSION / MULTILEADER is the pieces it draws (lines exactly, texts as
  boxes, arrows and arcs as their boxes): it is cut when a piece crosses the
  frame, or when some pieces are inside and others outside.
* The window is the viewport's view centre, view height, aspect and twist,
  in model units. Layers frozen in the viewport, frozen or off layers, an
  invisible entity and a viewport that is off are not tested.

Coverage (1.8-16): one row per paper layout - ``checked`` with the number of
entities tested; ``viewport_scale_deferred`` when a viewport has no view
height (no window can be read); ``viewport_clip_boundary`` when a viewport is
clipped to a non-rectangular boundary (this lane tests rectangles only). A
drawing with no paper layout is ``domain_absent``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .plan_spec import Issue
from .qa_coverage import DOCUMENT, domain_row, ledger_of

if TYPE_CHECKING:
    from backends.base import AutoCADBackend

FOCUS = "viewport_clip"
CODE = "viewport_clips_text"

#: Model-space types whose lettering a viewport frame can cut.
CLIP_TYPES = frozenset({"TEXT", "MTEXT", "DIMENSION", "MULTILEADER", "INSERT"})

#: Items an issue lists in ``detail.items`` / ``handles`` (``count`` is complete).
MAX_ITEMS = 50

#: A point this close to the frame is on it (model units): touching is not cutting.
EPS = 1e-6

#: DXF VIEWPORT status flag: the viewport is clipped to a non-rectangular boundary.
_NON_RECT_CLIPPING = 0x10000

Pt = tuple[float, float]


def _mtext_corners(e) -> list[Pt]:
    """The box of the lettering itself, placed by the attachment point (1-9:
    top / middle / bottom x left / centre / right of the insert) and rotation.
    ``mtext_size`` answers the column width when one is set (100 mm for a
    5 mm "ABS 2 mm"); the ink is the unwrapped line width when that fits the
    column, aligned to the same side of the insert as the column."""
    from ezdxf.tools.text_size import mtext_size

    size = mtext_size(e)
    w, h = float(size.total_width), float(size.total_height)
    if not e.has_columns and float(e.dxf.get("width", 0.0) or 0.0) > 0:
        unwrapped = e.copy()
        unwrapped.dxf.width = 0.0
        w = min(w, float(mtext_size(unwrapped).total_width))
    ap = int(e.dxf.get("attachment_point", 1) or 1)
    col, row = (ap - 1) % 3, (ap - 1) // 3  # col 0 left / 1 centre / 2 right; row 0 top
    x0 = -w * col / 2.0
    y1 = h * row / 2.0
    ins = e.dxf.insert
    a = math.radians(float(e.get_rotation()))
    c, s = math.cos(a), math.sin(a)
    return [
        (ins.x + x * c - y * s, ins.y + x * s + y * c)
        for x, y in ((x0, y1 - h), (x0 + w, y1 - h), (x0 + w, y1), (x0, y1))
    ]


def _box_corners(e) -> list[Pt]:
    from ezdxf import bbox as ezbbox

    box = ezbbox.extents([e], fast=True)
    if not box.has_data:
        return []
    (x0, y0), (x1, y1) = box.extmin.vec2, box.extmax.vec2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _flat(e) -> bool:
    ext = e.dxf.get("extrusion", (0.0, 0.0, 1.0))
    return abs(ext[0]) < 1e-9 and abs(ext[1]) < 1e-9 and ext[2] > 0


def _piece(e) -> list[Pt]:
    """One drawn piece of an entity as the points of a convex outline."""
    kind = e.dxftype()
    if kind == "LINE":
        return [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
    if kind == "MTEXT" and _flat(e):
        return _mtext_corners(e) if (e.plain_text() or "").strip() else []
    if kind in ("TEXT", "ATTRIB") and _flat(e):
        if not str(e.dxf.get("text", "") or "").strip():
            return []
        from ezdxf import disassemble

        points = [(v.x, v.y) for v in disassemble.make_primitive(e).vertices()]
        return points[:-1] if len(points) > 1 and points[0] == points[-1] else points
    if kind == "POINT":  # a defpoint: DEFPOINTS does not plot
        return []
    return _box_corners(e)


def _items(doc, handle: str, kind: str) -> list[tuple[str, str, str, list[list[Pt]]]]:
    """``(handle, type, text, pieces)`` for one snapshot record: an INSERT
    yields one item per ATTRIB, the rest one item each."""
    e = doc.entitydb.get(handle)
    if e is None or not e.is_alive:
        return []
    if kind == "INSERT":
        return [
            (a.dxf.handle, "ATTRIB", str(a.dxf.text), [p])
            for a in e.attribs
            if not a.is_invisible and (p := _piece(a))
        ]
    if kind in ("DIMENSION", "MULTILEADER"):
        pieces = [p for v in e.virtual_entities() if (p := _piece(v))]
        text = e.dxf.get("text", "") if kind == "DIMENSION" else ""
        return [(handle, kind, str(text or ""), pieces)] if pieces else []
    piece = _piece(e)
    text = e.plain_text() if kind == "MTEXT" else str(e.dxf.get("text", ""))
    return [(handle, kind, str(text), [piece])] if piece else []


def _classify(points: list[Pt], view: dict) -> str:
    """``inside`` / ``outside`` / ``cut`` of one convex piece against the
    window (separating axes: the window's two, then the piece's edge normals)."""
    cx, cy = view["centre"]
    c, s = math.cos(view["twist"]), math.sin(view["twist"])
    hx, hy = view["half"]
    uv = [((x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c) for x, y in points]
    if all(abs(u) <= hx + EPS and abs(v) <= hy + EPS for u, v in uv):
        return "inside"
    us, vs = [u for u, _ in uv], [v for _, v in uv]
    if min(us) >= hx - EPS or max(us) <= -hx + EPS or min(vs) >= hy - EPS or max(vs) <= -hy + EPS:
        return "outside"
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    n = len(uv)
    for i in range(n if n > 2 else 1):
        (x0, y0), (x1, y1) = uv[i], uv[(i + 1) % n]
        nx, ny = y0 - y1, x1 - x0
        if nx == 0 and ny == 0:
            continue
        piece = [nx * u + ny * v for u, v in uv]
        window = [nx * u + ny * v for u, v in corners]
        tol = EPS * math.hypot(nx, ny)
        if min(window) >= max(piece) - tol or max(window) <= min(piece) + tol:
            return "outside"
    return "cut"


def _status(pieces: list[list[Pt]], view: dict) -> str:
    seen = {_classify(p, view) for p in pieces}
    if seen == {"inside"}:
        return "inside"
    if seen == {"outside"}:
        return "outside"
    return "cut"


def _scan(doc, records) -> dict:
    """Per paper layout: its viewports' rows, and what each cuts."""
    from .critique import _read_viewports

    hidden = {layer.dxf.name.lower() for layer in doc.layers if layer.is_frozen() or layer.is_off()}
    items = []
    for r in records:
        if r.space != "Model" or r.type not in CLIP_TYPES or not r.visible or r.tilted:
            continue
        if str(r.layer).lower() in hidden:
            continue
        for item in _items(doc, r.handle, r.type):
            items.append((str(r.layer).lower(), *item))
    out: dict[str, dict] = {}
    for layout, views in _read_viewports(doc).items():
        rows = out.setdefault(layout, {"reason": None, "n": 0, "cuts": []})
        for view in views:
            vp = doc.entitydb.get(view["handle"])
            if vp is None or int(vp.dxf.get("status", 1) or 0) == 0:
                continue  # a viewport that is off shows nothing
            if int(vp.dxf.get("flags", 0) or 0) & _NON_RECT_CLIPPING and vp.dxf.get(
                "clipping_boundary_handle", "0"
            ) not in ("0", None):
                rows["reason"] = rows["reason"] or "viewport_clip_boundary"
                continue
            if view["half"] is None:
                rows["reason"] = rows["reason"] or "viewport_scale_deferred"
                continue
            seen = [it for it in items if it[0] not in view["frozen"]]
            rows["n"] += len(seen)
            cut = [it[1:4] for it in seen if _status(it[4], view) == "cut"]
            if cut:
                rows["cuts"].append((view["handle"], cut))
    return out


def _issue(layout: str, viewport: str, cut: list[tuple[str, str, str]]) -> Issue:
    shown = ", ".join(f"{kind} {h} '{text[:30]}'" for h, kind, text in cut[:3])
    more = f" and {len(cut) - 3} more" if len(cut) > 3 else ""
    return Issue(
        severity="warning",
        focus=FOCUS,  # type: ignore[arg-type]
        message=(
            f"Viewport {viewport} on '{layout}' cuts {len(cut)} model text / dimension "
            f"entit{'y' if len(cut) == 1 else 'ies'} at its frame: {shown}{more}. "
            "Reframe the viewport (view centre / view height) or move the text."
        ),
        handles=[h for h, _k, _t in cut[:MAX_ITEMS]],
        detail={
            "code": CODE,
            "layout": layout,
            "viewport": viewport,
            "count": len(cut),
            "items": [{"handle": h, "type": k, "text": t[:60]} for h, k, t in cut[:MAX_ITEMS]],
            "repairable": False,
        },
    )


async def check_viewport_clip(backend: AutoCADBackend, shared: dict) -> list[Issue]:
    """The focus: every model viewport of every paper layout."""
    from .critique import scan_incomplete_issue
    from .qa_scan import _async_readonly, qa_snapshot

    snap, coverage = await qa_snapshot(backend, shared)
    ledger = ledger_of(shared)
    if snap is None or not coverage.complete:
        ledger.not_checked(FOCUS, DOCUMENT, "scan_incomplete")
        return [scan_incomplete_issue(FOCUS, coverage)]
    papers = [name for name in snap.layouts if name != "Model"]
    if not papers:
        domain_row(shared, FOCUS, present=False, n=0)
        return []
    found = await _async_readonly(backend, lambda: _scan(backend._require_doc(), snap.records))
    issues: list[Issue] = []
    for layout in papers:
        row = found.get(layout, {"reason": None, "n": 0, "cuts": []})
        if row["reason"]:
            ledger.not_checked(FOCUS, layout, row["reason"])
        else:
            ledger.checked(FOCUS, layout, row["n"])
        issues.extend(_issue(layout, vp, cut) for vp, cut in row["cuts"])
    return issues


check_viewport_clip.needs_shared = True  # type: ignore[attr-defined]
